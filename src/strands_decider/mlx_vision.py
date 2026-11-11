"""The released Qwen3.5 decider's image path on Apple silicon.

Reuse the torch vision engine's pixels, tokenisation and limits, and the MLX
engine's adapter merge and fp32 readout. Only the multimodal forward moves to MLX.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import torch
from mlx.utils import tree_flatten
from mlx_lm.models.cache import make_prompt_cache
from mlx_vlm.utils import load_model
from transformers import AutoTokenizer, Qwen2VLImageProcessorPil

from .infer import EngineConfig
from .mlx_engine import _DTYPES, DEFAULT_CACHE_LIMIT, MLXEngine, _Readout, merge_lora
from .modeling import (
    StrandsDeciderConfig,
    base_revision,
    build_head,
    checkpoint_dir,
    config_path,
    load_head_state,
)
from .prompting import RenderedQuestion
from .schema import Content
from .vision import VisionEngine, VisionEngineConfig

if TYPE_CHECKING:
    from PIL import Image


class MLXVisionEngine(MLXEngine, VisionEngine):
    """MLX's worker/readout and text path, with VisionEngine's image API and limits.

    MLXEngine.evaluate dispatches VisionEngine.evaluate on its dedicated thread;
    text-only requests then take SystemOneEngine's unchanged question handling.
    """

    vlm: Any

    def _reset_positions(self) -> None:
        # Both forwards below pass positions explicitly; no rope state is retained.
        pass

    def _hidden(self, rows: list[list[int]], cache: list[Any] | None = None) -> Any:
        pad = self.tok.pad_token_id or 0
        width = max(map(len, rows))
        ids = mx.array([r + [pad] * (width - len(r)) for r in rows], dtype=mx.int32)
        if cache is None:
            cache = make_prompt_cache(self._cache_owner)
        offset = mx.array(cache[self._decoder.fa_idx].offset).reshape(-1, 1)
        positions = mx.broadcast_to(offset + mx.arange(width), (len(rows), width))
        positions = mx.broadcast_to(positions[None], (3, len(rows), width))
        hidden = self._decoder(ids, cache=cache, position_ids=positions)
        mx.eval(hidden)
        return hidden

    def _image_probs(
        self, state: Content, images: list[Image.Image], rendered: list[RenderedQuestion]
    ) -> tuple[torch.Tensor, int]:
        s, q, pixels, offsets = self._fit_images(state, images, [rq.text for rq in rendered])
        ids = mx.array([s], dtype=mx.int32)
        features = self.vlm.get_input_embeddings(
            input_ids=ids,
            pixel_values=mx.array(pixels["pixel_values"].numpy()),
            image_grid_thw=mx.array(pixels["image_grid_thw"].numpy()),
        )
        n, batch, width = len(s), len(q), max(map(len, q))
        pad = self.tok.pad_token_id or 0
        suffix = mx.array([r + [pad] * (width - len(r)) for r in q], dtype=mx.int32)
        # Image patches occupy fewer rotary positions than their number of tokens.
        suffix_pos = mx.arange(n, n + width) + features.rope_deltas.reshape(-1, 1)
        suffix_pos = mx.broadcast_to(suffix_pos[None], (3, batch, width))
        cache = make_prompt_cache(self._cache_owner)
        shared = self.cfg.use_prefix_cache and batch > 1
        if shared:
            prefix = self._decoder(
                ids,
                inputs_embeds=features.inputs_embeds,
                position_ids=features.position_ids,
                cache=cache,
            )
            mx.eval(prefix)
            cache = [type(layer).merge([layer] * batch) for layer in cache]
            hidden = self._decoder(suffix, position_ids=suffix_pos, cache=cache)
            base = 0
            tokens = n + sum(map(len, q))
        else:
            prefix_embeds = mx.broadcast_to(
                features.inputs_embeds, (batch, n, features.inputs_embeds.shape[-1])
            )
            embeds = mx.concatenate([prefix_embeds, self._decoder.embed_tokens(suffix)], axis=1)
            positions = mx.concatenate(
                [mx.broadcast_to(features.position_ids, (3, batch, n)), suffix_pos], axis=-1
            )
            full_ids = mx.concatenate([mx.broadcast_to(ids, (batch, n)), suffix], axis=1)
            hidden = self._decoder(
                full_ids, inputs_embeds=embeds, position_ids=positions, cache=cache
            )
            base = n
            tokens = batch * n + sum(map(len, q))
        mx.eval(hidden)
        return self._probs(
            hidden,
            [base + len(row) - 1 for row in q],
            self._option_idx(rendered, base, offsets),
            [rq.n_slots for rq in rendered],
            [rq.kind for rq in rendered],
        ), tokens


def load_mlx_vision_engine(
    checkpoint: str, config: EngineConfig | None = None, *, image_long_side: int = 448
) -> MLXVisionEngine:
    """Load and run on one Metal-owning thread, as the text MLX engine does."""
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-vision")
    try:
        return executor.submit(_load, checkpoint, config, image_long_side, executor).result()
    except BaseException:
        executor.shutdown(wait=True)
        raise


def _load(
    checkpoint: str,
    config: EngineConfig | None,
    image_long_side: int,
    executor: ThreadPoolExecutor,
) -> MLXVisionEngine:
    path = checkpoint_dir(checkpoint)
    cfg = StrandsDeciderConfig.from_json(config_path(path))
    cfg.base_revision = base_revision(path, cfg)
    base = Path(checkpoint_dir(cfg.base_model, cfg.base_revision))
    with open(base / "config.json", encoding="utf-8") as fh:
        raw = json.load(fh)
    if raw.get("model_type") != "qwen3_5":
        raise ValueError("MLX image input needs a multimodal qwen3_5 checkpoint")
    if raw.get("quantization") or raw.get("quantization_config"):
        raise ValueError("the MLX vision backend needs unquantised base weights")
    if cfg.torch_dtype not in _DTYPES:
        raise ValueError(f"the MLX backend has no dtype {cfg.torch_dtype!r}")
    mx.set_cache_limit(DEFAULT_CACHE_LIMIT)
    vlm = load_model(base)
    dtype = _DTYPES[cfg.torch_dtype]
    keep = vlm.cast_predicate
    vlm.load_weights(
        [
            (name, value.astype(dtype))
            for name, value in tree_flatten(vlm.parameters())
            if mx.issubdtype(value.dtype, mx.floating) and value.dtype != dtype and keep(name)
        ],
        strict=False,
    )
    if cfg.use_lora:
        merge_lora(vlm, os.path.join(path, "lora"), "language_model.model.")
    mx.eval(vlm.parameters())
    mx.clear_cache()
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    decoder = vlm.language_model.model
    head = build_head(cfg, int(decoder.embed_tokens.weight.shape[1]))
    head.load_state_dict(load_head_state(path))
    head.to(torch.float32).eval()
    names = {f.name for f in fields(EngineConfig)}
    base_config = replace(config or EngineConfig(), device="mlx")
    vcfg = VisionEngineConfig(
        **{k: v for k, v in asdict(base_config).items() if k in names},
        image_long_side=image_long_side,
    )
    engine = MLXVisionEngine(decoder, vlm.language_model, _Readout(cfg, tok, head), vcfg, executor)
    engine.vlm = vlm
    engine.vcfg = vcfg
    engine.image_processor = Qwen2VLImageProcessorPil.from_pretrained(
        cfg.base_model, revision=cfg.base_revision
    )
    return engine
