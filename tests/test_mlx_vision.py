"""Real multimodal backend parity, without downloads or private screenshots."""

from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

if sys.platform != "darwin":
    pytest.skip("MLX runs on Apple silicon only", allow_module_level=True)
pytest.importorskip("mlx_vlm")

from test_vision import (
    QUESTIONS,
    STATE,
    _b64,
    base_dir,  # noqa: F401 - shared offline checkpoint fixtures
    ckpt,  # noqa: F401
)

from strands_decider.infer import EngineConfig
from strands_decider.modeling import StrandsDeciderConfig
from strands_decider.schema import SystemOneRequest
from strands_decider.vision import VisionEngineConfig, load_vision_engine


def _probabilities(response):
    return {
        name: [answer.noul] if answer.type == "noul" else list(answer.probabilities.values())
        for name, answer in response.answers.items()
    }


@pytest.fixture(scope="module")
def vision_engines(ckpt):  # noqa: F811 - pytest discovers the imported fixture
    torch = load_vision_engine(ckpt, EngineConfig(device="cpu"))
    mlx = load_vision_engine(ckpt, EngineConfig(device="mlx", model_name="mlx-vision-test"))
    yield torch, mlx
    mlx.close()


@pytest.mark.parametrize("sizes", [[(160, 96)], [(448, 448)], [(600, 800), (1000, 200)]])
@pytest.mark.parametrize("multiple_questions", [False, True])
def test_images_match_torch(vision_engines, sizes, multiple_questions):
    torch, mlx = vision_engines
    request = SystemOneRequest(
        state=STATE,
        questions=QUESTIONS if multiple_questions else {"signed": QUESTIONS["signed"]},
        images=[_b64(w, h, i) for i, (w, h) in enumerate(sizes)],
    )
    want, have = torch.evaluate(request), mlx.evaluate(request)
    assert have.usage == want.usage
    expected, actual = _probabilities(want), _probabilities(have)
    for name in expected:
        assert actual[name] == pytest.approx(expected[name], abs=2e-4), name


def test_image_and_text_requests_leave_no_state(vision_engines):
    torch, mlx = vision_engines
    text = SystemOneRequest(state=STATE, questions=QUESTIONS)
    before = _probabilities(mlx.evaluate(text))
    for dimensions in [(128, 256), (256, 128), (448, 448)]:
        mlx.evaluate(SystemOneRequest(state="", questions=QUESTIONS, images=[_b64(*dimensions, 9)]))
        assert _probabilities(mlx.evaluate(text)) == before
    expected = _probabilities(torch.evaluate(text))
    for name in expected:
        assert before[name] == pytest.approx(expected[name], abs=2e-4)


def test_images_change_hidden_states(vision_engines):
    _, mlx = vision_engines
    results = [
        _probabilities(
            mlx.evaluate(
                SystemOneRequest(state="", questions=QUESTIONS, images=[_b64(160, 160, seed)])
            )
        )
        for seed in [1, 2]
    ]
    assert results[0] != results[1], "the image must actually reach the decision model"


def test_shared_prefix_and_batching_match_full_forward(vision_engines):
    _, mlx = vision_engines
    request = SystemOneRequest(state=STATE, questions=QUESTIONS, images=[_b64(192, 128, 4)])
    expected = _probabilities(mlx.evaluate(request))
    original = mlx.cfg
    try:
        for changes in [{"max_batch": 1}, {"max_batch": 2}, {"use_prefix_cache": False}]:
            mlx.cfg = replace(original, **changes)
            actual = _probabilities(mlx.evaluate(request))
            for name in expected:
                assert actual[name] == pytest.approx(expected[name], abs=2e-4)
    finally:
        mlx.cfg = original


def test_mixed_requests_from_other_threads_match_serial(vision_engines):
    _, mlx = vision_engines
    requests = [
        SystemOneRequest(
            state=f"{STATE} {i}", questions=QUESTIONS, images=[_b64(128, 128, i)] if i % 2 else []
        )
        for i in range(4)
    ]
    expected = [_probabilities(mlx.evaluate(request)) for request in requests]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(lambda request: _probabilities(mlx.evaluate(request)), requests))
    assert actual == expected


def test_refuses_invalid_images_and_overlong_prompts(vision_engines):
    _, mlx = vision_engines
    for images, message in [(["not base64!"], "base64"), ([_b64(32, 32, 0)] * 5, "at most")]:
        with pytest.raises(ValueError, match=message):
            mlx.evaluate(SystemOneRequest(state="", questions=QUESTIONS, images=images))
    with pytest.raises(ValueError, match="context window"):
        mlx.evaluate(
            SystemOneRequest(state="too long " * 500, questions=QUESTIONS, images=[_b64(32, 32, 0)])
        )


def test_server_routes_and_closes_mlx_vision(ckpt):  # noqa: F811
    from fastapi.testclient import TestClient

    from strands_decider.server import create_app

    app = create_app(ckpt, device="mlx", vision=True, max_batch=2)
    with TestClient(app) as client:
        assert client.get("/health").json()["vision"] is True
        body = SystemOneRequest(
            state="", questions={"signed": QUESTIONS["signed"]}, images=[_b64(96, 96, 0)]
        ).model_dump()
        response = client.post("/v1/systemone", json=body)
        assert response.status_code == 200, response.text
        assert 0 <= response.json()["answers"]["signed"]["noul"] <= 1
    with pytest.raises(RuntimeError, match="shutdown"):
        app.state.engine.evaluate(SystemOneRequest(state=STATE, questions=QUESTIONS))


@pytest.mark.parametrize(
    ("base_config", "message"),
    [
        ({"model_type": "gemma4"}, "qwen3_5"),
        ({"model_type": "qwen3_5", "quantization": {"bits": 4}}, "unquantised"),
    ],
)
def test_invalid_base_fails_before_loading_and_closes_worker(tmp_path, base_config, message):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps(base_config))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "strands_decider_config.json").write_text(
        StrandsDeciderConfig(base_model=str(base)).to_json()
    )
    before = {thread.ident for thread in threading.enumerate()}
    with pytest.raises(ValueError, match=message):
        load_vision_engine(str(checkpoint), EngineConfig(device="mlx"))
    assert not [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith("mlx-vision") and thread.ident not in before
    ]


@pytest.mark.parametrize("image_long_side", [None, 96])
def test_load_preserves_engine_options_and_image_size(ckpt, image_long_side):  # noqa: F811
    config = VisionEngineConfig(
        device="mlx",
        model_name="custom",
        max_batch=2,
        use_prefix_cache=False,
        image_long_side=64,
        max_images=1,
        max_image_pixels=256 * 256,
    )
    engine = load_vision_engine(ckpt, config, image_long_side=image_long_side)
    try:
        assert engine.cfg.model_name == "custom"
        assert engine.cfg.max_batch == 2
        assert engine.cfg.use_prefix_cache is False
        assert engine.vcfg.image_long_side == (64 if image_long_side is None else image_long_side)
        request = SystemOneRequest(state="", questions=QUESTIONS, images=[_b64(192, 192, 0)])
        assert engine.evaluate(request).model == "custom"
        with pytest.raises(ValueError, match="at most 1"):
            engine.evaluate(request.model_copy(update={"images": [_b64(32, 32, 0)] * 2}))
        with pytest.raises(ValueError, match="pixels are accepted"):
            engine.evaluate(request.model_copy(update={"images": [_b64(257, 256, 0)]}))
    finally:
        engine.close()
