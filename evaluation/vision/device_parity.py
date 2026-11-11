"""Image backend parity and latency on generated, non-private images.

    python evaluation/vision/device_parity.py CHECKPOINT --device mps --device mlx \
        --out report-vision-parity.json

This checks implementation agreement, not visual accuracy. Each device loads and
unloads separately. Timing includes preprocessing and the CPU readout.
"""

from __future__ import annotations

import argparse
import base64
import gc
import io
import json
import math
import platform
import statistics
import time
from importlib.metadata import version

import torch
from PIL import Image, ImageDraw

from strands_decider.infer import EngineConfig
from strands_decider.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOneRequest
from strands_decider.vision import load_vision_engine


def image(width: int, height: int, red: bool) -> str:
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        (width // 8, height // 8, width // 2, height // 2), fill="red" if red else "blue"
    )
    draw.ellipse((width // 2, height // 2, width * 7 // 8, height * 7 // 8), fill="green")
    output = io.BytesIO()
    canvas.save(output, "PNG")
    return base64.b64encode(output.getvalue()).decode()


def requests() -> dict[str, SystemOneRequest]:
    questions = {
        "red": NoulQuestion(instructions="Is there a red rectangle in the image?"),
        "color": ChoiceQuestion(
            instructions="What color is the rectangle?",
            criteria={"red": "red", "blue": "blue", "other": "another color"},
        ),
        "clutter": ScoreQuestion(
            instructions="How visually cluttered is the image?",
            criteria=["simple", "busy", "dense"],
        ),
    }
    out = {}
    for red in [False, True]:
        for sizes in [[(192, 96)], [(448, 448)], [(600, 800), (1000, 200)]]:
            images = [image(w, h, red) for w, h in sizes]
            for multiple in [False, True]:
                name = f"{'red' if red else 'blue'}-{sizes}-q{3 if multiple else 1}"
                out[name] = SystemOneRequest(
                    state="Inspect the shapes in the supplied image(s).",
                    questions=questions if multiple else {"red": questions["red"]},
                    images=images,
                )
    out["text-after-images"] = SystemOneRequest(
        state="There is a blue rectangle and a green circle on a white background.",
        questions=questions,
    )
    return out


def run(checkpoint: str, device: str, reps: int) -> dict:
    started = time.perf_counter()
    engine = load_vision_engine(checkpoint, EngineConfig(device=device))
    loaded = time.perf_counter() - started
    responses, timings = {}, {}
    try:
        for name, request in requests().items():
            engine.evaluate(request)  # untimed shape warmup
            times = []
            for _ in range(reps):
                started = time.perf_counter()
                response = engine.evaluate(request)
                times.append((time.perf_counter() - started) * 1000)
            responses[name] = response.model_dump()
            timings[name] = {"median_ms": round(statistics.median(times), 2), "samples_ms": times}
            print(f"{device}: {name}: {timings[name]['median_ms']} ms", flush=True)
        return {
            "load_s": loaded,
            "base_model": engine.model.config.base_model,
            "base_revision": engine.model.config.base_revision,
            "dtype": engine.model.config.torch_dtype
            if device == "mlx"
            else str(next(engine.model.torso.parameters()).dtype).removeprefix("torch."),
            "responses": responses,
            "timings": timings,
        }
    finally:
        if hasattr(engine, "close"):
            engine.close()
        del engine
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()


def probabilities(answer: dict) -> dict:
    if answer["type"] == "noul":
        return {"true": answer["noul"], "false": 1 - answer["noul"]}
    return answer["probabilities"]


def compare(reference: dict, actual: dict) -> dict:
    largest, changed, total = 0.0, 0, 0
    for name, response in reference["responses"].items():
        other = actual["responses"][name]
        assert response["usage"] == other["usage"], f"{name}: token counts differ"
        for question, answer in response["answers"].items():
            p, q = probabilities(answer), probabilities(other["answers"][question])
            for side, values in [("reference", p), ("actual", q)]:
                if not all(math.isfinite(value) for value in values.values()):
                    raise ValueError(f"{name}/{question}: {side} probabilities must be finite")
            largest = max(largest, max(abs(p[k] - q[k]) for k in p))
            changed += max(p, key=p.get) != max(q, key=q.get)
            total += 1
    return {"max_abs_dp": largest, "changed_answers": changed, "answers": total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--device", action="append", required=True)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.reps < 1:
        parser.error("--reps must be positive")
    packages = ["torch", "transformers", "strands-decider"]
    if "mlx" in args.device:
        packages += ["mlx", "mlx-lm", "mlx-vlm"]
    results = {}
    report = {
        "checkpoint": args.checkpoint,
        "platform": platform.platform(),
        "versions": {p: version(p) for p in packages},
        "comparison": {},
        "devices": results,
    }
    for device in args.device:
        results[device] = run(args.checkpoint, device, args.reps)
        report["comparison"] = {
            d: compare(results[args.device[0]], results[d]) for d in list(results)[1:]
        }
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps(report["comparison"], indent=2))


if __name__ == "__main__":
    main()
