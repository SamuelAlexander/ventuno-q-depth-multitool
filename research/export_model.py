#!/usr/bin/env python3
"""Build the model this project runs: Depth Anything V2 Small, float, at a
336x252 input, compiled for the VENTUNO Q's NPU through Qualcomm AI Hub.

Runs on any computer; only the finished .tflite goes to the board. Needs a free
Qualcomm AI Hub account (`qai-hub configure --api_token <token>`) and:

    pip install "qai-hub-models[depth-anything-v2]"

    python research/export_model.py             # -> models/dav2s_float_336x252.tflite
    python research/export_model.py --profile   # also time it on AI Hub's hosted VENTUNO Q
"""

import argparse
from pathlib import Path

import qai_hub as hub
import torch
from qai_hub_models.models.depth_anything_v2.model import DepthAnythingV2
from qai_hub_models.utils.input_spec import make_torch_inputs

HEIGHT, WIDTH = 252, 336          # both multiples of 14, the ViT's patch size
# Chosen by chipset, not by name: AI Hub listed this chip as "QCS8275 (Proxy)"
# until it began hosting the real board as "Arduino VENTUNO Q".
DEVICE_CHIPSET = "chipset:qualcomm-qcs8275"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "models" / "dav2s_float_336x252.tflite"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--profile", action="store_true", help="also time it on AI Hub's device farm")
    args = p.parse_args()

    model = DepthAnythingV2.from_pretrained().to("cpu")
    spec = model.get_input_spec(height=HEIGHT, width=WIDTH)
    traced = torch.jit.trace(model, make_torch_inputs(spec), check_trace=False)

    device = hub.Device(attributes=DEVICE_CHIPSET)
    job = hub.submit_compile_job(model=traced, input_specs=spec, device=device,
                                 options="--target_runtime tflite")
    print(f"compiling on AI Hub, a few minutes: {job.url}", flush=True)
    job.wait()
    target = job.get_target_model()
    if target is None:
        raise SystemExit(f"compile failed: {job.get_status().message}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    target.download(str(args.out))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")

    if args.profile:
        summary = hub.submit_profile_job(model=target, device=device).download_profile()
        ms = summary["execution_summary"]["estimated_inference_time"] / 1000
        # The farm profiles a lower-level runtime than the TFLite delegate used
        # here, so expect the board itself to be slower than this figure.
        print(f"AI Hub, {DEVICE_CHIPSET}: {ms:.1f} ms per frame on the farm's runtime")


if __name__ == "__main__":
    main()
