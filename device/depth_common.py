#!/usr/bin/env python3
"""Shared depth-model plumbing: load a .tflite, run it on CPU or NPU, get a map.

Every script in this project goes through `DepthModel`, so the benchmark, the
accuracy check and the live demo are provably running the same arithmetic on the
same tensors, and only the backend differs.

Three things here are less obvious than they look:

* **Input layout is not fixed.** Depending on how a model reached AI Hub, the
  compiled graph wants NCHW `(1,3,H,W)` or NHWC `(1,H,W,3)`. Read it from the
  interpreter rather than assuming.
* **Preprocessing is a plain resize, not a letterbox.** Ultralytics letterboxes
  internally, but every accuracy number this project quotes was produced with a
  straight `INTER_AREA` resize, so the runner has to match that or the numbers
  stop being comparable.
* **The models disagree about what "depth" means.** Depth Anything emits
  relative inverse depth (bigger = nearer); YOLO26-depth emits metres (bigger =
  farther). `to_nearness()` is where that gets reconciled, and it is the only
  place in the codebase that knows the difference.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

DELEGATE_LIB = "/usr/lib/libQnnTFLiteDelegate.so"

# The single most important NPU setting on the VENTUNO Q. Left at the delegate's
# default the Hexagon runs at low clocks: Depth Anything V2 Small at 336x252 took
# 58.3 ms per frame. "Sustained high performance" took 12.7 ms, 4.6x faster, and
# is the mode meant for continuous load (burst, "2", was only 2% faster). The
# value is QNN's HtpPerformanceMode enum, passed as a numeric string.
SUSTAINED_HIGH_PERFORMANCE = "1"

# Which way each model's output runs. These three do not agree, and the
# disagreement is invisible until a scene comes out inside out:
#
#   dav2   relative INVERSE depth  bigger = nearer   (verified against NYU)
#   da3    relative depth          bigger = farther  (verified against NYU;
#                                   opposite to V2 despite the shared name, and
#                                   NOT metric: as-predicted delta1 is 0.001)
#   yolo26 metric metres           bigger = farther
#
# Matched on the filename because that is all a .tflite carries, so anything
# unrecognised raises rather than defaulting to a coin flip.
#
# `metric` is a separate question from orientation: it records whether the model
# claims its numbers are metres. Only YOLO26 does. Scoring a scale-free model
# against absolute ground truth produces a delta1 near zero that says nothing
# about the model, so the scorer needs to know which claim to test.
ORIENTATION = {"dav2": True, "da3": False, "yolo": False}
METRIC_CLAIM = {"dav2": False, "da3": False, "yolo": True}


def _family(model_path: str) -> str:
    stem = model_path.rsplit("/", 1)[-1].lower()
    for key in ORIENTATION:
        if key in stem:
            return key
    raise ValueError(
        f"cannot tell which way depth runs for '{stem}'. Name the file so it "
        f"contains one of {sorted(ORIENTATION)}, or pass inverse= explicitly.")


def infer_orientation(model_path: str) -> bool:
    """True if the model emits inverse depth (bigger = nearer)."""
    return ORIENTATION[_family(model_path)]


def claims_metric(model_path: str) -> bool:
    """True if the model claims its output is in metres."""
    return METRIC_CLAIM[_family(model_path)]


class DepthModel:
    """One .tflite, on one backend."""

    def __init__(self, model_path: str, backend: str = "cpu", threads: int = 8,
                 htp_precision: str | None = None, inverse: bool | None = None,
                 performance_mode: str | None = SUSTAINED_HIGH_PERFORMANCE,
                 cache_dir: str | None = None):
        self.model_path = model_path
        self.backend = backend

        if backend == "cpu":
            self.interp = Interpreter(model_path=model_path, num_threads=threads)
        else:
            from ai_edge_litert.interpreter import load_delegate

            # Every option value is parsed with stoi, so a word like "error"
            # aborts the process rather than raising. Pass numeric strings only,
            # or leave the option out.
            options = {"backend_type": backend}
            if htp_precision is not None:
                options["htp_precision"] = htp_precision
            if performance_mode is not None and backend == "htp":
                options["htp_performance_mode"] = performance_mode
            if cache_dir is not None and backend == "htp":
                # The prepared graph is cached on disk: preparing Depth Anything V2
                # Small at 392x294 took 13.2 s, loading the cached copy 0.4 s. The
                # token carries the file's size and timestamp, so a changed model
                # gets a fresh cache instead of someone else's graph.
                stat = Path(model_path).stat()
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                options["cache_dir"] = str(cache_dir)
                options["model_token"] = f"{Path(model_path).stem}_{stat.st_size}_{int(stat.st_mtime)}"
            # Graph preparation happens here and on the first invoke, not at
            # export time: expect seconds of `Starting stage:` logs for a float
            # model, well under one for a quantized one. Once per process.
            self.delegate = load_delegate(DELEGATE_LIB, options=options)
            self.interp = Interpreter(model_path=model_path,
                                      experimental_delegates=[self.delegate])

        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.outp = self.interp.get_output_details()[0]

        shape = [int(x) for x in self.inp["shape"]]
        self.nhwc = shape[3] == 3
        self.height, self.width = (shape[1], shape[2]) if self.nhwc else (shape[2], shape[3])

        if inverse is None:
            inverse = infer_orientation(model_path)
        self.inverse = inverse

    @property
    def layout(self) -> str:
        return "NHWC" if self.nhwc else "NCHW"

    @property
    def orientation(self) -> str:
        """Human-readable output convention. Deliberately does not say 'metres':
        only YOLO26 claims a metric scale, and that claim is measured, not
        assumed, elsewhere in this project."""
        return ("inverse depth, bigger = nearer" if self.inverse
                else "depth, bigger = farther")

    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        """BGR frame -> the tensor the graph wants, [0,1] RGB."""
        resized = cv2.resize(bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.expand_dims(rgb if self.nhwc else rgb.transpose(2, 0, 1), 0)
        return x.astype(self.inp["dtype"])

    def invoke(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Run one tensor. Returns the raw output map and the invoke time in ms."""
        self.interp.set_tensor(self.inp["index"], x)
        t0 = time.perf_counter()
        self.interp.invoke()
        ms = (time.perf_counter() - t0) * 1000
        return np.squeeze(self.interp.get_tensor(self.outp["index"])), ms

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        """Frame in, raw depth map out. Units are whatever the model emits."""
        return self.invoke(self.preprocess(bgr))[0]

    def to_nearness(self, raw: np.ndarray) -> np.ndarray:
        """Raw model output -> normalized nearness, 0-1, 1.0 = nearest.

        This is the only common ground the models have. Depth Anything's output
        is already nearness-shaped; YOLO26's metres have to be inverted first.
        """
        return normalize(raw if self.inverse else 1.0 / np.maximum(raw, 1e-6))


def normalize(x: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
    """Stretch [p2, p98] to [0, 1] and clip.

    Percentiles rather than min/max because a single hot pixel would otherwise
    compress the whole scene. Identical to the scoring script's normalizer, so
    what the demo displays is what the accuracy tables measured.
    """
    lo, hi = np.percentile(x, lo_pct), np.percentile(x, hi_pct)
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def colorize(nearness: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """Nearness map -> BGR image for display."""
    return cv2.applyColorMap((nearness * 255).astype(np.uint8), colormap)
