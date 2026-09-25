#!/usr/bin/env python3
"""Depth Multitool: live monocular depth on the Arduino VENTUNO Q.

One USB webcam, Depth Anything V2 Small on the board's NPU, and four tools for
reading the depth it estimates. Fullscreen on the board's own display, driven
entirely from the keyboard.

    ./venv/bin/python device/depth_multitool.py
    ./venv/bin/python device/depth_multitool.py --windowed
    ./venv/bin/python device/depth_multitool.py --video in.mp4 --out out.mp4 --headless

Keys
    1 2 3 4      heatmap, slicer, flood, sonar     Tab           next tool
    Up / Down    the tool's main setting           Left / Right  its second setting
    F            mirror (camera facing you)        V             flip upside down
    S            sonar sound on / off              Space         freeze the frame
    P            save a snapshot                   H             hide the bars
    Q / Esc      quit
"""

import argparse
import json
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from depth_common import DepthModel
from modes import MODES
from sonar_audio import SonarAudio

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "dav2s_float_336x252.tflite"
SETTINGS = Path.home() / ".config" / "depth-multitool" / "settings.json"
NPU_CACHE = Path.home() / ".cache" / "depth-multitool" / "npu"
SNAPSHOTS = Path.home() / "Pictures" / "Depth Multitool"

WINDOW = "Depth Multitool"
VIEW_W, VIEW_H = 1280, 720          # the camera image, and what every tool renders
BAR_H = 40                           # 720 + two 40 px bars = the 10" panel's 800 rows
CANVAS_W, CANVAS_H = VIEW_W, VIEW_H + 2 * BAR_H

FONT = cv2.FONT_HERSHEY_SIMPLEX
BAR_BG = (24, 24, 24)
TEXT = (235, 235, 235)
DIM = (135, 135, 135)
ACCENT = (143, 135, 0)              # Arduino teal, in BGR

# X keysyms, as cv2.waitKeyEx reports them through OpenCV's Qt backend on the
# board. Using full codes matters: the old `waitKey() & 0xFF` folded the arrows
# onto 81-84, which are also the codes for capital Q, R, S and T.
KEY_LEFT, KEY_UP, KEY_RIGHT, KEY_DOWN = 65361, 65362, 65363, 65364
KEY_TAB, KEY_ESC = 9, 27


class Fps:
    """Rolling median of per-frame times. Median, not mean, so one stall does
    not poison the readout for the next few seconds."""

    def __init__(self, window: int = 30):
        self.samples = deque(maxlen=window)

    def add(self, ms: float) -> None:
        self.samples.append(ms)

    @property
    def ms(self) -> float:
        return float(np.median(self.samples)) if self.samples else 0.0

    @property
    def fps(self) -> float:
        return 1000.0 / self.ms if self.ms > 0 else 0.0


class SteadyNormalizer:
    """Stretch each depth map to 0-1, with the stretch smoothed over time.

    The model's output scale is arbitrary per frame, so every frame is mapped to
    nearness in 0-1 before display. Mapping each frame to its own range makes the
    picture pulse as the extremes jump around; an EMA on the range removes the
    pulse while still following a real scene change within about half a second.
    """

    def __init__(self, inverse: bool, alpha: float = 0.15):
        self.inverse, self.alpha = inverse, alpha
        self.lo = self.hi = None

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        near = raw if self.inverse else 1.0 / np.maximum(raw, 1e-6)
        lo, hi = np.percentile(near[::2, ::2], (2.0, 98.0))
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        if self.lo is None:
            self.lo, self.hi = lo, hi
        else:
            self.lo += self.alpha * (lo - self.lo)
            self.hi += self.alpha * (hi - self.hi)
        return np.clip((near - self.lo) / (self.hi - self.lo), 0.0, 1.0).astype(np.float32)


class LatestFrame:
    """Reads the camera on its own thread and keeps only the newest frame.

    Two wins over reading inline: the MJPG decode overlaps with inference instead
    of adding to it, and the frame processed is always the freshest one rather
    than the oldest one queued in the driver's buffer, which is what makes the
    view feel attached to the camera.
    """

    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.frame, self.seq, self.alive = None, 0, True
        self.cond = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            with self.cond:
                if not ok:
                    self.alive = False
                    self.cond.notify_all()
                    return
                self.frame, self.seq = frame, self.seq + 1
                self.cond.notify_all()

    def next(self, after: int, timeout: float = 2.0):
        """The newest frame with a sequence number above `after`, or None if the
        camera stopped delivering."""
        with self.cond:
            self.cond.wait_for(lambda: self.seq > after or not self.alive, timeout)
            if self.seq <= after:
                return None, after
            return self.frame, self.seq

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


class DepthWorker:
    """Runs the model on the newest camera frame, continuously, on its own thread.

    The NPU call releases Python's GIL and leaves the CPU idle for ~56 ms a
    frame. Drawing frame N on the main thread while the NPU works on frame N+1
    overlaps the two instead of adding them, which is worth a fifth more frames
    per second on the board and makes the heavy tools about as fast as the light
    ones. Each published result pairs a frame with its own depth map, so what is
    drawn is always consistent, never a new image over an old depth.
    """

    def __init__(self, grabber: "LatestFrame", process, max_fps: float = 0):
        self.grabber, self.process = grabber, process
        self.period = 1.0 / max_fps if max_fps > 0 else 0.0
        self.result, self.count, self.alive = None, 0, True
        self.paused = threading.Event()
        self.cond = threading.Condition()
        self._last = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        seq = 0
        while not self._stop.is_set():
            if self.paused.is_set():          # frozen: no point spending the NPU
                time.sleep(0.03)
                continue
            if self.period:
                # Capped: wait out the rest of the period so the NPU and CPU idle
                # between frames instead of running flat out.
                wait = self._last + self.period - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                self._last = time.perf_counter()
            frame, seq = self.grabber.next(seq)
            if frame is None:
                with self.cond:
                    self.alive = False
                    self.cond.notify_all()
                return
            result = self.process(frame)
            with self.cond:
                self.result, self.count = result, self.count + 1
                self.cond.notify_all()

    def next(self, after: int, timeout: float = 3.0):
        """The newest result numbered above `after`, or None if the camera died."""
        with self.cond:
            self.cond.wait_for(lambda: self.count > after or not self.alive, timeout)
            if self.count <= after:
                return None, after
            return self.result, self.count

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)


def load_settings() -> dict:
    settings = {"tool": "heatmap", "mirror": False, "flip": False, "sound": True}
    try:
        settings.update(json.loads(SETTINGS.read_text()))
    except (OSError, ValueError):
        pass
    return settings


def save_settings(settings: dict) -> None:
    try:
        SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS.write_text(json.dumps(settings, indent=2))
    except OSError:
        pass                         # a read-only home should not stop the demo


def orient(frame: np.ndarray, mirror: bool, flip: bool) -> np.ndarray:
    """Undo the camera's physical orientation before anything else sees the frame.

    This runs *before* inference on purpose. A mirror does not matter to the
    model, but an upside-down image does: depth models lean on gravity cues,
    floor below and ceiling above, and estimate an inverted scene badly.
    """
    if mirror and flip:
        return cv2.flip(frame, -1)
    if mirror:
        return cv2.flip(frame, 1)
    if flip:
        return cv2.flip(frame, 0)
    return frame


def put(img, text, org, scale=0.55, color=TEXT, thickness=1) -> None:
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def width_of(text, scale=0.55, thickness=1) -> int:
    return cv2.getTextSize(text, FONT, scale, thickness)[0][0]


def message_screen(title: str, lines: list[str]) -> np.ndarray:
    img = np.full((CANVAS_H, CANVAS_W, 3), 14, np.uint8)
    cy = CANVAS_H // 2 - 30
    put(img, title, ((CANVAS_W - width_of(title, 1.3, 2)) // 2, cy), 1.3, TEXT, 2)
    cv2.line(img, (CANVAS_W // 2 - 60, cy + 22), (CANVAS_W // 2 + 60, cy + 22), ACCENT, 3)
    for i, line in enumerate(lines):
        put(img, line, ((CANVAS_W - width_of(line, 0.65)) // 2, cy + 70 + 34 * i), 0.65, DIM)
    footer = "Arduino VENTUNO Q  |  Depth Anything V2 on the NPU"
    put(img, footer, ((CANVAS_W - width_of(footer, 0.5)) // 2, CANVAS_H - 40), 0.5, DIM)
    return img


def draw_top_bar(canvas, tools, active, stats: str, badges: list[str]) -> None:
    canvas[:BAR_H] = BAR_BG
    put(canvas, "DEPTH MULTITOOL", (16, 26), 0.55, TEXT, 2)

    x = 220
    for i, tool in enumerate(tools):
        label = f"{i + 1} {tool.name.upper()}"
        w = width_of(label, 0.55, 1)
        if i == active:
            cv2.rectangle(canvas, (x - 10, 7), (x + w + 10, BAR_H - 7), ACCENT, -1)
            put(canvas, label, (x, 26), 0.55, (255, 255, 255), 1)
        else:
            put(canvas, label, (x, 26), 0.55, DIM, 1)
        x += w + 30

    right = CANVAS_W - 16
    right -= width_of(stats, 0.55)
    put(canvas, stats, (right, 26), 0.55, TEXT)
    for badge in badges:
        w = width_of(badge, 0.45)
        right -= w + 26
        cv2.rectangle(canvas, (right - 8, 10), (right + w + 8, BAR_H - 10), (70, 70, 70), -1)
        put(canvas, badge, (right, 25), 0.45, TEXT)


def draw_bottom_bar(canvas, tool) -> None:
    y0 = CANVAS_H - BAR_H
    canvas[y0:] = BAR_BG
    base = y0 + 25

    # The two settings this tool exposes, with the keys that move them.
    x = 16
    for keys, label in (("UP/DOWN", tool.main_label), ("LEFT/RIGHT", tool.alt_label)):
        put(canvas, keys, (x, base), 0.5, TEXT, 1)
        x += width_of(keys, 0.5) + 8
        value = f"{label} {getattr(tool, label)}"
        put(canvas, value, (x, base), 0.5, DIM)
        x += width_of(value, 0.5) + 28

    hints = [("F", "mirror"), ("V", "flip"), ("S", "sound"), ("SPACE", "freeze"),
             ("P", "snapshot"), ("H", "hide"), ("Q", "quit")]
    total = sum(width_of(k, 0.5) + 6 + width_of(d, 0.5) + 20 for k, d in hints) - 20
    x = CANVAS_W - 16 - total
    for key, desc in hints:
        put(canvas, key, (x, base), 0.5, TEXT)
        x += width_of(key, 0.5) + 6
        put(canvas, desc, (x, base), 0.5, DIM)
        x += width_of(desc, 0.5) + 20


def draw_toast(canvas, text: str) -> None:
    w = width_of(text, 0.7, 2)
    x0, y0 = (CANVAS_W - w) // 2 - 18, BAR_H + VIEW_H - 74
    cv2.rectangle(canvas, (x0, y0), (x0 + w + 36, y0 + 44), (20, 20, 20), -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + w + 36, y0 + 44), ACCENT, 2)
    put(canvas, text, (x0 + 18, y0 + 30), 0.7, TEXT, 2)


def open_camera(index: int, width: int, height: int, fps: float = 0) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    # MJPG before the resolution, or a UVC camera negotiates raw YUYV and caps
    # near 10 FPS at 720p.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps > 0:
        # With a frame cap, have the camera send only that many frames rather
        # than decoding 30 a second and throwing most of them away.
        cap.set(cv2.CAP_PROP_FPS, fps)
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap


def main() -> None:
    # Hand the GIL over every 1 ms instead of Python's default 5 ms. Drawing the
    # flood tool is numpy-heavy and holds the lock in long stretches; at 5 ms the
    # inference thread queued behind it several times a frame, and preprocessing
    # took 21.7 ms instead of 5.6. At 1 ms every tool runs at the NPU's pace.
    sys.setswitchinterval(0.001)

    # Logout, shutdown and `kill` send SIGTERM. Turning it into a normal exit runs
    # the cleanup below (camera, audio) and reports success, which it is.
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: sys.exit(0))

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--backend", default="htp", choices=["cpu", "htp"],
                   help="htp is the NPU")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--video", help="a video file instead of the webcam")
    p.add_argument("--windowed", action="store_true", help="a window instead of fullscreen")
    p.add_argument("--headless", action="store_true", help="no window; use with --out")
    p.add_argument("--out", help="also write what is shown to this .mp4")
    p.add_argument("--tool", choices=[m.name for m in MODES],
                   help="start on this tool instead of the last one used")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames")
    # Defaults chosen for a battery: measured on the board's own power monitor,
    # the NPU's balanced mode at 20 FPS draws 12.0 W against 15.2 W for sustained
    # high performance uncapped at 30 FPS. Run with --fps 0 --npu-mode 1 for full speed.
    p.add_argument("--fps", type=float, default=20,
                   help="cap the frame rate (0 follows the camera); lower saves power")
    p.add_argument("--npu-mode", default="8",
                   help="QNN htp_performance_mode: 8 balanced, 1 sustained high performance")
    p.add_argument("--prepare", action="store_true",
                   help="prepare the model on the NPU and cache it, then exit")
    args = p.parse_args()

    if args.prepare:
        t0 = time.time()
        DepthModel(args.model, backend=args.backend, cache_dir=str(NPU_CACHE),
                   performance_mode=args.npu_mode)
        print(f"model prepared and cached in {time.time() - t0:.1f} s", flush=True)
        return

    settings = load_settings()
    if args.tool:
        settings["tool"] = args.tool
    show = not args.headless

    if show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(WINDOW, CANVAS_W, CANVAS_H)

    def pump(img: np.ndarray, seconds: float, until_key: bool = False) -> None:
        """Keep the window painted and answering the window manager."""
        end = time.time() + seconds
        while time.time() < end:
            cv2.imshow(WINDOW, img)
            if cv2.waitKeyEx(20) != -1 and until_key:
                return

    def show_message(title: str, lines: list[str], wait_s: float = 0.0) -> None:
        """Put a full-screen message up; optionally hold it until a key or timeout."""
        if not show:
            print(f"{title}: {' '.join(lines)}", flush=True)
            return
        pump(message_screen(title, lines), max(wait_s, 0.25), until_key=wait_s > 0)

    if not Path(args.model).exists():
        show_message("MODEL NOT FOUND",
                     [f"Expected {Path(args.model).name} in models/.",
                      "Run install.sh, then launch again.", "Press any key to close."], 30)
        raise SystemExit(1)

    # Preparing the graph on the NPU takes several seconds. It runs on a thread so
    # the splash stays painted and the window keeps answering GNOME, which flags
    # a window that goes quiet for ~5 s as "not responding".
    loaded: dict = {}

    def load() -> None:
        try:
            loaded["model"] = DepthModel(args.model, backend=args.backend,
                                         cache_dir=str(NPU_CACHE),
                                         performance_mode=args.npu_mode)
        except Exception as exc:          # re-raised on the main thread below
            loaded["error"] = exc

    loader = threading.Thread(target=load, daemon=True)
    loader.start()
    if show:
        started, dots, fullscreen_asked = time.time(), 0, False
        while loader.is_alive():
            dots = dots % 3 + 1
            splash = message_screen("DEPTH MULTITOOL",
                                    ["Preparing Depth Anything V2 on the NPU" + "." * dots])
            pump(splash, 0.35)
            # Qt drops a fullscreen request on a window the compositor has not
            # mapped yet: asked at once, the app opened as a 400x300 window in
            # the corner. Measured on the board, about a second is enough.
            if not args.windowed and not fullscreen_asked and time.time() - started > 1.0:
                cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                fullscreen_asked = True
        # A fast load can finish inside that first second; still give the window
        # its second before asking, then confirm the request took.
        if time.time() - started < 1.0:
            pump(splash, 1.0 - (time.time() - started))
        for _ in range(3):
            if args.windowed or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN) >= 1:
                break
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            pump(splash, 0.5)
    loader.join()
    if "error" in loaded:
        raise loaded["error"]
    model = loaded["model"]
    normalize = SteadyNormalizer(inverse=model.inverse)

    grabber, cap = None, None
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"cannot open {args.video}")
    else:
        cap = open_camera(args.camera, VIEW_W, VIEW_H, args.fps)
        if cap is None:
            show_message("NO CAMERA FOUND",
                         ["Plug a USB webcam into the VENTUNO Q and launch again.",
                          "Press any key to close."], 30)
            return
        grabber = LatestFrame(cap)

    def process(raw_frame: np.ndarray):
        """Camera frame in; (oriented frame, nearness map, NPU ms) out."""
        frame = orient(raw_frame, settings["mirror"], settings["flip"])
        if frame.shape[:2] != (VIEW_H, VIEW_W):
            frame = cv2.resize(frame, (VIEW_W, VIEW_H), interpolation=cv2.INTER_AREA)
        # The model's own invoke time, not the wrapper's, so the number on
        # screen means the same thing as the latency table in the docs.
        raw, invoke_ms = model.invoke(model.preprocess(frame))
        depth = cv2.resize(normalize(raw), (VIEW_W, VIEW_H), interpolation=cv2.INTER_LINEAR)
        return frame, depth, invoke_ms

    # A video file is processed synchronously so every frame is kept, in order;
    # the camera gets the pipelined worker, where skipping stale frames is the point.
    worker = DepthWorker(grabber, process, args.fps) if grabber is not None else None

    tools = [mode() for mode in MODES]          # one instance each, so settings survive switching
    names = [t.name for t in tools]
    active = names.index(settings["tool"]) if settings["tool"] in names else 0

    audio = SonarAudio(enabled=settings["sound"])
    audio.start()
    print(f"model {Path(args.model).name} on {args.backend}, sound: {audio.status}", flush=True)

    writer = None
    infer, loop = Fps(), Fps()
    frozen, show_bars, frames, count = False, True, 0, 0
    frame = depth = None
    last_shown = None
    toast, toast_until = "", 0.0

    def notify(text: str) -> None:
        nonlocal toast, toast_until
        toast, toast_until = text, time.time() + 1.3

    try:
        while True:
            tool = tools[active]

            if not frozen or frame is None:
                if worker is not None:
                    result, count = worker.next(count)
                else:
                    ok, raw_frame = cap.read()
                    result = process(raw_frame) if ok else None
                if result is None:
                    if args.video:
                        break
                    show_message("CAMERA DISCONNECTED",
                                 ["The webcam stopped sending frames.",
                                  "Reconnect it and launch again.",
                                  "Press any key to close."], 30)
                    break
                frame, depth, invoke_ms = result
                infer.add(invoke_ms)

                # Display rate is the interval between frames actually shown,
                # which is what a viewer experiences.
                now = time.perf_counter()
                if last_shown is not None:
                    loop.add((now - last_shown) * 1000)
                last_shown = now

            view = tool.render(frame, depth)

            if tool.uses_audio and settings["sound"]:
                audio.set_levels(*tool.levels(depth))
            else:
                audio.set_levels(0.0, 0.0)

            canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
            canvas[BAR_H:BAR_H + VIEW_H] = view
            if show_bars:
                badges = [b for b, on in (("SOUND OFF", tool.uses_audio and not settings["sound"]),
                                          ("FLIPPED", settings["flip"]),
                                          ("MIRRORED", settings["mirror"])) if on]
                stats = ("FROZEN" if frozen else
                         f"NPU {infer.ms:4.1f} ms   {loop.fps:4.1f} FPS")
                draw_top_bar(canvas, tools, active, stats, badges)
                draw_bottom_bar(canvas, tool)
            if time.time() < toast_until:
                draw_toast(canvas, toast)

            if args.out:
                if writer is None:
                    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                    # mp4v, not avc1: the board's OpenCV build has no H.264 encoder.
                    # Recorded at the source's own rate: the app keeps up with the
                    # camera, so anything else would play back too fast or too slow.
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps, (CANVAS_W, CANVAS_H))
                writer.write(canvas)

            frames += 1
            if args.max_frames and frames >= args.max_frames:
                break
            if not show:
                continue

            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKeyEx(30 if frozen else 1)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break                                   # closed with Alt+F4 or the window button
            if key == -1:
                continue
            if key in (ord("q"), ord("Q"), KEY_ESC):
                break
            elif ord("1") <= key <= ord(str(len(tools))):
                active = key - ord("1")
                settings["tool"] = tools[active].name
                save_settings(settings)
            elif key == KEY_TAB:
                active = (active + 1) % len(tools)
                settings["tool"] = tools[active].name
                save_settings(settings)
            elif key == KEY_UP:
                tool.adjust(+1, alt=False)
            elif key == KEY_DOWN:
                tool.adjust(-1, alt=False)
            elif key == KEY_RIGHT:
                tool.adjust(+1, alt=True)
            elif key == KEY_LEFT:
                tool.adjust(-1, alt=True)
            elif key in (ord("f"), ord("F")):
                settings["mirror"] = not settings["mirror"]
                save_settings(settings)
                notify("MIRROR ON" if settings["mirror"] else "MIRROR OFF")
            elif key in (ord("v"), ord("V")):
                settings["flip"] = not settings["flip"]
                save_settings(settings)
                notify("FLIPPED" if settings["flip"] else "UPRIGHT")
            elif key in (ord("s"), ord("S")):
                settings["sound"] = audio.toggle()
                save_settings(settings)
                notify("SONAR SOUND ON" if settings["sound"] else "SONAR SOUND OFF")
            elif key == ord(" "):
                frozen = not frozen
                if worker is not None:
                    (worker.paused.set if frozen else worker.paused.clear)()
                last_shown = None
                notify("FROZEN: tools still work" if frozen else "LIVE")
            elif key in (ord("p"), ord("P")):
                SNAPSHOTS.mkdir(parents=True, exist_ok=True)
                path = SNAPSHOTS / f"depth-multitool-{time.strftime('%Y%m%d-%H%M%S')}.png"
                cv2.imwrite(str(path), canvas)
                notify("SNAPSHOT SAVED")
            elif key in (ord("h"), ord("H")):
                show_bars = not show_bars
    finally:
        audio.close()
        if worker is not None:
            worker.stop()
        if grabber is not None:
            grabber.stop()
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    print(f"{frames} frames   NPU {infer.ms:.1f} ms ({infer.fps:.1f} FPS)   "
          f"end-to-end {loop.fps:.1f} FPS", flush=True)
    if args.out:
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
