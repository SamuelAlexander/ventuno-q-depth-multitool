"""Four ways to consume one depth map.

A depth model does not produce a picture, it produces a field, and how much its
accuracy matters depends entirely on what you do with that field. These four
modes stress it differently, which is the point of shipping all four:

  heatmap  every pixel, continuous     forgiving: smooth errors look fine
  slicer   a thin plane through depth  brutal: needs correct fine structure
  flood    one threshold               forgiving: only the level line matters
  sonar    left/right occupancy        coarse: only regional averages matter

The slicer is the mode that separates these models. A depth map that renders a
pleasant heatmap can still cut a slice through the wrong half of a bookshelf.

Every mode takes `depth` as float32 nearness in 0-1, where 1.0 is nearest, and
returns a BGR image the same size as the frame.
"""

import time
from abc import ABC, abstractmethod

import cv2
import numpy as np

COLORMAPS = {
    "inferno": cv2.COLORMAP_INFERNO,
    "turbo": cv2.COLORMAP_TURBO,
    "magma": cv2.COLORMAP_MAGMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "gray": None,
}
COLORMAP_NAMES = list(COLORMAPS)


def colorize(nearness: np.ndarray, colormap: str = "inferno") -> np.ndarray:
    u8 = (nearness * 255).astype(np.uint8)
    cmap = COLORMAPS[colormap]
    if cmap is None:
        return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    return cv2.applyColorMap(u8, cmap)


def dimmed_gray(frame: np.ndarray, factor: float = 0.25) -> np.ndarray:
    """Desaturated, darkened base so highlighted geometry reads clearly."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor((gray * factor).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def mask_edge(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Boundary pixels of a boolean mask, via a morphological gradient."""
    m = mask.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness * 2 + 1,) * 2)
    return cv2.morphologyEx(m, cv2.MORPH_GRADIENT, k) > 0


class Param:
    """An adjustable value with clamping and display formatting."""

    def __init__(self, value: float, lo: float, hi: float, step: float,
                 fmt: str = "{:.2f}"):
        self.value = value
        self.lo, self.hi, self.step, self.fmt = lo, hi, step, fmt

    def nudge(self, delta: int) -> None:
        self.value = min(self.hi, max(self.lo, self.value + delta * self.step))

    def __format__(self, _spec: str) -> str:
        return self.fmt.format(self.value)

    def __float__(self) -> float:
        return float(self.value)


class DepthMode(ABC):
    name = "mode"
    main_label = "param"
    alt_label = "alt"
    default_colormap = "inferno"
    uses_audio = False

    def __init__(self):
        self.colormap = self.default_colormap

    @abstractmethod
    def adjust(self, delta: int, alt: bool) -> None:
        """Arrow keys. `alt` selects the secondary parameter."""

    @abstractmethod
    def render(self, frame: np.ndarray, depth: np.ndarray) -> np.ndarray:
        ...

    def levels(self, depth: np.ndarray) -> tuple[float, float]:
        """Left/right intensity 0-1, for modes that drive audio."""
        return (0.0, 0.0)

    @abstractmethod
    def status(self) -> str:
        ...


class HeatmapMode(DepthMode):
    """The classic view, with a continuous blend from camera to depth."""

    name = "heatmap"
    main_label = "blend"
    alt_label = "colormap"

    def __init__(self):
        super().__init__()
        self.blend = Param(0.75, 0.0, 1.0, 0.05)

    def adjust(self, delta: int, alt: bool) -> None:
        if alt:
            i = (COLORMAP_NAMES.index(self.colormap) + delta) % len(COLORMAP_NAMES)
            self.colormap = COLORMAP_NAMES[i]
        else:
            self.blend.nudge(delta)

    def render(self, frame: np.ndarray, depth: np.ndarray) -> np.ndarray:
        depth_bgr = colorize(depth, self.colormap)
        a = float(self.blend)
        if a >= 0.999:
            return depth_bgr
        if a <= 0.001:
            return frame.copy()
        return cv2.addWeighted(depth_bgr, a, frame, 1 - a, 0)

    def status(self) -> str:
        return f"blend {self.blend}  cmap {self.colormap}"


class SlicerMode(DepthMode):
    """An MRI-style plane swept through the room.

    Only geometry within a band around the plane is lit. This is the mode that
    exposes a weak depth model: it has to place a thin surface correctly, so
    smeared fine structure shows up immediately as a slice that grabs a whole
    wall instead of one shelf.
    """

    name = "slicer"
    main_label = "plane"
    alt_label = "band"
    default_colormap = "turbo"

    def __init__(self):
        super().__init__()
        self.plane = Param(0.5, 0.0, 1.0, 0.02)
        self.band = Param(0.12, 0.02, 0.40, 0.01)

    def adjust(self, delta: int, alt: bool) -> None:
        (self.band if alt else self.plane).nudge(delta)

    def render(self, frame: np.ndarray, depth: np.ndarray) -> np.ndarray:
        plane, band = float(self.plane), float(self.band)
        inside = np.abs(depth - plane) < band / 2

        out = dimmed_gray(frame)
        lit = colorize(depth, self.colormap)
        # cv2.copyTo is the SIMD equivalent of np.copyto(..., where=mask[:,:,None])
        # and roughly halves this mode's render cost.
        cv2.copyTo(lit, inside.astype(np.uint8), out)
        out[mask_edge(inside)] = (255, 255, 255)

        self._draw_gauge(out, plane, band)
        return out

    @staticmethod
    def _draw_gauge(img: np.ndarray, plane: float, band: float) -> None:
        h, w = img.shape[:2]
        x, top, bot = w - 28, int(h * 0.15), int(h * 0.85)
        cv2.rectangle(img, (x - 6, top - 6), (x + 6, bot + 6), (40, 40, 40), -1)
        cv2.line(img, (x, top), (x, bot), (150, 150, 150), 1, cv2.LINE_AA)

        def y_of(v: float) -> int:  # 1.0 (near) at the bottom
            return int(bot - v * (bot - top))

        y0, y1 = y_of(plane + band / 2), y_of(plane - band / 2)
        cv2.rectangle(img, (x - 5, y0), (x + 5, y1), (0, 200, 255), -1)
        cv2.line(img, (x - 8, y_of(plane)), (x + 8, y_of(plane)), (255, 255, 255),
                 2, cv2.LINE_AA)
        cv2.putText(img, "near", (x - 44, bot + 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(img, "far", (x - 34, top - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1, cv2.LINE_AA)

    def status(self) -> str:
        return f"plane {self.plane}  band {self.band}"


class FloodMode(DepthMode):
    """Virtual water rising through the room.

    Water floods the far field first and creeps toward the camera, so near
    objects stay dry as islands the longest.

    Blending runs in uint8 through cv2's SIMD paths rather than float numpy: at
    panel resolution the naive float version costs more than half the frame
    budget on top of inference.
    """

    name = "flood"
    main_label = "level"
    alt_label = "opacity"

    WATER_BGR = (200, 120, 20)
    ATTENUATION = 0.45
    RIPPLE_AMPLITUDE = 0.12

    def __init__(self):
        super().__init__()
        self.level = Param(0.45, 0.0, 1.0, 0.02)
        self.opacity = Param(0.65, 0.1, 0.95, 0.05)
        self._t0 = time.perf_counter()
        self._shape = None

    def adjust(self, delta: int, alt: bool) -> None:
        (self.opacity if alt else self.level).nudge(delta)

    def _ensure_buffers(self, shape: tuple[int, int]) -> None:
        if self._shape == shape:
            return
        h, w = shape
        self._shape = shape
        self._full = (w, h)
        self._half = (w // 2, h // 2)
        self._rows = np.arange(h // 2, dtype=np.float32)[:, None]
        self._water = np.full((*shape, 3), self.WATER_BGR, dtype=np.uint8)

    def render(self, frame: np.ndarray, depth: np.ndarray) -> np.ndarray:
        level = float(self.level)
        if level <= 0.001:
            return frame.copy()
        self._ensure_buffers(depth.shape)

        submerged = depth < level
        if not submerged.any():
            return frame.copy()

        # Tint and attenuation are smooth fields, so they are computed at half
        # resolution and upscaled: 4x less float math for no visible loss. The
        # hard waterline is restored afterwards from the full-res mask.
        d_half = cv2.resize(depth, self._half, interpolation=cv2.INTER_AREA)
        below = np.clip((level - d_half) * (1.0 / level), 0.0, 1.0)

        t = time.perf_counter() - self._t0
        ripple = self.RIPPLE_AMPLITUDE * np.sin(self._rows * 0.18 + t * 2.2)

        alpha = float(self.opacity) * (0.45 + 0.55 * below) + ripple
        np.clip(alpha, 0.0, 1.0, out=alpha)
        a8 = cv2.resize((alpha * 255).astype(np.uint8), self._full,
                        interpolation=cv2.INTER_LINEAR)
        a8 = cv2.bitwise_and(a8, submerged.astype(np.uint8) * 255)
        a3 = cv2.cvtColor(a8, cv2.COLOR_GRAY2BGR)

        atten = cv2.resize(((1.0 - self.ATTENUATION * below) * 255).astype(np.uint8),
                           self._full, interpolation=cv2.INTER_LINEAR)
        lit = cv2.multiply(frame, cv2.cvtColor(atten, cv2.COLOR_GRAY2BGR), scale=1 / 255)

        out = cv2.add(cv2.multiply(lit, cv2.bitwise_not(a3), scale=1 / 255),
                      cv2.multiply(self._water, a3, scale=1 / 255))
        out[mask_edge(submerged, thickness=1)] = (255, 250, 235)

        self._draw_level_bar(out, level)
        return out

    def _draw_level_bar(self, img: np.ndarray, level: float) -> None:
        h = img.shape[0]
        x, top, bot = 18, int(h * 0.15), int(h * 0.85)
        y = int(bot - level * (bot - top))
        cv2.line(img, (x, top), (x, bot), (120, 120, 120), 1, cv2.LINE_AA)
        cv2.rectangle(img, (x - 4, y), (x + 4, bot), self.WATER_BGR, -1)
        cv2.line(img, (x - 9, y), (x + 9, y), (255, 250, 235), 2, cv2.LINE_AA)

    def status(self) -> str:
        return f"level {self.level}  opacity {self.opacity}"


class SonarMode(DepthMode):
    """Stereo proximity: how blocked is the left half, how blocked is the right.

    Each side drives one click train in `sonar_audio`, low pitch for left and
    high for right, rising in rate as that side fills.

    Percentile normalization rescales every frame so its nearest content maps to
    1.0, which makes absolute proximity unreadable. So the signal here is
    deliberately scale-free: the fraction of each side occupied by near-band
    content. It answers "which way is more open", not "how many metres".

    Only the central band is sampled. Floor and ceiling are always near and
    would otherwise pin both channels permanently.
    """

    name = "sonar"
    main_label = "gain"
    alt_label = "near"
    uses_audio = True

    ROI_TOP, ROI_BOTTOM = 0.20, 0.80
    GAMMA = 2.0
    SMOOTHING = 0.3

    def __init__(self):
        super().__init__()
        self.gain = Param(1.5, 0.5, 4.0, 0.1)
        self.near = Param(0.65, 0.30, 0.95, 0.05)
        self._level = [0.0, 0.0]
        self._occupancy = (0.0, 0.0)
        self._cache_key = None

    def adjust(self, delta: int, alt: bool) -> None:
        (self.near if alt else self.gain).nudge(delta)

    def _roi(self, depth: np.ndarray) -> np.ndarray:
        h = depth.shape[0]
        return depth[int(h * self.ROI_TOP):int(h * self.ROI_BOTTOM), :]

    def _compute(self, depth: np.ndarray) -> tuple[float, float]:
        if self._cache_key is depth:
            return self._occupancy
        roi = self._roi(depth)
        near = roi > float(self.near)
        w = near.shape[1]
        self._occupancy = (float(near[:, : w // 2].mean()),
                           float(near[:, w // 2:].mean()))
        self._cache_key = depth
        return self._occupancy

    def levels(self, depth: np.ndarray) -> tuple[float, float]:
        gain = float(self.gain)
        for i, occupancy in enumerate(self._compute(depth)):
            target = min(occupancy * gain, 1.0) ** self.GAMMA
            self._level[i] = ((1 - self.SMOOTHING) * self._level[i]
                              + self.SMOOTHING * target)
        return (self._level[0], self._level[1])

    def render(self, frame: np.ndarray, depth: np.ndarray) -> np.ndarray:
        out = colorize(depth, self.colormap)
        out = cv2.addWeighted(out, 0.75, frame, 0.25, 0)

        h, w = out.shape[:2]
        top, bot = int(h * self.ROI_TOP), int(h * self.ROI_BOTTOM)

        # Show exactly which pixels are driving the audio: the mode explains
        # itself, and tuning `near` becomes a visual operation.
        counted = np.zeros((h, w), np.uint8)
        counted[top:bot] = (self._roi(depth) > float(self.near)).astype(np.uint8)
        cv2.copyTo(cv2.addWeighted(out, 0.55, np.full_like(out, (255, 255, 255)),
                                   0.45, 0), counted, out)

        cv2.rectangle(out, (0, top), (w - 1, bot), (255, 255, 255), 1)
        cv2.line(out, (w // 2, top), (w // 2, bot), (255, 255, 255), 1)

        # Labelled with the pitch each side plays at, because the panel's single
        # speaker cannot carry left and right: the side is heard as a pitch.
        left, right = self._compute(depth)
        for label, value, x in (("LEFT  low", left, w // 4), ("RIGHT  high", right, 3 * w // 4)):
            text = f"{label}  {value * 100:.0f}%"
            tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]
            for colour, thickness in (((0, 0, 0), 5), ((255, 255, 255), 2)):
                cv2.putText(out, text, (x - tw // 2, bot - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, thickness,
                            cv2.LINE_AA)
        return out

    def status(self) -> str:
        l, r = self._level
        return f"gain {self.gain}  near {self.near}  out L{l:.2f} R{r:.2f}"


MODES = [HeatmapMode, SlicerMode, FloodMode, SonarMode]
