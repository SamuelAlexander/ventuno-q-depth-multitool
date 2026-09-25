"""Audible sonar for the VENTUNO Q.

Two click trains, one per side of the view, told apart by *pitch* rather than by
speaker. The 10" panel this was built on plays HDMI audio through a single
speaker, so a left/right pan is inaudible: low clicks mean something is close on
the left, high clicks on the right, and each train speeds up as its side fills.
The channels are still panned, so headphones get the stereo cue on top.

An open view is silent. That is deliberate: the useful sound is the one that
appears when something gets close, like a parking sensor.

Audio goes to `aplay` through PipeWire's ALSA plugin, with PIPEWIRE_NODE naming
the HDMI sink found by its description. That routes this one stream to the
panel without changing the system's default output. A generator thread keeps the
pipe fed so the frame loop never blocks on audio: a starved pipe underruns and
crackles, which would be indistinguishable from the sonar reacting.
"""

import json
import os
import shutil
import subprocess
import threading

import numpy as np

RATE = 44100
BLOCK = 441                     # 10 ms per write: small enough to feel immediate
PITCH_HZ = (520.0, 1040.0)      # left, right: an octave apart, easy to tell apart
CLICKS_PER_S = (3.0, 22.0)      # at the quietest audible level, and at full
CLICK_S = 0.016
SILENT_BELOW = 0.04             # levels under this make no sound at all
AMPLITUDE = 0.42                # per side; both trains at full sum to 0.84, no clipping


def find_hdmi_sink() -> str | None:
    """PipeWire node name of the HDMI output, or None to use the default.

    Matched on the description rather than a hardcoded node name, which encodes
    the ALSA card and device numbers and so differs between boards and images.
    """
    if not shutil.which("pw-dump"):
        return None
    try:
        dump = subprocess.run(["pw-dump"], capture_output=True, text=True,
                              timeout=5).stdout
        for obj in json.loads(dump):
            props = obj.get("info", {}).get("props", {})
            if (props.get("media.class") == "Audio/Sink"
                    and "HDMI" in props.get("node.description", "")):
                return props.get("node.name")
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return None


class SonarAudio:
    """Feeds `aplay` a stereo stream whose two click trains track two levels."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.status = "not started"
        self._levels = (0.0, 0.0)
        self._phase = [0.0, 0.0]        # position within each side's click period, 0-1
        self._sample = 0                # running sample count, keeps the oscillator continuous
        self._proc = None
        self._thread = None
        self._stop = threading.Event()

    def start(self) -> None:
        if not shutil.which("aplay"):
            self.status = "aplay not found, sound disabled"
            return
        env = dict(os.environ)
        sink = find_hdmi_sink()
        if sink:
            env["PIPEWIRE_NODE"] = sink
        try:
            self._proc = subprocess.Popen(
                ["aplay", "-q", "-D", "pipewire", "-t", "raw", "-f", "S16_LE",
                 "-r", str(RATE), "-c", "2", "--buffer-time=60000", "-"],
                stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
        except OSError as exc:
            self.status = f"could not start aplay ({exc})"
            return
        self.status = "HDMI" if sink else "default output"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_levels(self, left: float, right: float) -> None:
        self._levels = (float(left), float(right))

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled

    def _block(self) -> np.ndarray:
        n = BLOCK
        t = (self._sample + np.arange(n)) / RATE
        self._sample += n
        out = np.zeros((n, 2), np.float32)
        if not self.enabled:
            return out

        for side in (0, 1):
            level = self._levels[side]
            if level < SILENT_BELOW:
                # Reset so the first click fires the instant something appears,
                # rather than partway through a stale period.
                self._phase[side] = 0.0
                continue
            rate = CLICKS_PER_S[0] + level * (CLICKS_PER_S[1] - CLICKS_PER_S[0])
            phase = (self._phase[side] + np.arange(n) * rate / RATE) % 1.0
            self._phase[side] = (self._phase[side] + n * rate / RATE) % 1.0

            since_click = phase / rate          # seconds since this period's click began
            envelope = np.where(since_click < CLICK_S,
                                np.exp(-since_click / (CLICK_S / 3)), 0.0)
            loudness = AMPLITUDE * (0.55 + 0.45 * level)
            out[:, side] = loudness * envelope * np.sin(2 * np.pi * PITCH_HZ[side] * t)
        return out

    def _run(self) -> None:
        while not self._stop.is_set():
            pcm = (np.clip(self._block(), -1.0, 1.0) * 32767).astype(np.int16)
            try:
                self._proc.stdin.write(pcm.tobytes())   # blocks when aplay is full: real-time pacing
            except (BrokenPipeError, OSError, ValueError):
                self.status = "sound output lost"
                return

    def close(self) -> None:
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=2)
