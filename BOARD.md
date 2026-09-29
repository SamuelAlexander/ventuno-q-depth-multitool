# VENTUNO Q notes

What building Depth Multitool taught us about the Arduino VENTUNO Q, for anyone
bringing their own project to the board. Everything here was measured on a
VENTUNO Q between August and September 2026.

## The board, as a computer

| | |
|---|---|
| OS | Ubuntu 24.04, kernel 6.8 (qcom), aarch64 |
| CPU | 8× Cortex-A55 |
| Desktop | GNOME on Wayland, the `arduino` user logged in automatically |
| NPU | Qualcomm Hexagon, reached through the QNN TFLite delegate |
| Power in | 7–24 V on the barrel jack or screw terminals, or USB-C Power Delivery at 9–20 V |

`ping` to the board fails because ICMP is filtered; that says nothing about
whether it is up. `nc -z <board-ip> 22` does.

## Getting the NPU into an ordinary program

Out of the box there is no `/usr/lib/libQnnTFLiteDelegate.so` on the host.
Arduino ships the AI stack inside the Arduino App Lab Docker images, so the NPU
libraries live in container layers. The Qualcomm package repository is already
configured though, so one command puts them on the host:

```bash
sudo apt install qairt-libs qairt-dsp-binaries
```

That installs the Qualcomm AI Runtime (QAIRT 2.46), after which any LiteRT
program reaches the NPU with one delegate:

```python
delegate = load_delegate("/usr/lib/libQnnTFLiteDelegate.so", options={"backend_type": "htp"})
interpreter = Interpreter(model_path=model, experimental_delegates=[delegate])
```

The `arduino` user is already in the `fastrpc` group, so no permission changes
are needed.

## Performance

Depth Anything V2 Small, float, 336×252 input, as the app ships:

| | |
|---|---|
| NPU inference, balanced mode | 15.7 ms |
| Whole app, live camera, fullscreen on a 1280×800 panel | 20 FPS in every tool, capped for battery life |
| Board power | 11.8 W |

Uncapped and in sustained high performance mode, the same app runs at 30 FPS,
the webcam's own limit, for 15.2 W. See Power below for the trade.

Five things decided the speed. The first one matters for almost any model on this board.

**Set the NPU's performance mode.** Left at the delegate's default, the Hexagon runs
at low clocks. At a 336×252 input the same model took 58.3 ms per frame by default
and 12.7 ms in *sustained high performance* mode, 4.6× faster, from one delegate
option:

```python
options = {"backend_type": "htp", "htp_performance_mode": "1"}   # 1 = sustained high performance
```

| `htp_performance_mode` | NPU time, 336×252 |
|---|---|
| unset, or 0 (default) | 58.3 ms |
| 1, sustained high performance | 12.7 ms |
| 2, burst | 12.4 ms |
| 3, high performance | 12.8 ms |
| 8, balanced | 17.5 ms |

Sustained high performance is the mode meant for continuous load; burst is only 2%
faster. This also explains why Qualcomm AI Hub's device farm used to look much
faster than the board: the farm profiles at high clocks. With the mode set, AI Hub's
hosted VENTUNO Q reports 15.7 ms for this model and the board measures 17.4 ms.

**The CPU governor, less than it first seemed.** At the default `schedutil`, the
Cortex-A55 cores clock down while they wait on the NPU, and CPU work such as
resizing and drawing ran about twice as slow in a serial loop. Once the model ran
on its own thread, that mostly stopped mattering: uncapped, `performance` gave
30.5 FPS against 29.7 for `schedutil`, and cost only 0.2 W more. The app leaves
the governor alone.

**Overlapping the NPU with drawing.** LiteRT releases Python's GIL during
`invoke()`, which we measured: pure Python work on another thread kept 99% of its
speed while the NPU ran. So the model runs on its own thread and the main thread
draws the previous frame at the same time, and drawing never adds to the frame time.

**The GIL switch interval.** With Python's default of 5 ms, the model thread queued
behind the numpy-heavy Flood tool several times a frame, and its preprocessing grew
from 5.6 ms to 21.7 ms. `sys.setswitchinterval(0.001)` fixed it.

**Host cost scales with input size.** Work outside the model was first assumed to be
a fixed cost. It is not: a 768×576 model input cost 16.1 ms of host time, a 336×252
one 6.1 ms. On an all-A55 board that is large enough to change which model is faster
end to end.

## Display

The desktop is GNOME on Wayland; OpenCV's Qt window runs through XWayland. From
the desktop that just works. Launched over SSH it needs three variables, one of
which has a random per-login suffix:

```bash
export XDG_RUNTIME_DIR=/run/user/1000 DISPLAY=:0
export XAUTHORITY=$(ls -t /run/user/1000/.mutter-Xwaylandauth.* | head -1)
```

**Fullscreen needs the window to be mapped first.** Requested at once, it is
ignored and the app opens as a 400×300 window in a corner. About a second after
the window first appears is enough.

**Cache the prepared NPU graph.** The delegate prepares a model's graph for the
NPU on every load: 13.2 s for Depth Anything V2 Small at 392×294. Given a cache
directory and a model token, it writes the prepared graph (54 MB here) to disk the
first time and loads it in 0.4 s afterwards. Put the file's size and timestamp in
the token, so a changed model never loads an old graph:

```python
options.update({"cache_dir": str(Path.home() / ".cache/myapp/npu"),
                "model_token": f"{stem}_{size}_{mtime}"})
```

With that, and the installer preparing the model once, the app goes from a
double-click to a live picture in 4.2 s instead of 16.1 s.

**Keep the event loop turning during slow startup work.** GNOME marks a window as
not responding after about 5 s without events. Preparing a float model on the NPU
takes several seconds, so load it on a thread and keep drawing a splash screen.

**Desktop icons need two things.** The `.desktop` file must be executable and
marked trusted, which is what right-click, Allow Launching does:
`gio set <file> metadata::trusted true`.

## Audio

PipeWire runs two outputs: HDMI, and the board's own analog output, which is the
default. To send one stream to the HDMI display without changing the system
default, name the sink for that stream only:

```bash
PIPEWIRE_NODE=<hdmi sink node name> aplay -D pipewire -t raw -f S16_LE -r 44100 -c 2 -
```

The node name encodes ALSA card and device numbers, so find it by description
(`pw-dump`, look for an `Audio/Sink` with HDMI in `node.description`) rather than
hardcoding it.

Small HDMI displays often have a single speaker, and the one used here plays
stereo as mono. Left and right are lost, so the Sonar tool tells the sides apart
by pitch instead.

## Camera

With a UVC webcam, request MJPG before the resolution, or it negotiates raw YUYV
and caps near 10 FPS at 720p. The Logitech MX Brio then delivers 30 FPS at
1280×720. Two of its defaults are worth checking:

- `exposure_dynamic_framerate` is on, so in a dim room the camera lowers its own
  frame rate to lengthen exposure.
- `power_line_frequency` defaults to 60 Hz. Where mains runs at 50 Hz, set it to
  avoid banding under some lights:
  `v4l2-ctl -d /dev/video0 -c power_line_frequency=1`.

## Power

The board has its own power monitor, an INA232 on the 12 V input, so power can
be measured without a USB meter. Its hwmon number can change between boots, so
find it by name; the reading is in microwatts:

```bash
grep -l ina232 /sys/class/hwmon/*/name | sed 's/name$/power1_input/' | xargs cat
```

It measures the board only; a display fed from the same power bank through a
splitter does not show up in it. Board power for the whole app, fullscreen, live
camera, Slicer:

| NPU mode | Frame cap | CPU governor | Board | FPS |
|---|---|---|---|---|
| app closed | | schedutil | 6.1 W | |
| 1, sustained high performance | none | performance | 15.2 W | 30.5 |
| 1, sustained high performance | none | schedutil | 15.0 W | 29.7 |
| 8, balanced | none | schedutil | 13.2 W | 22.6 |
| 8, balanced | 20 FPS | schedutil | 12.0 W | 20 |
| 8, balanced | 15 FPS | schedutil | 11.9 W | 15 |
| 4, power saver | none | schedutil | 11.7 W | 10 |

The balanced NPU mode and a frame cap are the levers; the CPU governor is not.
Capping works best when the camera is asked for the capped rate too, so it
captures and sends fewer frames instead of the app decoding 30 a second and
dropping the rest: at 15 FPS that saved another 0.45 W.

The whole setup, board plus 10 inch display plus webcam, ran from a UGREEN
Nexode PB724 (100 W, 12 V at 3 A) with no reset and no throttling in every
configuration above.

## The microcontroller

This project does not use the STM32, but the path is open and verified.
`arduino-router.service` runs on the Linux host and listens on
`unix:///var/run/arduino-router.sock`, connected to the MCU over `/dev/ttyACM0`.
Its Python client, `arduino.app_utils`, ships only inside the App Lab images, but
it is pure Python under MPL-2.0 and works from an ordinary program once copied out.
A `Bridge.call` from a native process reaches the router and gets a proper RPC
reply.

## Small traps

- A freshly booted board runs `unattended-upgrades`, which holds the package lock
  for minutes. `apt-get -o DPkg::Lock::Timeout=900 install ...` waits instead of
  failing.
- `python3 -m venv` needs `python3-venv` installed first.
- Over SSH, `pkill -f some_script.py` also matches the SSH shell running it and
  kills the session. `pkill -f "[s]ome_script.py"` does not.
- GNOME keeps notification banners on screen while nobody touches the board, so
  an old notification can look like a new one.
- Screenshots on the board cost enough CPU to lower a running app's frame rate
  while they are taken.
