#!/usr/bin/env bash
# Set up Depth Multitool on an Arduino VENTUNO Q. Safe to run again.
#
#   git clone <this repository> ~/depth-multitool
#   cd ~/depth-multitool && ./install.sh
#
# 1. installs the Qualcomm AI Runtime, which puts the NPU delegate on the host
# 2. builds a Python environment with the three packages the app needs
# 3. fetches the compiled model into models/
# 4. puts a Depth Multitool icon on the desktop and in the app grid
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
ROOT="$(pwd)"

MODEL="models/dav2s_float_336x252.tflite"
# Where to download the compiled model from: this project's GitHub release. Skipped
# if the model is already in models/, e.g. built with research/export_model.py.
MODEL_URL="${MODEL_URL:-https://github.com/SamuelAlexander/ventuno-q-depth-multitool/releases/download/v1.0/dav2s_float_336x252.tflite}"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

say "Qualcomm AI Runtime (the NPU delegate)"
# Arduino ships the NPU libraries inside its App Lab containers rather than on
# the host. The Qualcomm PPA is already configured on the VENTUNO Q image, so
# one package puts the delegate where a normal Python program can load it.
packages=()
[ -f /usr/lib/libQnnTFLiteDelegate.so ] || packages+=(qairt-libs qairt-dsp-binaries)
python3 -c "import ensurepip" 2>/dev/null || packages+=(python3-venv)
if [ "${#packages[@]}" -gt 0 ]; then
    # A freshly booted board runs unattended-upgrades, which holds the package
    # lock for minutes. Wait for it instead of failing.
    sudo apt-get -o DPkg::Lock::Timeout=900 install -y "${packages[@]}"
else
    echo "Already installed."
fi
if [ ! -f /usr/lib/libQnnTFLiteDelegate.so ]; then
    echo "The NPU delegate is still missing after the install; stopping." >&2
    exit 1
fi

say "Python environment"
[ -x venv/bin/python ] || python3 -m venv venv
./venv/bin/pip install --quiet --upgrade pip
./venv/bin/pip install --quiet ai-edge-litert numpy opencv-python

say "Model"
if [ -f "$MODEL" ]; then
    echo "Already present: $MODEL"
elif [ -n "$MODEL_URL" ]; then
    mkdir -p models
    curl -fL --progress-bar -o "$MODEL" "$MODEL_URL"
else
    cat >&2 <<EOF
No model yet. Either:
  - copy dav2s_float_336x252.tflite into $ROOT/models/
  - build it with research/export_model.py (needs a free Qualcomm AI Hub account)
  - or re-run with:  MODEL_URL=<link to the .tflite> ./install.sh
EOF
    exit 1
fi

say "Preparing the model on the NPU (first time only)"
# Preparing a float model's graph takes about 15 s. Doing it once here, into the
# app's cache, makes every double-click, including the first, start in seconds.
./venv/bin/python device/depth_multitool.py --prepare

say "Desktop icon"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}"
chmod +x device/launch.sh

entry="$(mktemp)"
sed "s|@ROOT@|$ROOT|g" device/depth-multitool.desktop > "$entry"
mkdir -p "$HOME/.local/share/applications"
install -m 644 "$entry" "$HOME/.local/share/applications/depth-multitool.desktop"

desktop_dir="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
mkdir -p "$desktop_dir"
install -m 755 "$entry" "$desktop_dir/depth-multitool.desktop"
rm -f "$entry"

# GNOME only launches desktop files that are marked trusted: the same thing as
# right-clicking the icon and choosing Allow Launching.
if ! gio set "$desktop_dir/depth-multitool.desktop" metadata::trusted true 2>/dev/null; then
    echo "Could not mark the icon trusted: right-click it and choose Allow Launching."
fi

say "Done"
echo "Double-click Depth Multitool on the desktop, or run: $ROOT/device/launch.sh"
