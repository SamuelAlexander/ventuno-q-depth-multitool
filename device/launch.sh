#!/usr/bin/env bash
# What the desktop icon runs.
#
# A double-clicked icon has no terminal to print into, so output goes to a log
# and a dialog appears if the app fails to start. Also safe to run over SSH: the
# display variables are filled in if the desktop session did not provide them.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

LOG_DIR="$HOME/.cache/depth-multitool"
LOG="$LOG_DIR/last-run.log"
mkdir -p "$LOG_DIR"

# One instance at a time: a second copy would fight the first for the camera.
exec 9>"$LOG_DIR/lock"
if ! flock -n 9; then
    notify-send "Depth Multitool" "Already running." 2>/dev/null
    exit 0
fi

# The GNOME session is Wayland; OpenCV's window goes through XWayland, which
# needs DISPLAY and mutter's Xauthority file. That file's name carries a random
# per-login suffix, hence the glob.
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DISPLAY="${DISPLAY:-:0}"
if [ -z "${XAUTHORITY:-}" ]; then
    XAUTHORITY="$(ls -t "$XDG_RUNTIME_DIR"/.mutter-Xwaylandauth.* 2>/dev/null | head -1)"
    export XAUTHORITY
fi

./venv/bin/python device/depth_multitool.py "$@" >"$LOG" 2>&1
status=$?

# 129/130/143 are HUP/INT/TERM: logout, Ctrl+C or kill. Stopped, not crashed.
case "$status" in 0|129|130|143) exit "$status" ;; esac

# Release the single-instance lock before the dialog, or a relaunch would be
# refused for as long as the error sits on screen unanswered.
exec 9>&-
# zenity renders Pango markup, so strip the characters that would break it, and
# drop the NPU's graph-preparation chatter so the real error is what shows.
detail="$(grep -v -E '(Starting|Completed) stage|_bytes=|rpcmem|bandwidth summary|^=+|\[#|^[[:space:]]*$' "$LOG" \
          | tail -n 6 | tr -d '<>&')"
zenity --error --title="Depth Multitool" --width=520 \
    --text="Depth Multitool stopped unexpectedly (exit $status).\n\n$detail\n\nFull log: $LOG" \
    2>/dev/null
exit "$status"
