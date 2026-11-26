#!/bin/bash
# Run a Go2 Isaac Lab script on this machine (source-built Isaac Sim + IsaacLab .venv).
# Replaces the old activate_isaac.sh, which pointed at ~/env_isaaclab and ~/isaac_sim.
#
#   ./run_isaac.sh go2_combined.py                # GUI
#   ./run_isaac.sh go2_combined.py --headless     # no window
#
# Override locations with ISAACLAB_DIR / ISAACSIM_DIR if they move.
set -e
GO2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/Robotics/IsaacLab}"
ISAACSIM_DIR="${ISAACSIM_DIR:-$ISAACLAB_DIR/_isaac_sim}"
ISAACSIM_DIR="$(readlink -f "$ISAACSIM_DIR")"

export CARB_APP_PATH="$ISAACSIM_DIR/kit"
export ISAAC_PATH="$ISAACSIM_DIR"
export EXP_PATH="$ISAACSIM_DIR/apps"
# shellcheck disable=SC1091
source "$ISAACSIM_DIR/setup_python_env.sh"
export PYTHONPATH="$GO2_DIR:$PYTHONPATH"

if [ $# -eq 0 ]; then
    echo "usage: $0 <script.py> [args...]" >&2
    exit 1
fi
cd "$GO2_DIR"
# Isaac Lab 3 is headless unless a visualizer is requested; keep the old GUI-by-default
# behaviour unless the caller passed --headless or chose a visualizer.
case " $* " in
    *" --headless "*|*" --viz"*|*" --visualizer"*) ;;
    *) set -- "$@" --viz kit ;;
esac
# The GUI app config enables isaacsim.robot.policy.examples, which needs
# omni.isaac.ml_archive; in this source build that lives in extsDeprecated/, which Kit
# doesn't search by default (startup fails with "Failed to resolve extension dependencies").
case " $* " in
    *" --kit_args"*) ;;
    *) set -- "$@" --kit_args "--ext-folder $ISAACSIM_DIR/extsDeprecated" ;;
esac
exec "$ISAACLAB_DIR/.venv/bin/python" "$@"
