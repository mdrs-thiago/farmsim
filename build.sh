#!/usr/bin/env bash
# Build the ROS 2 packages (robots/, farm/, control/, perception/).
# --symlink-install keeps the Python packages pointing at the source tree,
# so edits need no rebuild and farm/fields is found from installed nodes.
# colcon runs under the ACTIVE python (`python -m colcon`), so the node
# scripts get the farmsim venv's interpreter (torch, torch-fis); /usr/bin/colcon
# would stamp them with the system python instead.
set -eo pipefail
cd "$(dirname "$0")"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "activate the farmsim venv first (see README, Setup)" >&2
    exit 1
fi
source "${FARMSIM_ROS_SETUP:-/opt/ros/lyrical/setup.bash}"
python -m colcon build --symlink-install --base-paths robots farm control perception "$@"
