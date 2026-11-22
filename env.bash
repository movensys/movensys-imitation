#!/usr/bin/env bash
# Source this in every terminal that runs this repo's nodes or training commands:
#   source ~/workspaces/movensys-imitation/env.bash
# Keeps the host, the containers and Isaac Sim on the same ROS domain (plan, Section 5).
export ROS_DOMAIN_ID=55   # the project domain (2026-10-24 decision); ~/.bashrc, the containers and Isaac use 55
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export MANIPULATOR_MODEL=dobot_cr3a

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
[ -f "$HOME/workspaces/movensys_ws/install/setup.bash" ] && source "$HOME/workspaces/movensys_ws/install/setup.bash"
# This repo's ROS package (colcon build --symlink-install --base-paths movensys_imitation_ros).
# shellcheck disable=SC1091
[ -f "$_HERE/install/setup.bash" ] && source "$_HERE/install/setup.bash"
# shellcheck disable=SC1091
[ -f "$_HERE/.venv/bin/activate" ] && source "$_HERE/.venv/bin/activate"
# Cyclone DDS on loopback only (2026-11-20): left to pick an interface it chose the EtherCAT port
# and the image traffic there dropped the servos. Same setting as the containers' cyclonedds.xml.
# Set CYCLONEDDS_URI before sourcing this to talk to another machine instead.
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-file://$_HERE/cyclonedds.xml}"
export MOVENSYS_IMITATION_ROOT="$_HERE"
unset _HERE
