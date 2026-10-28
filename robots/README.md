# Thorvald II — ROS2 Humble Port

> **Agricultural robot platform by [Saga Robotics](https://sagarobotics.com/), ported from ROS1 Melodic to ROS2 Humble.**

![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue?logo=ros)
![Gazebo Classic](https://img.shields.io/badge/Gazebo-Classic%2011-orange)
![Python](https://img.shields.io/badge/Python-3.10-green?logo=python)
![C++17](https://img.shields.io/badge/C++-17-blue?logo=cplusplus)
![License](https://img.shields.io/badge/License-BSD-lightgrey)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)


> **farmsim note.** This is the original Thorvald port README, kept for the
> package reference. In farmsim the stack runs on **ROS 2 Lyrical** (Python
> 3.14) and the simulation uses the **new Gazebo (gz-sim)** through
> `thorvald_gz` + `farm/launch/farm_sim.launch.py`, not Gazebo Classic. The
> Classic `thorvald_sim` package and the hardware drivers are kept as they
> were. See the top-level README for how farmsim uses the robot.

---

## Table of Contents

- [Overview](#overview)
- [Robot Variants](#robot-variants)
- [Package Structure](#package-structure)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Building](#building)
- [Simulation](#simulation)
- [Velocity Control](#velocity-control)
- [RViz Visualization](#rviz-visualization)
- [Roadmap](#roadmap)

---

## Overview

The **Thorvald II** is a modular, field-scale agricultural robot designed for precision farming tasks. This repository contains the ongoing port of its full software stack from **ROS1 Melodic** to **ROS2 Humble**, targeting feature parity with the original including:

- Gazebo Classic simulation with `ros2_control`
- Real hardware bringup via **CAN bus** (PEAK-USB / Roboteq FBL2360 motor drives) and **serial** (enclosure board, light rig)
- Platform-independent kinematics supporting all 8 robot configurations
- Teleop, velocity multiplexing, navigation (Nav2)

| Feature | Status |
|---|---|
| URDF / Xacro model (all variants) | ✅ Done |
| Gazebo Classic simulation | ✅ Done |
| `ros2_control` velocity + steering controllers | ✅ Done |
| `/cmd_vel` → joint kinematics node | ✅ Done |
| Custom messages & services | ✅ Done |
| CAN bus hardware driver | ✅ Done |
| Teleop (joystick) | ✅ Done |
| Twist mux (priority-based cmd_vel) | ✅ Done |
| Enclosure board / serial I/O | ✅ Done |
| Navigation (Nav2) | ⏳ Planned |
| RQT GUI | ✅ Done |

---

## Robot Variants

The URDF supports **8 distinct configurations** via YAML-driven xacro macros:

| Model | Drive | Steer | Notes |
|---|---|---|---|
| `thorvald_ii_4wd4ws_std` | 4WD | 4WS | Standard platform |
| `thorvald_ii_4wd4ws_slim` | 4WD | 4WS | Slim frame variant |
| `thorvald_ii_6wd6ws` | 6WD | 6WS | Extended platform |
| `thorvald_ii_4wd2ws_tora` | 4WD | 2WS | Tora frame |
| `thorvald_ii_2wd0ws_tora` | 2WD | Fixed | Tora fixed steering |
| `thorvald_ii_1wd1ws_tric` (×3) | 1WD | 1WS | Tricycle — front/rear/offset |
| `thorvald_ii_guren_ii` | 4WD | 4WS | Guren II body |
| `thorvald_ii_crazy` | — | — | Experimental |

---

## Package Structure

```
thorvald_ros2_ws/
├── src/
│   ├── thorvald_model/       # URDF/Xacro, meshes, robot configs, ros2_control
│   ├── thorvald_base/        # Kinematics node: /cmd_vel → wheel/steer commands
│   ├── thorvald_sim/         # Gazebo Classic simulation bringup
│   ├── imrt_teleop/          # Teleoperation utilities
│   └── imrt_virtual_joy/     # Virtual joystick (keyboard / GUI)
│
│   ├── thorvald_msgs/        # Custom messages & services
│   ├── thorvald_can_devices/ # Roboteq FBL2360 + KC Battery CAN drivers
│   ├── thorvald_base_driver/ # Hardware driver node, odometry, kinematics plugins
│   ├── thorvald_io_serial/   # Serial I/O (enclosure board / light rig)
│   ├── thorvald_teleop/      # Joystick teleop (Logitech F710, Xbox)
│   ├── thorvald_twist_mux/   # Priority-based velocity multiplexer
│   └── thorvald_bringup/     # Top-level launch files (real + sim, all variants)
└── ROS1_older_thorvald_compilable/   # Original ROS1 source (reference only)
```

---

## Architecture

### Simulation Stack

```
  /cmd_vel  (Twist)
      │
      ▼
┌─────────────────────┐
│   twist_to_joints   │  ← thorvald_base pkg
│  (kinematics node)  │     Port of pltf_clc_std (ICR-based 4WS kinematics)
└──────────┬──────────┘
           │  /wheel_velocity_controller/commands   [Float64MultiArray]  rad/s
           │  /steering_position_controller/commands [Float64MultiArray]  rad
           ▼
┌───────────────────────────────────┐
│  ros2_control / controller_manager│  ← gazebo_ros2_control plugin
│  ┌────────────────────────────┐   │
│  │ JointGroupVelocityController│  │  → wheel0..3  (continuous joints)
│  │ JointGroupPositionController│  │  → steering0..3 (revolute joints ±90°)
│  │ JointStateBroadcaster      │  │  → /joint_states
│  └────────────────────────────┘   │
└───────────────────────────────────┘
           │
           ▼
   Gazebo Classic (physics)
           │
           ▼
 /joint_states → robot_state_publisher → /tf
```

### Hardware Stack *(in progress)*

```
  CAN bus (PEAK-USB)
          │
          ▼
┌───────────────────────┐
│  thorvald_can_devices │  Roboteq FBL2360 (motors)
│  (C++ library)        │  KC Battery BMS
└──────────┬────────────┘
           ▼
┌──────────────────────┐
│   thorvald_base      │  base_driver node
│   (hardware driver)  │  Kinematics plugins (std / diff │/ tricycle)           │
└──────────────────────┘

      Serial (USB)
          │
          ▼
┌──────────────────────┐
│  thorvald_io_serial  │  Enclosure board + light rig │(Arduino)             │
└──────────────────────┘
```

---

## Prerequisites

| Dependency | Version |
|---|---|
| Ubuntu | 22.04 LTS |
| ROS2 | Humble Hawksbill |
| Gazebo | Classic 11 |
| `ros-humble-gazebo-ros2-control` | latest |
| `ros-humble-ros2-controllers` | latest |
| `ros-humble-joint-state-broadcaster` | latest |
| `xacro` | latest |

Install ROS2 simulation dependencies:

```bash
sudo apt install \
  ros-humble-gazebo-ros-pkgs \
  ros-humble-gazebo-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-joint-state-broadcaster \
  ros-humble-position-controllers \
  ros-humble-velocity-controllers \
  ros-humble-xacro \
  ros-humble-robot-state-publisher
```

---

## Building

```bash
# Clone the repository
git clone https://github.com/Gbrtenorio/thorvald_ros2_ws.git
cd thorvald_ros2_ws

# Source ROS2
source /opt/ros/humble/setup.bash

# Build
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

# Source the workspace
source install/setup.bash
```

---

## Simulation

### Launch (default: `thorvald_ii_4wd4ws_std`)

```bash
# Kill any stale Gazebo processes first
killall -9 gzserver gzclient 2>/dev/null; sleep 1

ros2 launch thorvald_sim sim.launch.py
```

The launch sequence:
1. Gazebo Classic opens with an empty world
2. Robot is spawned from `/robot_description`
3. `joint_state_broadcaster`, `wheel_velocity_controller`, `steering_position_controller` activate
4. `twist_to_joints` kinematics node starts

### Launch a different robot variant

```bash
ros2 launch thorvald_sim sim.launch.py \
  robot_model:=$(ros2 pkg prefix thorvald_model)/share/thorvald_model/config/thorvald_ii_6wd6ws/thorvald_ii_6wd6ws.yaml
```

### Headless (no GUI)

```bash
ros2 launch thorvald_sim sim.launch.py gui:=false
```

---

## Velocity Control

Once the simulation is running, control the robot from another terminal:

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash

# Drive forward at 0.5 m/s
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"

# Strafe left (omnidirectional — unique to 4WS robots!)
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.5, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"

# Spin in place
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.5}}"

# Continuous publishing (10 Hz)
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.3, y: 0.2, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.3}}"

# Stop
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}"
```

### Check active topics

```bash
ros2 topic list | grep -E "cmd_vel|joint_states|controller|odom"
ros2 topic echo /joint_states --once
```

---

## GUI Virtual Controller

Launch a virtual joystick (keyboard or on-screen sliders) to control the robot without needing a physical joystick:

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash

ros2 launch thorvald_sim thorvald_teleop.launch.py
```

This starts the `imrt_virtual_joy` interface that publishes to `/cmd_vel`, allowing full robot control from the desktop GUI.

---

## Joystick Teleop

```bash
# Logitech F710 (default /dev/input/js0)
ros2 launch thorvald_teleop teleop_logitech_f710.launch.py

# Xbox controller
ros2 launch thorvald_teleop teleop_xbox.launch.py

# Custom device
ros2 launch thorvald_teleop teleop_logitech_f710.launch.py joy_dev:=/dev/input/js1
```

Teleop publishes to `teleop_joy/cmd_vel` and `teleop_joy/joy_priority` (for twist_mux locking).

---

## Full Bringup (real hardware)

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash

# Standard 4WD4WS platform
ros2 launch thorvald_bringup thorvald_bringup.launch.py \
  robot_variant:=thorvald_ii_4wd4ws_std sim:=false

# Simulation
ros2 launch thorvald_bringup thorvald_bringup.launch.py \
  robot_variant:=thorvald_ii_4wd4ws_std sim:=true
```

Available variants: `thorvald_ii_4wd4ws_std`, `thorvald_ii_4wd4ws_slim`, `thorvald_ii_6wd6ws`,
`thorvald_ii_4wd2ws_tora`, `thorvald_ii_2wd0ws_tora`, `thorvald_ii_1wd1ws_tric`,
`thorvald_ii_guren_ii`, `thorvald_ii_crazy`.

---

## Hardware Monitor (RQT GUI)

Launch the Thorvald monitor panel from rqt:

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash

# Standalone
ros2 run thorvald_gui thorvald_gui

# Or load inside rqt
rqt --force-discover
# Plugins → Robot Tools → Thorvald Monitor
```

The panel has three tabs:
- **Motor Controllers** — mode, voltage, current, temperature, speed per drive
- **Batteries** — voltage, current, SoC, SoH, temperature, cycle count per BMS
- **Enclosure** — comm state, relay/safety flags, digital and analog pin values

---

## RViz Visualization

```bash
ros2 run rviz2 rviz2
```

Suggested displays:
| Display | Topic / Setting |
|---|---|
| **RobotModel** | Description Source: `Topic` → `/robot_description` |
| **TF** | — |
| **Fixed Frame** | `base_link` |

---

## Roadmap

### Phase 1 — Simulation ✅
- [x] URDF/Xacro model for all 8 variants
- [x] Gazebo Classic + `ros2_control` integration
- [x] ICR-based 4WS kinematics node (`/cmd_vel` → wheel/steer commands)
- [x] `joint_state_broadcaster` + velocity/position controllers

### Phase 2 — Hardware Drivers ✅
- [x] `thorvald_msgs` — 14 custom messages + 5 services
- [x] `thorvald_can_devices` — Roboteq FBL2360 + KC Battery BMS (CAN / PEAK-USB stub)
- [x] `thorvald_base_driver` — hardware driver node, odometry, kinematics plugins (pluginlib)
- [x] `thorvald_io_serial` — enclosure board + serial I/O (POSIX serial, no extra deps)

### Phase 3 — Full Bringup ✅
- [x] `thorvald_teleop` — joystick teleop (Logitech F710, Xbox)
- [x] `thorvald_twist_mux` — priority-based velocity multiplexer (self-contained Python node)
- [x] `thorvald_bringup` — per-variant launch files (real + sim, all 8 variants)
- [x] `thorvald_gui` — RQT monitoring panel (motors, batteries, enclosure)

---

## Original ROS1 Source

The original ROS1 Melodic packages are kept locally in `ROS1_older_thorvald_compilable/` as a reference (excluded from this repository via `.gitignore`). They are **not** required to build or run this ROS2 workspace.

---

## Credits

- **Original platform & ROS1 stack:** [Saga Robotics](https://sagarobotics.com/)
- **ROS2 port:** Togopuff
- **Kinematics:** Ported from `pltf_clc_std` (Lars Grimstad, Saga Robotics)
- **Gazebo plugin:** Based on `thorvald_gazebo_plugins` (Lars Grimstad, Saga Robotics)
