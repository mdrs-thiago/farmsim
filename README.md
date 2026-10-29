# farmsim

Thorvald II row following in photogrammetry models of the Vollebekk fields.
You can simulate the robot in a surveyed field, record labelled camera data,
train the row-perception CNN, and close the loop CNN → controller → robot.

Stack: Ubuntu 26.04, ROS 2 Lyrical (Python 3.14), Gazebo (gz-sim 10, via
`ros_gz`), PyTorch, [torch-fis](https://github.com/mdrs-thiago/fuzzy_torch).

- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Running the simulation](#running-the-simulation)
- [The pipeline](#the-pipeline)
- [Configuration](#configuration)
- [Paths: which rows to drive](#paths-which-rows-to-drive)
- [Controllers](#controllers)
- [Perception and ground truth](#perception-and-ground-truth)
- [Adding a new field](#adding-a-new-field)
- [Troubleshooting](#troubleshooting)

---

## Repository layout

```
farmsim/
├── robots/        Thorvald ROS 2 packages (URDF for all variants, kinematics, drivers,
│                  teleop, GUI) + thorvald_gz (twist_to_joints -> gz-sim joint plugins)
├── farm/          ROS package farmsim_farm: worlds, launch files, row geometry, paths
│   ├── fields/<date>/   one folder per survey: rows, label mesh, canopy/structure
│   │                    rasters + georef, heightmap; model/ + topdown.png come from Drive
│   ├── models/          shared props (the weed mesh scattered into corridors)
│   └── tools/           build the ground-truth rasters/mesh of a field
├── control/       ROS package farmsim_control: controllers -> (v, omega), path-following
│                  node, RL environment
├── perception/    ROS package farmsim_perception: CNN / CNN-LSTM row models, line models,
│                  training, label rendering + dataset export, perception node
├── utils/         farmsim_utils: config loading, sim process control, asset fetching
├── cfg/           YAML for every script (robot, farm, control, paths/, ...)
├── tests/         pytest suite (no simulator needed)
├── dataset/       (not in git) labelled sequences
├── models/        (not in git) trained weights
├── outputs/       (not in git) recordings, runs, previews
│
├── select_path.py      define / check / preview the rows to drive
├── run_perception.py   perception only: offline on the dataset, or live in the sim
├── record_dataset.py   record labelled sequences for training
├── run_control.py      full stack in the sim: CNN + controller (or ground truth)
├── optimize.py         optimise CNN + controller together
└── build.sh            colcon build of the ROS packages
```

---

## Installation

### 0. Requirements

| | |
|---|---|
| OS | Ubuntu 26.04 (ROS 2 Lyrical's tier-1 platform) |
| GPU | NVIDIA with a recent driver recommended. Gazebo rendering, label rendering (OpenGL via moderngl) and CNN training all use it. CPU works but is slow. |
| Disk | ~1 GB for the repo + field assets, ~7 GB for the dataset, ~6 GB for the Python env (torch + CUDA) |

### 1. ROS 2 Lyrical and Gazebo

Follow the official ROS 2 Lyrical install guide for Ubuntu (deb packages),
then add the simulation and build packages:

```bash
sudo apt install ros-lyrical-desktop ros-lyrical-ros-gz \
    ros-dev-tools python3-rosdep python3-venv
sudo rosdep init      # only once per machine
rosdep update
```

`ros-lyrical-ros-gz` pulls in Gazebo (gz-sim), the ROS↔Gazebo bridge and
the gz-transport Python bindings the recorder uses. Check it with:

```bash
source /opt/ros/lyrical/setup.bash
gz sim --versions                       # prints the gz-sim version
python3 -c "import rclpy, gz.transport; print('ok')"
```

### 2. Clone and install the ROS dependencies of the packages

```bash
git clone git@github.com:mdrs-thiago/farmsim.git
cd farmsim
source /opt/ros/lyrical/setup.bash
rosdep install --from-paths robots farm control perception --ignore-src -y \
    --skip-keys "ament_python gazebo_ros gazebo_ros2_control gazebo_plugins \
                 velocity_controllers position_controllers joint_state_broadcaster controller_manager"
```

The skipped keys belong to `robots/thorvald_sim`, the legacy Gazebo Classic
launcher. farmsim doesn't use it (it runs the new Gazebo through
`thorvald_gz`), and Classic has no Lyrical packages.

### 3. Python environment

One venv holds everything. It is created with `--system-site-packages`, so
it still sees ROS's `rclpy`, `cv2`, `numpy` and `yaml`. Don't pip-install
those into it, or the pip copies would shadow ROS's.

```bash
source /opt/ros/lyrical/setup.bash
python3 -m venv --system-site-packages ~/.venvs/farmsim
source ~/.venvs/farmsim/bin/activate
pip install -e '.[dev,drive]'
```

This installs the four farmsim Python packages in editable mode, plus torch,
torchvision, torch-fis, scipy, trimesh, moderngl, pytest and gdown. The venv
can live anywhere; `~/.venvs/farmsim` is just the convention used here. If
you're also editing torch-fis, follow with
`pip install -e /path/to/fuzzy_torch`.

### 4. Build the ROS packages

With the venv **active**:

```bash
./build.sh
source install/setup.bash
```

`build.sh` runs `python -m colcon build --symlink-install` under the venv's
Python. This matters: the node executables (`controller_node`,
`perception_node`, ...) get that interpreter, and therefore torch. With
`--symlink-install`, Python edits take effect without rebuilding. Rebuild
only after adding files, entry points or launch files, or after changing
C++ code.

### 5. Field assets (meshes and top-down photos)

The photogrammetry meshes and the top-down orthophotos (~100–200 MB per
field) aren't in git. Each field has one zip on Google Drive, linked in
`cfg/assets.yaml`:

```bash
python -m farmsim_utils.fetch_assets fetch                  # all fields
python -m farmsim_utils.fetch_assets fetch --field 20250425
```

This fills `farm/fields/<date>/model/` and `farm/fields/<date>/topdown.png`.
To update the zips after changing a model, run
`python -m farmsim_utils.fetch_assets pack`. It writes `outputs/asset_zips/`;
upload those and update the links.

### 6. Dataset (optional, for training)

`dataset/` isn't in git. Put `row_perception_v1/` (and optionally
`row_anomalies_v1/`) there, see `dataset/README.md`. You can also record a
new dataset with `record_dataset.py`. The training cache
(`cache/256x192/*.npz`) is built automatically on first use.

### 7. Check the installation

```bash
pytest                                       # 72 tests, ~30 s, no simulator
python select_path.py cfg/paths/example.yaml # writes outputs/paths/example.png
python run_control.py run.pose_source=gt run.path=paths/single_row.yaml run.max_duration_s=120
```

The last command starts the simulator headless, drives row 9 with the PD
controller on ground truth, and prints the tracking metrics.

### Every new terminal

```bash
cd farmsim
source /opt/ros/lyrical/setup.bash
source ~/.venvs/farmsim/bin/activate
source install/setup.bash
```

---

## Running the simulation

### Through the scripts (recommended)

The root scripts start Gazebo, spawn the robot, start the nodes, run the
experiment, tear everything down and save results under `outputs/`. Add
`sim.headless=false` to watch it in the Gazebo GUI:

```bash
python run_control.py run.pose_source=gt sim.headless=false
```

or record it: `run.video=true` adds a chase camera behind the robot and writes
`chase_camera.mp4`, `front_camera.mp4` and `video.mp4` (both cameras with the
controller state drawn on) to the run folder. Runs can be put side by side:

```bash
python run_control.py run.pose_source=gt run.video=true control.strategy=pd run.name=pd
python run_control.py run.pose_source=gt run.video=true control.strategy=fuzzy run.name=fuzzy
python -m farmsim_utils.video compose outputs/runs/pd_vs_fuzzy.mp4 outputs/runs/pd outputs/runs/fuzzy
```

### By hand with `ros2 launch`

`farm/launch/farm_sim.launch.py` brings up one field with the robot:

```bash
# just the field and the robot, GUI on
ros2 launch farmsim_farm farm_sim.launch.py field:=20250425 headless:=false

# drive it yourself: camera window with the rows projected, w/a/s/d to drive
ros2 launch farmsim_farm farm_sim.launch.py field:=20250425 headless:=false teleop_view:=true

# controller on a path (ground truth), spawned at the path start
python select_path.py cfg/paths/single_row.yaml       # prints the spawn pose
ros2 launch farmsim_farm farm_sim.launch.py field:=20250425 headless:=false \
    x:=-0.63 y:=-14.05 yaw:=0.0 \
    control:=true control_config:=$PWD/cfg/control.yaml \
    path_file:=$PWD/cfg/paths/single_row.yaml pose_source:=gt

# same, but the CNN drives (needs a trained checkpoint)
ros2 launch farmsim_farm farm_sim.launch.py ... pose_source:=perception \
    perception:=true checkpoint:=$PWD/models/row_net/cnn/last.pt perception_view:=true
```

Main launch arguments:

| argument | default | meaning |
|---|---|---|
| `field` | `20250425` | folder in `farm/fields/` |
| `world` | `worlds/vollebekk_<field>.sdf` | custom world (record_dataset passes a randomised one) |
| `headless` | `false` | server only, no GUI |
| `x y z yaw` | `0 0 2 0` | spawn pose. The robot drops from `z` onto the terrain. |
| `robot_config` | 4wd2ws tora | Thorvald variant YAML (`robots/thorvald_model/config/`) |
| `camera_height camera_pitch camera_x camera_hfov ...` | 0.9 m, 0.15 rad, 0.8 m, 1.396 rad | front camera mount |
| `control`, `control_config`, `path_file`, `pose_source`, `control_log` | off | farmsim_control controller node |
| `perception`, `checkpoint`, `perception_view` | off | farmsim_perception CNN node |
| `teleop_view`, `live_view` | off | keyboard driving window / live segmentation window |
| `record`, `record_path` | off | ros2 bag of camera + tf + odom |

Topics: `/front_camera` (+ `/camera_info`), `/odom` (ground truth from gz),
`/tf`, `/cmd_vel` (`geometry_msgs/Twist`, consumed by `twist_to_joints`),
`/perception/row_state` (CNN output, see `utils/farmsim_utils/row_state.py`).

All farmsim sims use the gz partition `farmsim` (`export GZ_PARTITION=farmsim`
to use `gz topic` / `gz service` on them). Run one simulation at a time.

---

## The pipeline

| script | what it does | output |
|---|---|---|
| `select_path.py` | define/validate a row path, preview it on the top-down photo | `cfg/paths/*.yaml`, `outputs/paths/*.png` |
| `run_perception.py` | CNN on dataset sequences, scored, with overlay videos. `--live`: the CNN on the live camera while the robot drives on ground truth. | `outputs/perception/<run>/` |
| `record_dataset.py` | randomised sessions → labelled sequences appended to the dataset | `dataset/row_perception_v1/`, `outputs/recordings/` |
| `run_control.py` | drive a path with CNN + controller, or on ground truth, scored on ground-truth error | `outputs/runs/<run>/` |
| `optimize.py` | optimise CNN + controller together (control-aware loss, or RL) | `models/optimize/<run>/` |

A typical cycle:

```bash
# 1. pick the rows to drive
python select_path.py --field 20250425 --list-rows
python select_path.py --field 20250425 --passes 3:west:east 5:east:west --out cfg/paths/two_rows.yaml

# 2. (more) training data
python record_dataset.py record.sessions=20

# 3. train the perception
python -m farmsim_perception.training.train --model cnn --run-name cnn
python run_perception.py perception.checkpoint=models/row_net/cnn/last.pt

# 4. close the loop
python run_control.py run.path=paths/two_rows.yaml run.checkpoint=models/row_net/cnn/last.pt
python run_control.py run.path=paths/two_rows.yaml control.strategy=fuzzy run.repeats=3

# 5. optimise CNN + controller together, then drive with the result
python optimize.py optimize.control_aware.checkpoint=models/row_net/cnn/last.pt
python run_control.py control.strategy=fuzzy \
    control.checkpoint=models/optimize/<run>/controller.pt \
    run.checkpoint=models/optimize/<run>/perception.pt
```

**`record_dataset.py`.** Each session draws a random seed and, from it,
lighting, weeds, spawn row and offset, controller and gains, and camera
mount. It records `record.target_frames` frames with ground-truth camera
poses, renders the labels, runs a coarse alignment check, and appends the
session to the dataset with a split (by world seed) and a `sessions.csv`
row. A failed session doesn't stop a campaign. `record.pose_source=perception`
records with the CNN driving instead, for DAgger-style data on the states
the CNN itself reaches.

**`run_control.py`.** It spawns at the path start and drives until the path
is done, the time limit hits, or the robot stops making progress. It writes
`metrics.json` with in-row RMSE and max offset, heading RMSE, progress, speed,
and, when the CNN runs, its estimate error and confidence.
`control_log.jsonl` has every control cycle.

**`optimize.py`.** `method: control_aware` fine-tunes the CNN and trains the
controller jointly on the recorded data. The loss adds the error in the
*command* the controller would issue from the CNN estimate vs from the truth,
plus an anchor to a reference controller. An optional closed-loop check in
the sim runs at the end. `method: rl` is an experimental REINFORCE loop on
the simulator (`control/farmsim_control/rl_env.py`).

---

## Configuration

Every root script reads one YAML file from `cfg/` and takes `key.sub=value`
overrides after it. Values are parsed as YAML (`true`, `0.3`, `[1, 2]`):

```bash
python run_control.py control.strategy=fuzzy run.path=paths/single_row.yaml sim.headless=false
python run_control.py --config my_experiment.yaml          # file in cfg/ or any path
```

| file | contents |
|---|---|
| `robot.yaml` | Thorvald variant, spawn height, camera mount |
| `farm.yaml` | field, headless, startup wait |
| `control.yaml` | strategy, gains, speed policy, fuzzy system, rate limits, headland |
| `assets.yaml` | Drive links of the field zips |
| `paths/*.yaml` | row paths |
| `run_control.yaml`, `run_perception.yaml`, `record_dataset.yaml`, `optimize.yaml` | per script, `include:` the files above |

Each run saves its fully resolved config next to the results, so a run can
be repeated with `--config outputs/runs/<run>/config.yaml`.

---

## Paths: which rows to drive

A path is an ordered list of row passes. Each pass enters a row at one end
and leaves at the other:

```yaml
# cfg/paths/example.yaml
field: 20250425
lead_in_m: 2.0      # straight approach before each row
lead_out_m: 2.0     # straight exit after each row
passes:
  - {row: 3, enter: west, exit: east}
  - {row: 5, enter: east, exit: west}
```

Row ids come from `farm/fields/<date>/rows_centerline.yaml`. Row ends are
compass directions in the world frame (east = +x, north = +y) or
`start`/`end`. Inside rows the configured controller drives (on ground truth
or the CNN). Between rows the robot crosses the headland with ground-truth
pure pursuit. The robot's progress along the path always comes from ground
truth, so it never jumps to the neighbouring row.

```bash
python select_path.py --field 20250425 --list-rows
python select_path.py --field 20250425 --passes 3:west:east 5:east:west --out cfg/paths/mine.yaml
python select_path.py cfg/paths/mine.yaml        # validate, print spawn pose, outputs/paths/mine.png
```

---

## Controllers

All strategies output **linear and angular velocity**
(`control/farmsim_control/laws.py`). e = lateral offset (> 0 left of the
row), ψ = heading error, r = yaw rate:

| strategy | angular velocity | linear velocity |
|---|---|---|
| `pd` | −k_o·e + k_h·ψ − k_d·r | shared speed policy |
| `pure_pursuit` | lookahead point on the (locally straight) row | shared speed policy |
| `stanley` | ψ − atan(k·e / (v+ε)) − k_d·r | shared speed policy |
| `bangbang` | ± magnitude, hysteresis deadband | shared speed policy |
| `fuzzy` | TSK (torch-fis): 9 rules on e × ψ, damped by r | own TSK: 16 rules on \|ψ\|, \|e\|, CNN confidence, distance to row end |

The shared speed policy (`control.speed`) slows down in turns, when
misaligned, when the perception confidence is low, and before the row end.
The fuzzy rule base starts as a smoothly interpolated PD law, so it drives
sensibly before any training. Every controller is an `nn.Module` with
trainable parameters, which `optimize.py` relies on. `control.checkpoint`
loads trained parameters.

---

## Perception and ground truth

`perception/farmsim_perception/models/row_net.py` holds the ResNet-18
multitask models (`cnn`, `cnn_lstm`). They predict segmentation
(background / crop / corridor / bare bed), the row-bed box, the row line,
and the lateral offset / heading that the controller consumes.
`models/line_models*.py` are the lightweight line estimators. All of them
train on `dataset/row_perception_v1`:

```bash
python -m farmsim_perception.training.train --model cnn --run-name cnn      # -> models/row_net/cnn/last.pt
python -m farmsim_perception.training.train --model cnn_lstm --run-name cnn_lstm
python -m farmsim_perception.training.evaluate --models cnn cnn_lstm
python -m farmsim_perception.training.train_line_models
python -m farmsim_perception.training.train_line_models_v2
```

`perception/EXPERIMENTS.md` describes the CNN vs CNN-LSTM experiment and the
anomaly variant (puddles, ruts, debris).

**Ground truth is rendered, not annotated.** A field with ground truth has
a label mesh (the terrain surface) and two georeferenced class rasters: the
vegetation raster (canopy vs bare soil, the canopy mask) and the structure
raster (bed vs corridor). For every recorded frame, the mesh is rendered,
textured with the rasters, from the exact ground-truth camera pose
(`labeling/render_labels.py`). This gives pixel-aligned masks. The row line,
offset and heading come from the row geometry (`labeling/export_recording.py`).

---

## Adding a new field

A field is a folder `farm/fields/<date>/` plus a world `farm/worlds/vollebekk_<date>.sdf`.

1. **Model.** Put the photogrammetry model in `farm/fields/<date>/model/<name>/`
   (`model.config`, `model.sdf`, OBJ/MTL, textures). The world includes it
   as `model://<name>`.
2. **Rows.** Capture a top-down image
   (`ros2 launch farmsim_farm capture_topdown.launch.py`), then mark rows
   (`ros2 run farmsim_farm mark_rows`, `polygon_to_centerline`) to get
   `rows_centerline.yaml`. With these, the field can be driven.
3. **Ground truth** (needed for recording datasets). Run `farm/tools/`:
   `bake_world_mesh.py` (label mesh + georef) → `bake_terrain_heightmap.py`
   → `build_label_raster.py` (vegetation/canopy) → `build_structure_raster.py`.
   Each script's docstring has the exact command.
   `python -c "from farmsim_farm.fields import Field; print(Field('<date>').missing_ground_truth)"`
   lists what's still missing.
4. **Share.** Run `python -m farmsim_utils.fetch_assets pack`, upload the zip,
   and add its link to `cfg/assets.yaml`.

Only 20250425 has full ground truth today. 20250624 has rows, and 20260331
only has its model.

---

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `sim exited during startup` | read `outputs/runs/<run>/sim.log`. Usually a missing field model: run `fetch_assets fetch`. |
| Gazebo log: `uri [model://...] could not be resolved` | model not on `GZ_SIM_RESOURCE_PATH`. The scripts set it; for manual launches the launch file adds `farm/fields/*/model` and `farm/models`. |
| robot never moves, `control_log.jsonl` empty | the world didn't load (see above), or a leftover sim from a killed run: `python -c "from farmsim_utils.sim import stop; stop()"` |
| `ModuleNotFoundError: torch` from a node | the packages were built without the venv active. Activate it and run `rm -rf build install && ./build.sh`. |
| `record_dataset`: 0 frames | camera or pose topics missing. Check `outputs/recordings/<session>/record.log` and that only one sim runs. |
| label rendering fails (moderngl / EGL) | needs an OpenGL-capable GPU driver. Check with `python -c "import moderngl; moderngl.create_standalone_context()"`. |
| runs much slower than real time | expected with the large terrain mesh. Use `sim.headless=true` and close other GPU apps. `run.max_duration_s` is wall-clock time. |
| `run_control` stops with `stalled` | the robot made no progress for 60 s: off the row or stuck. Check `control_log.jsonl`. |

---

## Provenance

Reorganised from `thorvald_ros2_ws` (robot packages, `thorvald_gz_bringup`,
`tools/`) and the field meshes in `ros_src/`. Not carried over: the raw
recordings and bags, trained checkpoints, the smoke-test model in
`tools/training`, and the campaign shell scripts. `farmsim_utils.sim` and
`record_dataset.py` replace those scripts.
