# farmsim

Thorvald II row following in photogrammetry models of the Vollebekk fields:
simulate the robot in a surveyed field, record labelled camera data, train
the row-perception CNN, and close the loop CNN → controller → robot.

Stack: ROS 2 Lyrical (Python 3.14), new Gazebo (gz-sim), PyTorch, torch-fis.

## Layout

```
farmsim/
├── robots/        Thorvald ROS 2 packages (model/URDF for all variants, kinematics,
│                  drivers, teleop, GUI) + thorvald_gz (twist_to_joints -> gz joints)
├── farm/          farmsim_farm: worlds, launch files, row geometry, row-pass paths,
│   ├── fields/<date>/   one folder per survey: rows, label mesh, canopy/structure
│   │                    rasters + georef, heightmap; model/ + topdown.png (Drive, not git)
│   ├── models/          shared props (the weed mesh world_gen scatters)
│   └── tools/           build the ground-truth rasters/mesh of a field
├── control/       farmsim_control: controllers -> (v, omega), path-following node, RL env
├── perception/    farmsim_perception: CNN / CNN-LSTM row models, line models, training,
│                  label rendering + dataset export, perception node
├── utils/         farmsim_utils: config loading, sim process control, asset fetching
├── cfg/           YAML for every script (robot, farm, control, paths/, ...)
├── dataset/       (not in git) row_perception_v1, row_anomalies_v1
├── models/        (not in git) trained weights
├── outputs/       (not in git) recordings, runs, previews
│
├── select_path.py      define/check/preview the row path to drive
├── run_perception.py   perception only: offline on the dataset, or live in the sim
├── record_dataset.py   record labelled sequences for training
├── run_control.py      full stack in the sim: CNN + controller (or ground truth)
└── optimize.py         optimise CNN + controller together
```

## Setup

```bash
# 1. Python env that also sees ROS's packages (rclpy, cv2, numpy, yaml)
source /opt/ros/lyrical/setup.bash
python3 -m venv --system-site-packages ~/.venvs/farmsim
source ~/.venvs/farmsim/bin/activate
pip install -e '.[dev]'           # add ,drive for Google Drive downloads (gdown)
# co-developing torch-fis:  pip install -e ../fuzzy/fuzzy_torch

# 2. ROS packages (symlink install, so Python edits need no rebuild)
./build.sh
source install/setup.bash

# 3. Field meshes + top-down photos (~100-200 MB per field; not in git)
python -m farmsim_utils.fetch_assets fetch      # links in cfg/assets.yaml
```

Run everything from the repo root with the venv active and
`install/setup.bash` sourced. `pytest` runs the tests (no simulator needed).

## Configuration

Every root script reads one file in `cfg/` and takes `key.sub=value`
overrides after it:

```bash
python run_control.py control.strategy=fuzzy run.path=paths/single_row.yaml sim.headless=false
python run_control.py --config my_experiment.yaml
```

`cfg/robot.yaml` (variant, camera mount), `cfg/farm.yaml` (field, headless)
and `cfg/control.yaml` (controller) are included by the script configs.
Each run saves the fully resolved config next to its results.

## Paths: which rows to drive

A path is an ordered list of row passes. Each pass enters a row at one end
and leaves at the other. The robot drives the rows with the configured
controller and moves between them through the headland with ground-truth
pure pursuit:

```yaml
# cfg/paths/example.yaml
field: 20250425
passes:
  - {row: 3, enter: west, exit: east}
  - {row: 5, enter: east, exit: west}
```

```bash
python select_path.py --field 20250425 --list-rows
python select_path.py --field 20250425 --passes 3:west:east 5:east:west --out cfg/paths/mine.yaml
python select_path.py cfg/paths/mine.yaml        # validate + outputs/paths/mine.png
```

Row ends are compass directions (east = +x, north = +y) or `start`/`end`.

## Controllers

All strategies output **linear and angular velocity** (`control/farmsim_control/laws.py`):

| strategy | steering | speed |
|---|---|---|
| `pd` | −k_o·e + k_h·ψ − k_d·r | shared speed policy |
| `pure_pursuit` | lookahead point on the row | shared speed policy |
| `stanley` | ψ − atan(k·e / (v+ε)) − k_d·r | shared speed policy |
| `bangbang` | ± magnitude, hysteresis deadband | shared speed policy |
| `fuzzy` | TSK (torch-fis), 9 rules on e × ψ, damped by r | own TSK, 16 rules on \|ψ\|, \|e\|, confidence, distance to row end |

The shared speed policy slows down in turns, when misaligned, when the
perception confidence is low, and before the row end (`control.speed`).
Every controller is an `nn.Module` with trainable parameters, which is what
`optimize.py` relies on. The fuzzy rule base starts as a smoothly
interpolated PD law, so it drives sensibly before any training.

## Perception

`perception/farmsim_perception/models/row_net.py` holds the ResNet-18
multitask models (`cnn`, `cnn_lstm`): segmentation, row-bed box, row line,
and lateral offset / heading. The controller consumes the offset and
heading. `models/line_models*.py` are the lightweight line estimators. Both
train on `dataset/row_perception_v1`:

```bash
python -m farmsim_perception.training.train --model cnn --run-name cnn      # -> models/row_net/cnn/last.pt
python -m farmsim_perception.training.evaluate --models cnn
python -m farmsim_perception.training.train_line_models
python -m farmsim_perception.training.train_line_models_v2
```

See `perception/EXPERIMENTS.md` for the CNN vs CNN-LSTM experiments and the
anomaly variant.

### Ground truth for training

Labels are rendered, not annotated. Each field with ground truth has a label
mesh (the terrain surface) textured with two class rasters. One is the
vegetation raster: canopy vs bare soil, the canopy mask. The other is the
structure raster: bed vs corridor. Both are georeferenced to the sim world.
For every recorded frame, the mesh is rendered from the exact ground-truth
camera pose (`perception/farmsim_perception/labeling/render_labels.py`).
The result is a pixel-aligned mask per frame, plus row line, offset and
heading from the row geometry. Build these assets for a new field with
`farm/tools/` (bake mesh → heightmap → rasters). Only 20250425 has them now.

## The pipeline scripts

| script | what it does | output |
|---|---|---|
| `select_path.py` | define/validate a path, preview it on the top-down image | `cfg/paths/*.yaml`, `outputs/paths/*.png` |
| `run_perception.py` | CNN on dataset sequences (scored, overlay videos); `--live`: CNN on the live camera while driving on ground truth | `outputs/perception/<run>/` |
| `record_dataset.py` | randomised sessions → labelled sequences appended to the dataset (`record.pose_source=perception` records with the CNN driving) | `dataset/row_perception_v1/`, `outputs/recordings/` |
| `run_control.py` | drive a path with CNN + controller (`run.pose_source=perception`) or on ground truth; scored on ground-truth error | `outputs/runs/<run>/metrics.json` |
| `optimize.py` | `control_aware`: fine-tune CNN + train controller on the control-relevant error, then a closed-loop check. `rl`: REINFORCE on the sim (experimental) | `models/optimize/<run>/` |

`optimize.py` writes a `control.yaml` pointing at the trained controller.
Use it with `python run_control.py --config models/optimize/<run>/control.yaml ...`
or set `control.checkpoint=` in any config.

## Simulation notes

- All sim processes share one gz partition (`farmsim`). The scripts kill
  leftovers before launching (`farmsim_utils.sim`). Don't run two sims at once.
- Manual launch:
  `ros2 launch farmsim_farm farm_sim.launch.py field:=20250425 teleop_view:=true headless:=false`
- The robot spawns from 2 m and drops onto the terrain mesh.

## Where things came from

Reorganised from `thorvald_ros2_ws` (robot packages, `thorvald_gz_bringup`,
`tools/`) and the field meshes in `ros_src/`. Left behind: the old
recordings and bags (`thorvald_ros2_ws/datasets/sim`, `recordings/`), trained
checkpoints, the smoke-test multitask model in `tools/training`,
`repair_sessions.sh`, `sim_ctl.sh` and the campaign shell scripts (now
`farmsim_utils.sim` and `record_dataset.py`).
