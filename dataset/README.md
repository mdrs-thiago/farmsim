# dataset/

Not in git (only this file is). Local contents:

- `row_perception_v1/`: 496 simulated camera sequences from field 20250425 with
  images, semantic/structure/vegetation masks and per-frame annotations
  (see its own README.md). `record_dataset.py` appends new sessions here.
- `row_anomalies_v1/`: world-anchored puddles/ruts/debris on part of the test
  split (`python -m farmsim_perception.anomalies.make_anomaly_variant`).

The training cache (`cache/256x192/*.npz`) is rebuilt automatically on first
use. The raw recordings and rosbags of these sessions stay in the old
`thorvald_ros2_ws/datasets/sim` workspace.
