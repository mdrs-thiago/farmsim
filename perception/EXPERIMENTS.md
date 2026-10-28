# Four-way row-line experiment

The dataset has fixed train, validation and test sequence splits in
`dataset/row_perception_v1/splits/`. The comparison crosses two model
architectures with two ResNet-18 encoder initializations:

| Run | Image encoder | Temporal features |
|---|---|---|
| `cnn_scratch` | Random weights | None |
| `cnn_lstm_scratch` | Random weights | Causal ConvLSTM at /16 and /32 |
| `cnn_imagenet` | ImageNet-1K weights | None |
| `cnn_lstm_imagenet` | ImageNet-1K weights | Causal ConvLSTM at /16 and /32 |

The decoder and prediction heads start from random weights in every run. All
four runs use the same epoch count, training-frame budget, optimizer, learning
rates, augmentation, seed, and final-epoch checkpoint rule. The CNN receives
single frames; the CNN-LSTM receives ordered 8-frame clips. Runs within the
same architecture and seed use the same sample order and augmentation random
stream. The recurrent model has more parameters and computation, so report
those costs alongside accuracy.

From the workspace root, with a CUDA-capable PyTorch environment:

```bash
bash perception/farmsim_perception/training/run_four_way.sh
```

Set `OUT` to an absolute path for a different output directory, `EPOCHS` for a
different budget, and `SEED` for a repeat. The script refuses to overwrite an
incomplete run. Set `ALLOW_CPU=1` only if a long CPU run is intentional.

The resulting `comparison_test.md` contains clean and augmented test metrics
for all four checkpoints. `analysis_test_cnn_scratch_vs_cnn_lstm_scratch.md`
and its ImageNet counterpart give paired sequence-level comparisons with
bootstrap intervals under clean input and frame dropout. Read
`line_near_err_px`, `line_far_err_px`, `line_far_jitter_px`, lateral and heading
errors together; segmentation and detection are secondary tasks. Keep the test
split untouched while choosing hyperparameters from validation results. For
stronger evidence, repeat the full four-way run with several seeds and compare
the architecture effect within each initialization.

This experiment tests whether visual history helps line estimation. The
ConvLSTM receives images only. It may learn motion cues from those images, but
it cannot establish that it learned the robot's physical dynamics. Testing
that claim would require a separate input and ablation for commands or odometry.

Checkpoints were not carried over from the old workspace; runs write to
`models/row_net/<run>/`.

## World-anchored visual anomalies

`make_anomaly_variant.py` adds irregular puddles, soil ruts, and flat debris to
cached RGB sequences. Their positions come from recorded camera poses and the
baked terrain heightmap, so a patch persists at one world location as the robot
moves. Only pixels labeled as corridor or bare bed are changed. Each output
frame has a uint8 anomaly mask: `0` unchanged, `1` puddle, `2` rut, `3` flat
debris. The masks record the composited pixels at the model's 256×192 input
resolution. The source dataset, row geometry, and semantic targets are kept.

```bash
python -m farmsim_perception.anomalies.make_anomaly_variant --split test
python -m farmsim_perception.training.evaluate \
  --variant dataset/row_anomalies_v1 --models cnn cnn_lstm
```

By default, 70% of sessions are selected for one anomaly; sessions without
enough visible ground remain clean. The rest provide clean controls. Use
`--affected-session-fraction` and `--anomalies-per-sequence` to
change that mix. `--seed` makes it reproducible. A variant has compact caches
at `cache/256x192/<session>.npz`, per-frame PNGs under
`sequences/<session>/masks/anomaly/`, and world polygons and exposure counts in
`sequences/<session>/anomalies.json`, plus a `preview.jpg` contact sheet.
Generate each desired split separately
into the same variant directory. `evaluate.py --variant` compares predictions
on affected frames with predictions from the unchanged source frames, then
writes a separate `comparison_test_thorvald_row_anomalies_v1.md` report with
paired sequence-level bootstrap intervals.

Run `python -m farmsim_perception.anomalies.visualize_anomaly_variant` to refresh
`map_overview.jpg` (all world locations over the survey photo) and
`examples.jpg` (source, edited image, and mask for each anomaly type).

To train on visual anomalies, generate the `train` split into the same variant
directory and pass `--train-variant dataset/row_anomalies_v1` to
`train.py`. Validation stays on source images unless `--val-variant` is given.
Use a distinct `--run-name` for every trained condition.

These anomalies change camera appearance at fixed map coordinates. They are
2D overlays, so they do not change wheel contact, terrain height, shadows, or
robot motion. The mask is exact for the image edit; the polygon projection is
an approximation of physical visibility on uneven terrain. Use the masks as
exposure labels for robustness analysis, not as 3D object segmentation truth.
