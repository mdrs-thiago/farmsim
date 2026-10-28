# models/

Trained weights, not in git. Default locations the scripts use:

- `row_net/<run>/last.pt`: row-perception CNN / CNN-LSTM (`python -m farmsim_perception.training.train`)
- `line_benchmark*/`: line-estimator benchmarks (`python -m farmsim_perception.training.train_line_models[_v2]`)
- `optimize/<run>/`: `optimize.py` output (`perception.pt`, `controller.pt`, `control.yaml`)
