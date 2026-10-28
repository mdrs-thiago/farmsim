import torch

from farmsim_perception.models.line_models import CNNLSTMLineModel
from farmsim_perception.line_data import split_sessions
from farmsim_perception.training.train_line_models import masked_loss


def test_temporal_model_is_causal():
    torch.manual_seed(1)
    model = CNNLSTMLineModel().eval()
    first = torch.randn(1, 8, 3, 32, 32)
    second = first.clone(); second[:, 4:] = torch.randn_like(second[:, 4:])
    with torch.no_grad():
        a, b = model(first), model(second)
    assert torch.equal(a[:, :4], b[:, :4])


def test_three_way_split_is_disjoint_and_stable_when_data_is_added():
    sessions = [f'/data/session_{i}' for i in range(100)]
    before = [set(x) for x in split_sessions(sessions, seed=9)]
    after = [set(x) for x in split_sessions(sessions + ['/data/new'], seed=9)]
    assert not (before[0] & before[1] or before[0] & before[2] or before[1] & before[2])
    for old, new in zip(before, after):
        assert old == new.intersection(sessions)


def test_masked_loss_ignores_invisible_point():
    prediction = torch.zeros(1, 1, 4, requires_grad=True)
    target = torch.tensor([[[0.0, 0.0, 100.0, 100.0]]])
    visible = torch.tensor([[[True, False]]])
    loss = masked_loss(prediction, target, visible)
    assert loss.item() == 0.0

