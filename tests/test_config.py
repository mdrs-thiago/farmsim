from farmsim_utils.config import apply_overrides, deep_merge, load_yaml


def test_include_and_override():
    cfg = load_yaml('run_control.yaml')
    assert cfg['control']['strategy'] == 'pd' and 'robot' in cfg and 'farm' in cfg
    apply_overrides(cfg, ['control.strategy=fuzzy', 'sim.headless=false', 'control.speed.v_nominal=0.4',
                          'new.key=[1, 2]'])
    assert cfg['control']['strategy'] == 'fuzzy'
    assert cfg['sim']['headless'] is False
    assert cfg['control']['speed']['v_nominal'] == 0.4
    assert cfg['new']['key'] == [1, 2]


def test_optimize_overrides_included_strategy():
    assert load_yaml('optimize.yaml')['control']['strategy'] == 'fuzzy'
    assert load_yaml('optimize.yaml')['control']['gains']['pd']['k_offset'] == 0.3


def test_deep_merge_does_not_mutate():
    a = {'x': {'y': 1, 'z': 2}}
    b = deep_merge(a, {'x': {'y': 5}})
    assert b == {'x': {'y': 5, 'z': 2}} and a['x']['y'] == 1
