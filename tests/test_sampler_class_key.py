"""Phase 4.6 — sampler class_key contract (configurable; no silent class-0 bucketing)."""

import pytest

torch = pytest.importorskip("torch")


class _FakeDataset:
    def __init__(self, targets, dist):
        self._targets = targets
        self.class_distribution = dist

    def __len__(self):
        return len(self._targets)

    def __getitem__(self, i):
        return None, self._targets[i]


def test_target_class_id_explicit_key_and_fallbacks():
    from tcip_mcp.pipelines.data.samplers import _target_class_id
    assert _target_class_id({"my_cls": 2}, class_key="my_cls") == 2
    assert _target_class_id({"my_cls": torch.tensor(3)}, class_key="my_cls") == 3
    assert _target_class_id({"label": 1}) == 1                       # classification fallback
    assert _target_class_id({"ranks": 4}) == 4                       # ordinal fallback
    assert _target_class_id({"labels": torch.tensor([5, 6])}) == 5   # detection fallback
    assert _target_class_id({"other": 1}) is None                    # nothing applicable


def test_class_balanced_sampler_honors_custom_class_key():
    from tcip_mcp.pipelines.data.samplers import ClassBalancedSampler
    # Class lives under a non-standard key; without class_key every sample would fall
    # through to None and get a uniform weight (the old silent-class-0 bug).
    targets = [{"my_cls": 0}, {"my_cls": 0}, {"my_cls": 0}, {"my_cls": 1}]
    ds = _FakeDataset(targets, {0: 3, 1: 1})
    s = ClassBalancedSampler(ds, class_key="my_cls")
    assert s._weights[3] > s._weights[0]   # rare class up-weighted vs the common one
