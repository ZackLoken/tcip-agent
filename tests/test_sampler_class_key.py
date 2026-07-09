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
    # Detection fallback: tensor labels are 1-indexed (cid + 1, background = 0),
    # class_distribution keys are 0-indexed -> shifted back by 1.
    assert _target_class_id({"labels": torch.tensor([5, 6])}) == 4
    assert _target_class_id({"labels": torch.tensor([1])}) == 0      # single-class detection
    assert _target_class_id({"other": 1}) is None                    # nothing applicable
    # Explicit class_key bypasses the shift — documented; don't use class_key="labels"
    # for detection targets.
    assert _target_class_id({"labels": torch.tensor([5, 6])}, class_key="labels") == 5


def test_class_balanced_sampler_honors_custom_class_key():
    from tcip_mcp.pipelines.data.samplers import ClassBalancedSampler
    # Class lives under a non-standard key; without class_key every sample would fall
    # through to None and get a uniform weight (the old silent-class-0 bug).
    targets = [{"my_cls": 0}, {"my_cls": 0}, {"my_cls": 0}, {"my_cls": 1}]
    ds = _FakeDataset(targets, {0: 3, 1: 1})
    s = ClassBalancedSampler(ds, class_key="my_cls")
    assert s._weights[3] > s._weights[0]   # rare class up-weighted vs the common one


def test_class_balanced_sampler_detection_1indexed_labels():
    from tcip_mcp.pipelines.data.samplers import ClassBalancedSampler
    # Detection targets carry 1-indexed tensor labels (cid + 1) while
    # class_distribution is 0-indexed; the weight lookup must land on the right
    # class (previously class k got class k+1's weight, or missed entirely).
    targets = [
        {"labels": torch.tensor([1])},  # class 0 (common)
        {"labels": torch.tensor([1])},
        {"labels": torch.tensor([1])},
        {"labels": torch.tensor([2])},  # class 1 (rare)
    ]
    ds = _FakeDataset(targets, {0: 3, 1: 1})
    s = ClassBalancedSampler(ds)
    assert s._weights[3] > s._weights[0]   # rare class up-weighted
    # Single-class detection: every lookup must hit (weight != default 1.0 miss).
    ds1 = _FakeDataset([{"labels": torch.tensor([1])}] * 2, {0: 2})
    s1 = ClassBalancedSampler(ds1)
    assert torch.allclose(s1._weights, torch.ones(2, dtype=torch.double))
    # (weight 2/(1*2) == 1.0 here by construction; check the lookup key directly)
    from tcip_mcp.pipelines.data.samplers import _target_class_id
    assert _target_class_id(ds1[0][1]) in ds1.class_distribution


def test_oversampler_reshuffles_per_epoch_under_global_seed():
    from tcip_mcp.pipelines.data.samplers import OverSampler
    targets = [{"my_cls": 0}] * 8 + [{"my_cls": 1}]
    ds = _FakeDataset(targets, {0: 8, 1: 1})
    s = OverSampler(ds, min_count=4, class_key="my_cls")
    assert len(s) > len(ds)                # minority class duplicated
    torch.manual_seed(0)
    epoch1, epoch2 = list(s), list(s)
    assert sorted(epoch1) == sorted(epoch2)
    assert epoch1 != epoch2                # order reshuffles across epochs
    torch.manual_seed(0)
    assert list(s) == epoch1               # and is controlled by the global seed
