import numpy as np

from research.cove import GRID, LOCAL, Visual, coupling, encode, extract, fit_coupling, real_code
from research.experiment import inverse, ridge, split
from research.things import checksum


def test_visual_is_causal_and_accepts_constant_and_video():
    rng = np.random.default_rng(7)
    frames = rng.integers(0, 256, (5, 64, 64, 3), dtype=np.uint8)
    video = extract(frames)
    prefix = extract(frames[:2])
    np.testing.assert_allclose(video.drive[:2], prefix.drive)
    np.testing.assert_allclose(encode(video)[:2], encode(prefix))
    assert np.max(abs(encode(video))) < 1
    blank = extract(np.full((1, 64, 64), 128, dtype=np.uint8))
    assert blank.contrast.max() < 1e-7
    assert np.max(abs(blank.drive)) < .001


def test_fit_is_trainable_and_graph_is_row_stochastic():
    rng = np.random.default_rng(12)
    drive = rng.normal(size=(5, GRID**2, 4)) / 10
    contrast = rng.uniform(0, .3, size=(5, GRID**2))
    data = Visual(drive, contrast, None, None)
    weights = coupling(contrast, fit_coupling([data]))
    np.testing.assert_allclose(weights.sum(axis=-1), 1)
    np.testing.assert_array_equal(weights[:, ~LOCAL], 0)
    np.testing.assert_array_equal(encode(data), encode(data))


def test_split_never_leaks_repeated_groups():
    groups = np.repeat(np.arange(15), 3)
    train, validation, test = [set(groups[index]) for index in split(groups, 17)]
    assert not (train & validation or train & test or validation & test)
    assert len(train | validation | test) == 15


def test_inverse_improves_known_visual_objective_and_respects_bounds():
    frames = np.random.default_rng(29).integers(0, 256, (15, 64, 64), dtype=np.uint8)
    visual = extract(frames)
    code = real_code(encode(visual))
    decoder = ridge(code[:10], visual.pixels[:10], .01)
    neural = ridge(code[:10], visual.pixels[:10, :3], .1)
    before = code[10:]
    after = inverse(before, visual.pixels[10:], decoder, neural, coupling(visual.contrast[10:], (2., 0.)))
    loss = lambda values: np.mean((values @ decoder[0] + decoder[1] - visual.pixels[10:])**2)
    assert loss(after) < loss(before)
    assert np.linalg.norm((after - before).reshape(-1, GRID**2, 2, 4), axis=2).max() <= .15 + 1e-12
    assert np.linalg.norm(after.reshape(-1, GRID**2, 2, 4), axis=2).max() <= 1 + 1e-12


def test_checksum_failure(tmp_path):
    import pytest
    path = tmp_path / "unknown.npy"
    path.write_bytes(b"untrusted")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        checksum(path, "0" * 64)
