import pytest

from realtime_merc.metrics import weighted_f1


def test_weighted_f1_known_example():
    # Class 0: F1=2/3, support=2. Class 1: F1=4/5, support=2.
    assert weighted_f1([0, 0, 1, 1], [0, 1, 1, 1]) == pytest.approx(11 / 15)


def test_weighted_f1_empty_and_invalid_input():
    assert weighted_f1([], []) == 0.0
    with pytest.raises(ValueError):
        weighted_f1([0], [7])
    with pytest.raises(ValueError):
        weighted_f1([0], [])
