import math
import pytest
from src.certs.empirical import bonferroni_cell_budget, hoeffding_radius


def test_bonferroni_exact():
    assert bonferroni_cell_budget(0.05, 6) == pytest.approx(0.05 / 6, abs=1e-15)


def test_delta_cell_less_than_global():
    for C in (2, 4, 6, 12):
        dc = bonferroni_cell_budget(0.05, C)
        assert dc < 0.05


def test_invalid_inputs():
    with pytest.raises(ValueError):
        bonferroni_cell_budget(0.0, 6)
    with pytest.raises(ValueError):
        bonferroni_cell_budget(1.0, 6)
    with pytest.raises(ValueError):
        bonferroni_cell_budget(0.05, 0)


def test_hoeffding_radius_four_event_union():
    # eps = sqrt(log(4/delta_cell)/(2M)).
    dc = 0.05 / 6
    M = 250
    assert hoeffding_radius(dc, M) == pytest.approx(math.sqrt(math.log(4 / dc) / (2 * M)), abs=1e-15)
