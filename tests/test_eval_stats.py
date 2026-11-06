from flip7.experiments.evaluate_endgame import permutation_p, required_n, sd_upper80


def test_permutation_p_extremes():
    assert permutation_p([0.9, 0.95, 0.92], [0.1, 0.12, 0.11], "greater") == 1 / 20
    assert permutation_p([0.5, 0.5], [0.5, 0.5], "two-sided") == 1.0


def test_required_n_monotone():
    assert required_n(0.01) < required_n(0.02) < required_n(0.04)
    # sd = mde gives ~17 per arm for 80% power, alpha 0.05
    assert 15 <= required_n(0.03, mde=0.03) <= 19


def test_sd_upper80_wider():
    assert sd_upper80(0.02, 5) > 0.02
