"""A broken backend must not look like perfect agreement in the parity report."""

import pytest

from evaluation.vision.device_parity import compare


def _report(kind, p):
    answer = (
        {"type": "noul", "noul": p}
        if kind == "noul"
        else {"type": "choice", "probabilities": {"red": p, "blue": 1 - p}}
    )
    return {"responses": {"image": {"usage": {}, "answers": {"color": answer}}}}


@pytest.mark.parametrize("kind", ["noul", "choice"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("side", ["reference", "actual"])
def test_nonfinite_probabilities_fail_comparison(kind, value, side):
    reports = {"reference": _report(kind, 0.9), "actual": _report(kind, 0.9)}
    reports[side] = _report(kind, value)
    with pytest.raises(ValueError, match=f"image/color: {side}.*finite"):
        compare(**reports)


def test_comparison_reports_probability_and_answer_changes():
    result = compare(_report("choice", 0.9), _report("choice", 0.3))
    assert result["max_abs_dp"] == pytest.approx(0.6)
    assert result["changed_answers"] == 1
    assert result["answers"] == 1
