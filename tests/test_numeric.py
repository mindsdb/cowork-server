"""cowork.common.numeric: a JSON value read as a finite float."""
import pytest

from cowork.common.numeric import finite_float


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(0, 0.0, id="zero"),
        pytest.param(12, 12.0, id="int"),
        pytest.param(-3, -3.0, id="negative int"),
        pytest.param(0.25, 0.25, id="float"),
        pytest.param(1e308, 1e308, id="largest decade"),
    ],
)
def test_a_real_number_reads_as_a_float(value, expected):
    result = finite_float(value)

    assert result == expected
    assert type(result) is float


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(None, id="null"),
        pytest.param(True, id="true"),
        pytest.param(False, id="false"),
        pytest.param("12.4", id="numeric string"),
        pytest.param([1], id="list"),
        pytest.param({"usd": 1}, id="object"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="inf"),
        pytest.param(float("-inf"), id="-inf"),
        pytest.param(10**400, id="int too large for a float"),
        pytest.param(-(10**400), id="negative int too large for a float"),
    ],
)
def test_anything_but_a_finite_number_reads_as_none(value):
    assert finite_float(value) is None
