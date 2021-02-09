from minitorch import operators
from hypothesis import given
from hypothesis.strategies import lists
from .strategies import small_floats, assert_close
import math
import pytest


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_add_and_mul(x, y):
    assert_close(operators.mul(x, y), x * y)
    assert_close(operators.add(x, y), x + y)
    assert_close(operators.neg(x), -x)


@pytest.mark.task0_1
@given(small_floats)
def test_sigmoid(z):
    if z >= 0:
        assert operators.sigmoid(z) == 1.0 / (1.0 + math.exp(-z))
    else:
        assert operators.sigmoid(z) == math.exp(z) / (1.0 + math.exp(z))


@pytest.mark.task0_1
@given(small_floats)
def test_relu(a):
    if a > 0:
        assert operators.relu(a) == a
    else:
        assert operators.relu(a) == 0.0


## Task 0.2
## Property Testing


@pytest.mark.task0_2
@given(small_floats, small_floats)
def test_symmetric(x, y):
    assert_close(operators.mul(x, y), operators.mul(y, x))


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_distribute(x, y, z):
    assert_close(operators.mul(x, operators.add(y, z)), operators.add(operators.mul(x, y), operators.mul(x, z)))


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_other(x, y, z):
    assert_close(operators.mul(x, operators.mul(y, z)), operators.mul(operators.mul(x, y), z))


# HIGHER ORDER


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats, small_floats)
def test_zip_with(a, b, c, d):
    assert_close(operators.addLists([a, b], [c, d]), [a + c, b + d])


@pytest.mark.task0_3
@given(
    lists(small_floats, min_size=5, max_size=5),
    lists(small_floats, min_size=5, max_size=5),
)
def test_property(ls1, ls2):
    assert_close(operators.add(math.fsum(ls1), math.fsum(ls2)), math.fsum(operators.addLists(ls1, ls2)))


@pytest.mark.task0_3
@given(lists(small_floats))
def test_sum(ls):
    assert_close(operators.sum(ls), sum(ls))


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats)
def test_prod(x, y, z):
    assert_close(operators.prod([x, y, z]), x * y * z)
