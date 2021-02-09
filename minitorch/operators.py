import math

## Task 0.1
## Mathematical operators


def mul(x, y):
    return x * y


def id(x):
    return x


def add(x, y):
    return x + y


def neg(x):
    return -x


def lt(x, y):
    if x < y:
        return 1.0
    else:
        return 0.0


def eq(x, y):
    if x == y:
        return 1.0
    else:
        return 0.0


def max(x, y):
    if x > y:
        return x
    else:
        return y


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x):
    if x > 0:
        return x
    else:
        return 0.0


def relu_back(x, y):
    if x > 0:
        return y
    else:
        return 0.0


EPS = 1e-6


def log(x):
    ":math:`f(x) = log(x)`"
    return math.log(x + EPS)


def exp(x):
    ":math:`f(x) = e^{x}`"
    return math.exp(x)


def log_back(a, b):
    return b / (a + EPS)


def inv(x):
    ":math:`f(x) = 1/x`"
    return 1.0 / x


def inv_back(a, b):
    return -(1.0 / a ** 2) * b


## Task 0.3
## Higher-order functions.


def map(fn):
    def z(x):
        a = []
        for i in x:
            a.append(fn(i))
        return a
    return z


def negList(ls):
    return map(neg)(ls)


def zipWith(fn):
    def z(ls1, ls2):
        a = []
        for i in range(0 , len(ls1)):
            b = fn(ls1[i], ls2[i])
            a.append(b)
        return a
    return z


def addLists(ls1, ls2):
    "Add the elements of `ls1` and `ls2` using :func:`zipWith` and :func:`add`"
    return zipWith(add)(ls1, ls2)


def reduce(fn, start):
    def z(ls1):
        b = fn(start, ls1[0])
        v = 1
        while v < len(ls1):
            b = fn(b, ls1[v])
            v += 1
        return b
    return z


def sum(ls):
    if ls == []:
        return []
    else:
        return reduce(add, 0)(ls)


def prod(ls):
    if len(ls) <= 1:
        return ls
    else:
        return reduce(mul, 1)(ls)
