import math


def modified_sigmoid(x: float) -> float:
    """
    A modified sigmoid function that is used in the original NEAT paper
    :param x: the input value
    :return: the result of the activation function
    """
    return 1 / (1 + math.exp(-4.9 * x))


def step_function(x: float) -> float:
    """
    The step function
    :param x: the input value
    :return: 1 if x >= 0 else 0
    """
    if x > 0.0:
        return 1.0
    else:
        return 0.0
