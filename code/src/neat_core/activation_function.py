import math


def modified_sigmoid(x: float) -> float:
    """
    A modified sigmoid function that is used in the original NEAT paper
    :param x: the input value
    :return: the result of the activation function
    """
    return 1 / (1 + math.exp(-4.9 * x))
