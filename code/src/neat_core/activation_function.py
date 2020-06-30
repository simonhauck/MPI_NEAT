import math


def modified_sigmoid_activation(x: float) -> float:
    """
    A modified sigmoid function that is used in the original NEAT paper
    :param x: the input value
    :return: the result of the activation function, Is between 0, 1
    """
    if x >= 100:
        return 1
    elif x <= -100:
        return 0
    else:
        return 1 / (1 + math.exp(-4.9 * x))


def step_activation(x: float) -> float:
    """
    The step function
    :param x: the input value
    :return: 1 if x >= 0 else 0
    """
    return 1.0 if x > 0.0 else 0.0


def sigmoid_activation(x: float) -> float:
    """
    The sigmoid activation function
    :param x: the input value
    :return: the result of the activation function. Is between 0, 1
    """
    if x >= 100:
        return 1
    elif x <= -100:
        return 0
    else:
        return 1.0 / (1.0 + math.exp(-x))


def tanh_activation(x: float) -> float:
    """
    The tanhh activation function
    :param x: the input value
    :return: the result of the activation function. Is between -1, 1
    """
    return math.tanh(x)


def relu_activation(x: float) -> float:
    """
    The rectified linear function
    :param x: the input for the activation function
    :return: the result of the activation function. Is either 0 or x
    """
    return x if x > 0.0 else 0.0
