from typing import Union


class Connection(object):

    def __init__(self, innovation_number: Union[int, str], input_node: int, output_node: int, weight: float,
                 enabled: bool,
                 ) -> None:
        """
        :param innovation_number: the assigned innovation number for this connection.
        :param input_node: the innovation number of the input node for this connection
        :param output_node: the innovation number of the output node for this connection
        :param weight: the weight of the connection
        :param enabled: true, if the connection is active and should be used in the neural network
        """
        self.innovation_number: Union[int, str] = innovation_number
        self.input_node: int = input_node
        self.output_node: int = output_node
        self.weight: float = weight
        self.enabled: bool = enabled
