from typing import Union


class Connection(object):

    def __init__(self, id_: int, input_node: int, output_node: int, weight: float, enabled: bool,
                 innovation_number: Union[int, str]) -> None:
        self.id: int = id_
        self.input_node: int = input_node
        self.output_node: int = output_node
        self.weight: float = weight
        self.enabled: bool = enabled
        self.innovation_number: Union[int, str] = innovation_number
