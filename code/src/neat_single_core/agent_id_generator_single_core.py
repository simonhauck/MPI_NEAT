from neat_core.optimizer.generator.agent_id_generator_interface import AgentIDGeneratorInterface


class AgentIDGeneratorSingleCore(AgentIDGeneratorInterface):

    def __init__(self) -> None:
        self.next_id = 0

    def get_agent_id(self) -> int:
        tmp = self.next_id
        self.next_id += 1
        return tmp
