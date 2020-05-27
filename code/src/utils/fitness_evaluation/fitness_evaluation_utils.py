from typing import List

from neat_core.models.agent import Agent


def get_best_agent(agents: List[Agent]) -> Agent:
    """
    Get the best performing genome of the generation
    :param agents: a list with agents
    :return: the best genome, or None if the generation has no members
    """
    best_agent = None
    for agent in agents:
        if best_agent is None or best_agent.fitness < agent.fitness:
            best_agent = agent
    return best_agent
