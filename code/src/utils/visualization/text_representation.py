from loguru import logger

from neat_core.models.agent import Agent
from neat_core.models.genome import Genome


def print_genome(genome: Genome):
    logger.info("Genome: {}".format(genome.id, genome))
    logger.info("--- Nodes ---")
    for node in genome.nodes:
        logger.info(
            "InnovationNumber: {}, Type: {}, Bias: {}".format(node.innovation_number, node.node_type, node.bias))
    logger.info("--- Connections ---")
    for connection in genome.connections:
        logger.info("{} -> {}, Enabled: {}, InnovationNumber: {}, Weight: {}".
                    format(connection.input_node, connection.output_node, connection.enabled,
                           connection.innovation_number, connection.weight))


def print_agent(agent: Agent):
    logger.info("Agent - Fitness: {}, AdditionalInfo: ".format(agent.fitness, agent.additional_info))
    print_genome(agent.genome)
