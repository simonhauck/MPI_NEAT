from loguru import logger
from mpi4py import MPI

from neat_core.models.agent import Agent
from neat_core.optimizer.challenge import Challenge
from neural_network.basic_neural_network import BasicNeuralNetwork

initialized = False

comm = None
rank = None
size = None
name = None
challenge: Challenge = None


def setup(c: Challenge) -> None:
    """
    Setup a MPI worker with the given parameters
    :param c: the challenge for the worker
    :return: None
    """
    global initialized, comm, rank, size, name, challenge
    comm = MPI.COMM_WORLD
    name = MPI.Get_processor_name()
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Set up the challenge
    _setup_challenge(c)

    initialized = True
    logger.info("Worker Setup - Name: {}, Size: {}, Rank {}/{}", name, size, rank, size - 1)


def clean_up() -> None:
    """
    Cleanup all challenges etc
    :return:
    """
    _clean_up_challenge()


def _setup_challenge(c: Challenge) -> None:
    """
    Set the given challenge as global and initialize it
    :param c: the challenge, can not be None
    :return: None
    """
    global challenge
    challenge = c
    challenge.initialization()


def _clean_up_challenge():
    global initialized, challenge

    if not initialized:
        logger.error("Worker is not initialized")
        return

    challenge.clean_up()


def evaluate_agent(agent: Agent):
    global challenge

    challenge.before_evaluation()

    nn = BasicNeuralNetwork()
    nn.build(agent.genome)

    fitness, additional_info = challenge.evaluate(nn)

    challenge.after_evaluation()

    return fitness, additional_info
