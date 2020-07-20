import errno
import os
import pickle
from typing import Dict

from neat_core.models.genome import Genome
from neat_core.optimizer.neat_reporter import NeatReporter


def save_genome_file(path: str, genome: Genome) -> None:
    """
    Store the given genome in the given file
    :param path: path with filename where the genome should be stored
    :param genome: the genome that should be stored
    :return: None
    """

    # Create directory
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(path, 'wb') as output:
        pickle.dump(genome, output, pickle.HIGHEST_PROTOCOL)


def save_genome_and_reporter(path: str, genome: Genome, reporters: Dict[str, NeatReporter]):
    # Create directory
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(path, 'wb') as output:
        store_data = {"genome": genome, "reporters": reporters}
        pickle.dump(store_data, output, pickle.HIGHEST_PROTOCOL)


def load_genome_file(path: str) -> Genome:
    """
    Load a genome from the given file
    :param path: the path to the file with the genome
    :return: the loaded genome
    """
    with open(path, 'rb') as input_file:
        genome = pickle.load(input_file)

    return genome


def load_genome_and_reporter(path: str) -> (Genome, Dict[str, NeatReporter]):
    """
    Load a file that contains a genome as well as a dict with neat reporters
    :param path: path to the file
    :return: the loaded genome and the dict
    """
    with open(path, 'rb') as input_file:
        dict = pickle.load(input_file)

    return dict["genome"], dict["reporters"]
