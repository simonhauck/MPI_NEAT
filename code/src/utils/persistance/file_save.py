import pickle

from neat_core.models.genome import Genome


def save_genome_file(path: str, genome: Genome) -> None:
    """
    Store the given genome in the given file
    :param path: path with filename where the genome should be stored
    :param genome: the genome that should be stored
    :return: None
    """
    with open(path, 'wb') as output:
        pickle.dump(genome, output, pickle.HIGHEST_PROTOCOL)


def load_genome_file(path: str) -> Genome:
    """
    Load a genome from the given file
    :param path: the path to the file with the genome
    :return: the loaded genome
    """
    with open(path, 'rb') as input_file:
        genome = pickle.load(input_file)

    return genome
