from typing import Callable

from loguru import logger

from neat_core.models.generation import Generation
from neat_core.optimizer.neat_reporter import NeatReporter
from utils.fitness_evaluation import fitness_evaluation_utils
from utils.persistance import file_save


class CheckPointReporter(NeatReporter):

    def __init__(self, storage_path: str, name_prefix: str, store_func: Callable[[int], bool]) -> None:
        """
        Create a checkpoint reporter that stores the best genome of the selected generations
        :param storage_path: the path to storage folder. Should end with an /
        :param name_prefix: the prefix for the name
        :param store_func: function with the generation number as input. Return true to save the best genome
        """
        self.storage_path: str = storage_path
        self.name_prefix: str = name_prefix
        self.store_func: Callable[[int], bool] = store_func

    def on_generation_evaluation_end(self, generation: Generation) -> None:
        """
        Check if the generation should be saved and create a file with the best genome
        :param generation: the generation evaluated
        :return: None
        """
        if self.store_func(generation.number):
            best_agent = fitness_evaluation_utils.get_best_agent(generation.agents)
            file_name = self.storage_path + self.name_prefix + str(generation.number) + ".genome"
            file_save.save_genome_file(file_name, best_agent.genome)
            logger.info("Saved best genome of generation {} into file: {}".format(generation.number, file_name))

    def on_finish(self, generation: Generation) -> None:
        """
        Store the best genome every time
        :param generation: the last generation evaluated
        :return: None
        """
        best_agent = fitness_evaluation_utils.get_best_agent(generation.agents)
        file_name = self.storage_path + self.name_prefix + "finish" + str(generation.number) + ".genome"
        file_save.save_genome_file(file_name, best_agent.genome)
        logger.info("Saved final best genome of generation {} into file: {}".format(generation.number, file_name))
