class NeatOptimizerCallback(object):

    def on_initialization(self) -> None:
        """
        Called before the first generation is evaluated. With the following generations, this method will not be invoked
        again.
        :return: None
        """
        pass

    def on_generation_evaluation_start(self) -> None:
        """
        Called with the start of the evaluation of each generation.
        :return: None
        """
        pass

    def on_agent_evaluation_start(self) -> None:
        """
        Called with the start of the evaluation of each agent
        :return: None
        """
        pass

    def on_agent_evaluation_end(self) -> None:
        """
        Called with the end of the evaluation of each agent
        :return: None
        """
        pass

    def on_generation_evaluation_end(self) -> None:
        """
        Called with the end of the evaluation of each generation
        :return: None
        """
        pass

    def on_evaluation_stopped(self) -> None:
        """
        Called at the end of the evaluation process.
        :return: None
        """
        pass
