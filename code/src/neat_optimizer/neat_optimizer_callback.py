class NeatOptimizerCallback(object):

    def on_initialization(self):
        """
        Called before the first generation is evaluated. With the following generations, this method will not be invoked
        again.
        :return: None
        """
        pass

    def on_generation_evaluation_start(self):
        """
        Called with the start of the evaluation of each generation.
        :return: None
        """
        pass

    def on_agent_evaluation_start(self):
        """
        Called with the start of the evaluation of each agent
        :return: None
        """
        pass

    def on_agent_evaluation_end(self):
        """
        Called with the end of the evaluation of each agent
        :return:
        """
        pass

    def on_generation_evaluation_end(self):
        """
        Called with the end of the evaluation of each generation
        :return:
        """
        pass
