import copy


class Config(object):
    """Object to hold the config requirements for an agent/game"""
    def __init__(self):
        self.seed = None
        self.environment = None
        self.requirements_to_solve_game = None
        self.num_episodes_to_run = None
        self.file_to_save_data_results = None
        self.file_to_save_results_graph = None
        self.runs_per_agent = None
        self.visualise_overall_results = None
        self.visualise_individual_results = None
        self.hyperparameters = None
        self.use_GPU = None
        self.overwrite_existing_results_file = None
        self.save_model = False
        self.standard_deviation_results = 1.0
        self.randomise_random_seed = True
        self.show_solution_score = False
        self.debug_mode = False
        self.unpicklized_environment = False
        self.log_directory = "./checkpoints"
        self.task = ''

    def copy(self):
        if self.unpicklized_environment:
            new_config = Config()
            new_config.seed = copy.deepcopy(self.seed)
            new_config.environment = self.environment.copy()
            new_config.requirements_to_solve_game = copy.deepcopy(self.requirements_to_solve_game)
            new_config.num_episodes_to_run = copy.deepcopy(self.num_episodes_to_run)
            new_config.file_to_save_data_results = copy.deepcopy(self.file_to_save_data_results)
            new_config.file_to_save_results_graph = copy.deepcopy(self.file_to_save_results_graph)
            new_config.runs_per_agent = copy.deepcopy(self.runs_per_agent)
            new_config.visualise_overall_results = copy.deepcopy(self.visualise_overall_results)
            new_config.visualise_individual_results = copy.deepcopy(self.visualise_individual_results)
            new_config.hyperparameters = copy.deepcopy(self.hyperparameters)
            new_config.use_GPU = copy.deepcopy(self.use_GPU)
            new_config.overwrite_existing_results_file = copy.deepcopy(self.overwrite_existing_results_file)
            new_config.save_model = copy.deepcopy(self.save_model)
            new_config.standard_deviation_results = copy.deepcopy(self.standard_deviation_results)
            new_config.randomise_random_seed = copy.deepcopy(self.randomise_random_seed)
            new_config.show_solution_score = copy.deepcopy(self.show_solution_score)
            new_config.debug_mode = copy.deepcopy(self.debug_mode)
            new_config.log_directory = copy.deepcopy(self.log_directory)
            new_config.task = copy.deepcopy(self.task)
            return new_config
        else:
            return copy.deepcopy(self)