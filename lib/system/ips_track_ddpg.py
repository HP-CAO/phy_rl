from lib.agent.ddpg import DDPGAgent, DDPGAgentParams
from lib.trainer.trainer_ddpg import DDPGTrainer, DDPGTrainerParams
from lib.system.ips_track import ModelTrackParams, ModelTrackSystem
from utils import write_config


class TrackDDPGParams(ModelTrackParams):
    def __init__(self):
        super().__init__()
        self.agent_params = DDPGAgentParams()
        self.trainer_params = DDPGTrainerParams()


class TrackDDPG(ModelTrackSystem):
    def __init__(self, params: TrackDDPGParams):
        super().__init__(params)
        self.params = params
        if self.params.agent_params.add_actions_observations:
            self.shape_observations += self.params.agent_params.action_observations_dim
        self.agent = DDPGAgent(params.agent_params, self.shape_observations, self.shape_targets, shape_action=1)
        self.trainer = DDPGTrainer(params.trainer_params, self.agent)
        self.agent.initial_model()
        if self.params.stats_params.weights_path is not None:
            self.agent.load_weights(self.params.stats_params.weights_path)

        write_config(params, f"{self.model_stats.log_dir}/config.json")

    def test(self):
        self.evaluation_episode(ep=0, agent=self.agent)
