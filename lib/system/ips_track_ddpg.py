from lib.agent.ddpg import DDPGAgent, DDPGAgentParams
from lib.trainer.trainer_ddpg import DDPGTrainer, DDPGTrainerParams
from lib.system.ips_track import ModelTrackParams, ModelTrackSystem
from utils import write_config
#from lib.monitor.monitor import plot_trajectory


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

#    def test(self):
#        reset_states = [0.0, 0.0, -0.1, 0.0, False] # todo to set a parameter
#        self.evaluation_episode(ep=0, reset_states=reset_states, agent=self.agent, mode="test")
#        plot_trajectory(self.trajectory_tensor, self.reference_trajectory_tensor)





