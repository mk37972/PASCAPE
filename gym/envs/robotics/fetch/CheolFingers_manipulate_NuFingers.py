import os
from gym import utils
from gym.envs.robotics import CheolFingers_env_manipulate


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('CheolFingers', 'CheolFingersEnv.xml')


class CheolFingersManipulateEnv(CheolFingers_env_manipulate.CheolFingersEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', pert_type='none', n_actions=3, eval_env=False):
        initial_qpos = {
                'Joint_1_L' : 0.0,
                'Joint_2_L' : 0.0,
                'Joint_1_R' : 0.0,
                'Joint_2_R' : 0.0,
        }
        CheolFingers_env_manipulate.CheolFingersEnv.__init__(
            self, MODEL_XML_PATH, n_substeps=20, target_range=0.7853981633974483, distance_threshold=0.02,
            initial_qpos=initial_qpos, reward_type=reward_type, pert_type=pert_type, n_actions=n_actions, eval_env=eval_env)
        utils.EzPickle.__init__(self)
