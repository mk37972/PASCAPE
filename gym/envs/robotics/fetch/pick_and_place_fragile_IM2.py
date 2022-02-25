import os
from gym import utils
from gym.envs.robotics import fetch_env_IM


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('hand', 'manipulate_egg_touch_sensors_wall.xml')


class FetchPickAndPlaceFragileEnvIM2(fetch_env_IM.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', pert_type='none', n_actions=3):
        initial_qpos = {
            'robot0:WRJ2': 0.0,
            'robot0:WRJ1': 0.0,
            'robot0:WRJ0': 0.0,
        }
        fetch_env_IM.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, fragile_on=True, stiffness_on=True, pert_type=pert_type, n_actions=n_actions)
        utils.EzPickle.__init__(self)
