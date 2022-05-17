#!/usr/bin/sh

import sys
import re
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
from tfdeterminism import patch
patch()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import copy

from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module

from baselines.common import set_global_seeds

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def train(args, extra_args):
    if args.env.find('NuFingers_Experiment') == -1:
        env_type, env_id = get_env_type(args)
        alg_kwargs = get_learn_function_defaults(args.alg, env_type)
        print('env_type: {}'.format(env_type))
    else:
        try: alg_kwargs = {'demo_file': extra_args['demo_file'], 'pert_type': args.perturb, 'n_actions': args.algdim, 'network': 'mlp'}
        except: alg_kwargs = {'pert_type': args.perturb, 'n_actions': args.algdim, 'network': 'mlp'}

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    args_eval = copy.deepcopy(args)
    args.eval_env = False
    args_eval.eval_env = False
    print(args)
    print(args_eval)
    
    extra_args['pert_type'] = args.perturb
    extra_args['n_actions'] = args.algdim
    alg_kwargs.update(extra_args)
    
    env = None
    eval_env = None
    if args.env.find('NuFingers_Experiment') == -1:
        env = build_env(args)
        eval_env = build_env(args_eval)
        if args.save_video_interval != 0:
            env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"), record_video_trigger=lambda x: x % args.save_video_interval == 0, video_length=args.save_video_length)
        print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))
    else:
        print('Training NuFingers using {} with arguments \n{}'.format(args.alg, alg_kwargs))

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    model = learn(
        env=env,
        eval_env=eval_env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env, eval_env


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed
    env_kwargs = dict(pert_type=args.perturb, n_actions=args.algdim, eval_env=args.eval_env)

    env_type, env_id = get_env_type(args)

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)

    else:
        config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        get_session(config=config)

        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, args.num_env or 1, seed, env_kwargs=env_kwargs, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations)

        if env_type == 'mujoco':
            env = VecNormalize(env, use_tf=True)

    return env


def get_env_type(args):
    env_id = args.env

    if args.env_type is not None:
        return args.env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs



def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    # args.eval_env = False
    extra_args = parse_cmdline_kwargs(unknown_args)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.log_path, format_strs=[])
    
    model, env, eval_env = train(args, extra_args)
    
    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)

    if args.play:
        logger.log("Running trained model")
        
        
        seed = 100
        np.random.seed(seed)
        set_global_seeds(seed)
        
        obs = eval_env.reset()

        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))

        episode_rew = np.zeros(eval_env.num_envs) if isinstance(eval_env, VecEnv) else np.zeros(1)
        forces_list = []
        distance_list = []
        episodeFor = []
        episodeDis = []
        
        # actions_list = []
        # obs_list =[]
        # infos_list = []
        # episodeAct = []
        # episodeObs = []
        # episodeInfo = []
        max_force = 0
        max_acc = 0
        for k in range(50000):
            if state is not None:
                actions, _, state, _ = model.step(obs,S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)
                
                
            if args.env == 'NuFingersRotate-v1' or args.env == 'NuFingersRotate-v2':
                distance = np.linalg.norm(obs['achieved_goal'][0][:1] - obs['desired_goal'][0][:1])
                force = - eval_env.envs[0].env.prev_lforce- eval_env.envs[0].env.prev_rforce
            elif args.env == 'relocate-v0':
                distance = np.linalg.norm(obs['achieved_goal'][0][6:9] - obs['desired_goal'][0][6:9])
                force = - eval_env.envs[0].env.prev_lforce- eval_env.envs[0].env.prev_rforce
            elif args.env == 'CheolFingersManipulate-v1':
                distance = 0.0
                force = -eval_env.envs[0].env.prev_force
            elif args.env == 'CheolFingersSearch-v1':
                distance = 0.0
                force = eval_env.envs[0].env.prev_oforce
            elif args.env == 'CheolFingersLiquid-v1':
                distance = eval_env.envs[0].env.obj_acc
                force = eval_env.envs[0].env.prev_oforce
            else: 
                distance = np.linalg.norm(obs['achieved_goal'][0][:3] - obs['desired_goal'][0][:3])
                if args.env == 'FetchPickAndPlaceFragile-v1' or args.env == 'FetchPickAndPlaceFragile-v5':
                    force = - eval_env.envs[0].env.prev_lforce- eval_env.envs[0].env.prev_rforce
                elif args.env == 'FetchPickAndPlaceFragile-v2' or args.env == 'FetchPickAndPlaceFragile-v6':
                    force = eval_env.envs[0].env.prev_oforce
                elif args.env == 'FetchPickAndPlaceFragile-v3':
                    force = - eval_env.envs[0].env.prev_lforce- eval_env.envs[0].env.prev_rforce
                    acc = np.sqrt(eval_env.envs[0].env.obj_acc*eval_env.envs[0].env.obj_acc)
            
            obs, rew, done, info = eval_env.step(actions)
            # episodeInfo.append(info[0])
            # episodeAct.append(actions[0])
            # episodeObs.append(dict(observation=obs['observation'][0],
            #                        achieved_goal=obs['achieved_goal'][0],
            #                        desired_goal=obs['desired_goal'][0]))
            episode_rew += rew
            if args.filename is None: eval_env.render()
            done_any = done.any() if isinstance(done, np.ndarray) else done
            if force > max_force: max_force = force
            if distance > max_acc: max_acc = distance
            
            episodeFor.append(force)
            episodeDis.append(distance)
            
            if done_any:
                for i in np.nonzero(done)[0]:
                    print('episode_rew={}'.format(episode_rew))
                    episode_rew[i] = 0
                if args.filename is None:
                    print("Maximum force in the episode: {}".format(max_force))
                    print("Maximum acceleration of the object: {}".format(max_acc))
                    max_force = 0
                    max_acc = 0
                
                forces_list.append(episodeFor)
                distance_list.append(episodeDis)
                episodeFor = []
                episodeDis = []
                
                # actions_list.append(episodeAct)
                # obs_list.append(episodeObs)
                # infos_list.append(episodeInfo)
                # episodeInfo = []
                # episodeAct = []
                # episodeObs = []
                
                seed += 1000
                np.random.seed(seed)
                set_global_seeds(seed)
                
        
                    
#        fileName = "StiffnessCtrl_Demo"
#        fileName += "_" + "random"
#        fileName += "_" + str(1000)
#        fileName += ".npz"
    
#        np.savez_compressed(fileName, acs=actions_list, obs=obs_list, info=infos_list) # save the file
        if args.filename is not None: 
            fileName = args.filename
            fileName += ".npz"
            np.savez_compressed(fileName, force=forces_list, dist=distance_list)

    try: eval_env.close()
    except: pass

    return model

if __name__ == '__main__':
    main(sys.argv)
