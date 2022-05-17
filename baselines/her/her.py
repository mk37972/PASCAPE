import os

import click
import numpy as np
import json
from mpi4py import MPI

from baselines import logger
from baselines.common import set_global_seeds, tf_util
import tensorflow as tf
from baselines.common.mpi_moments import mpi_moments

from baselines.her.rollout import RolloutWorker
from baselines.her.rollout_NuFingers import RolloutWorker as RolloutNuFingers

import baselines.her.experiment.config as config

def mpi_average(value):
    if not isinstance(value, list):
        value = [value]
    if not any(value):
        value = [0.]
    return mpi_moments(np.array(value))[0]


def train(*, policy, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_path, demo_file, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()

    if save_path:
        latest_policy_path = os.path.join(save_path, 'policy_latest.pkl')
        best_policy_path = os.path.join(save_path, 'policy_best.pkl')
        periodic_policy_path = os.path.join(save_path, 'policy_{}.pkl')

#    logger.info("Training...")
    best_success_rate = -1

    if policy.bc_loss == 1: policy.init_demo_buffer(demo_file) #initialize demo buffer if training with demonstrations

    # num_timesteps = n_epochs * n_cycles * rollout_length * number of rollout workers
    for epoch in range(n_epochs):
        # train
        rollout_worker.clear_history()
        for _ in range(n_cycles):
            episode = rollout_worker.generate_rollouts()
            policy.store_episode(episode)
            for _ in range(n_batches):
                policy.train()
            policy.update_target_net()
            
        # test
        evaluator.clear_history()
        for _ in range(n_test_rollouts):#n_test_rollouts
            evaluator.generate_rollouts()
        
        # np.savez_compressed('NuFingers_2D_Demo_SimStrong10Policy_Disturbed.npz', acs=evaluator.actions_list, obs=evaluator.obs_list, info=evaluator.infos_list, forces=evaluator.forces_list)
        # record logs
        logger.record_tabular('epoch', epoch)
        for key, val in evaluator.logs('test'):
            avg = mpi_average(val)
            logger.record_tabular(key, avg)
            if key == 'test/success_rate':
                policy.success_rate = avg
        for key, val in rollout_worker.logs('train'):
            avg = mpi_average(val)
            logger.record_tabular(key, avg)
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))

        if rank == 0:
            logger.dump_tabular()
        
        # # save the policy if it's better than the previous ones
        # success_rate = mpi_average(evaluator.current_success_rate())
        # if rank == 0 and success_rate >= best_success_rate and save_path:
        #     best_success_rate = success_rate
        #     logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
        #     evaluator.save_policy(best_policy_path)
        #     evaluator.save_policy(latest_policy_path)
        # if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_path:
        #     policy_path = periodic_policy_path.format(epoch)
        #     logger.info('Saving periodic policy to {} ...'.format(policy_path))
        #     evaluator.save_policy(policy_path)

        # # make sure that different threads have different seeds
        # local_uniform = np.random.uniform(size=(1,))
        # root_uniform = local_uniform.copy()
        # MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        # if rank != 0:
        #     assert local_uniform[0] != root_uniform[0]
        
        
#        if epoch > 100: policy.remove_demo = 1
#        policy.n_epoch = np.mean(rollout_worker.success_history)
    return policy


def learn(*, network, env, total_timesteps,
    seed=None,
    eval_env=None,
    replay_strategy='future',
    policy_save_interval=5,
    clip_return=True,
    demo_file=None,
    override_params=None,
    load_path=None,
    save_path=None,
    params=None,
    **kwargs
):
    override_params = override_params or {}
    if MPI is not None:
        rank = MPI.COMM_WORLD.Get_rank()
        num_cpu = MPI.COMM_WORLD.Get_size()

    # Seed everything.
    rank_seed = seed + 1000000 * rank if seed is not None else None
    set_global_seeds(rank_seed)

    # Prepare params.
    params = {
        # env
        'max_u': 1.,  # max absolute value of actions on different coordinates
        # ddpg
        'layers': 3,  # number of layers in the critic/actor networks
        'hidden': 256,  # number of neurons in each hidden layers
        'network_class': 'baselines.her.actor_critic:ActorCritic',
        'Q_lr': 0.001,  # critic learning rate
        'pi_lr': 0.001,  # actor learning rate
        'buffer_size': int(1E6),  # for experience replay
        'polyak': 0.95,  # polyak averaging coefficient
        'action_l2': 1.0,  # quadratic penalty on actions (before rescaling by max_u)
        'clip_obs': 200.,
        'scope': 'ddpg',  # can be tweaked for testing
        'relative_goals': False,
        # training
        'n_cycles': 20,  # per epoch
        'rollout_batch_size': 2,  # per mpi thread
        'n_batches': 40,  # training batches per cycle
        'batch_size': 256,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
        'n_test_rollouts': 10,  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts (10 for Block, Chip)
        'test_with_polyak': False,  # run test episodes with the target network
        # exploration
        'random_eps':0.2,  # percentage of time a random action is taken
        'noise_eps':0.3,  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
        # HER
        'replay_strategy': 'none',  # supported modes: future, none
        'replay_k': 4,  # number of additional goals used for replay, only used if off_policy_data=future
        # normalization
        'norm_eps': 0.01,  # epsilon used for observation normalization
        'norm_clip': 5,  # normalized observations are cropped to this values
    
        'bc_loss': 0, # whether or not to use the behavior cloning loss as an auxilliary loss
        'q_filter': 0, # whether or not a Q value filter should be used on the Actor outputs
        'num_demo': 25, # number of expert demo episodes
        'demo_batch_size': 200, #number of samples to be used from the demonstrations buffer, per mpi thread 128/1024 or 32/256
        'prm_loss_weight': 0.001, #Weight corresponding to the primary loss
        'aux_loss_weight':  0.01, #Weight corresponding to the auxilliary loss also called the cloning loss 0.0078
        'perturb':  kwargs['pert_type'],
        'n_actions':  kwargs['n_actions'],
    }
    # params['replay_strategy'] = replay_strategy
    if env is not None:
        env_name = env.spec.id
        params['env_name'] = env_name
        if env_name in config.DEFAULT_ENV_PARAMS:
            params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    else:
        params['env_name'] = 'NuFingers_Experiment'
    params.update(**override_params)  # makes it possible to override any parameter
    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
        json.dump(params, f)

    if demo_file is not None:
        params['bc_loss'] = 1
        params['q_filter'] = 1
        params['n_cycles'] = 20
        params['random_eps'] = 0.2 # chip,block: 0.1
        params['noise_eps'] = 0.3 # chip,block: 0.1
        params['demo_batch_size'] = 32
        params['batch_size']: 256
    params = config.prepare_params(params)
    params['rollout_batch_size'] = 1
    params.update(kwargs)

    config.log_params(params, logger=logger)

    if num_cpu == 1:
        logger.warn()
        logger.warn('*** Warning ***')
        logger.warn(
            'You are running HER with just a single MPI worker. This will work, but the ' +
            'experiments that we report in Plappert et al. (2018, https://arxiv.org/abs/1802.09464) ' +
            'were obtained with --num_cpu 19. This makes a significant difference and if you ' +
            'are looking to reproduce those results, be aware of this. Please also refer to ' +
            'https://github.com/openai/baselines/issues/314 for further details.')
        logger.warn('****************')
        logger.warn()

    if env is not None:
        dims = config.configure_dims(params)
    else:
        dims = dict(o = 11 + kwargs['n_actions'],
                    u = kwargs['n_actions'],
                    g = 7,
                    info_is_success = 1)
    policy = config.configure_ddpg(dims=dims, params=params, clip_return=clip_return)
    if load_path is not None:
        tf_util.load_variables(load_path)

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    
    eval_env = eval_env or env

    print("NAME={}".format(params['env_name']))
    
    print(rollout_params)
    
    if params['env_name'].find('NuFingers_Experiment') == -1:
        rollout_worker = RolloutWorker(env, policy, dims, logger, monitor=True, **rollout_params)
        evaluator = RolloutWorker(eval_env, policy, dims, logger, **eval_params)
    else:
        rollout_worker = RolloutNuFingers(policy, dims, logger, monitor=True, **rollout_params)
        evaluator = RolloutNuFingers(policy, dims, logger, **eval_params)

    n_cycles = params['n_cycles']
    n_epochs = total_timesteps // n_cycles // rollout_worker.T // rollout_worker.rollout_batch_size
    
    return train(
        save_path=save_path, policy=policy, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], n_batches=params['n_batches'],
        policy_save_interval=policy_save_interval, demo_file=demo_file)


@click.command()
@click.option('--env', type=str, default='FetchReach-v1', help='the name of the OpenAI Gym environment that you want to train on')
@click.option('--total_timesteps', type=int, default=int(5e5), help='the number of timesteps to run')
@click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
@click.option('--policy_save_interval', type=int, default=5, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
@click.option('--replay_strategy', type=click.Choice(['future', 'none']), default='future', help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')
@click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
@click.option('--demo_file', type=str, default = 'PATH/TO/DEMO/DATA/FILE.npz', help='demo data file path')
def main(**kwargs):
    learn(**kwargs)


if __name__ == '__main__':
    main()
