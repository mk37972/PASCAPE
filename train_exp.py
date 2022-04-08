#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:36:15 2020

@author: mincheol
"""
from baselines import run
import mpi4py

# defaultargs = ['--alg=her','--env=relocate-v0', '--num_timesteps=0e5', '--play']
# for dim in [6]:
#     for seed in [1000]:
#         # loadpath = '--load_path=./hand_dapg/dapg/policies/relocate.pb'
#         savepath = '--save_path=./models/relocate/fpp_demo25bad{}dim_{}'.format(dim,seed)
#         # demofile = '--demo_file=./hand_dapg/dapg/utils/demo_data.npz'
#         logpath = '--log_path=./models/relocate/fpp_demo25bad{}dim_{}_log'.format(dim,seed)
#         perturb = '--perturb=none'
#         algdim = '--algdim={}'.format(dim)
# #        if seed >= 100 and seed < 1000: seed = 10
# #        elif seed >= 1000 and seed < 10000: seed = 100
# #        elif seed >= 10000 and seed < 100000: seed = 1000
        
#         finalargs = defaultargs + [savepath, logpath, perturb, algdim, '--seed={}'.format(seed)]
        
#         run.main(finalargs)

# defaultargs = ['--alg=her','--env=FetchPickAndPlaceFragile-v3', '--num_timesteps=3e5']
# for dim in [6]:
#     for seed in [750]:
#         savepath = '--save_path=./models/animate/harsh_65/PassiveCtrl_randomly_heavy/fpp_demo25bad{}dim_{}'.format(dim,seed)
#         demofile = '--demo_file=./gym_adjustments/data_fetch_animate_random_25_bad{}dim.npz'.format(dim)
#         logpath = '--log_path=./models/animate/harsh_65/PassiveCtrl_randomly_heavy/fpp_demo25bad{}dim_{}_log'.format(dim,seed)
#         perturb = '--perturb=delay'
#         algdim = '--algdim={}'.format(dim)
# #        if seed >= 100 and seed < 1000: seed = 10
# #        elif seed >= 1000 and seed < 10000: seed = 100
# #        elif seed >= 10000 and seed < 100000: seed = 1000
        
#         finalargs = defaultargs + [savepath, demofile, logpath, perturb, algdim, '--seed={}'.format(seed)]
        
#         run.main(finalargs)
     
# defaultargs = ['--alg=her','--env=FetchPickAndPlaceFragile-v2', '--num_timesteps=1e5']
# for dim in [3]:
#     for seed in [10,500,1000]:
#         savepath = '--save_path=./models/chip/harsh_85/For_IM/fpp_demo25bad{}dim_{}'.format(dim,seed)
#         demofile = '--demo_file=./gym_adjustments/data_chip_vel_random_25_bad{}dim.npz'.format(dim)
#         logpath = '--log_path=./models/chip/harsh_85/For_IM/fpp_demo25bad{}dim_{}_log'.format(dim,seed)
#         perturb = '--perturb=delay'
#         algdim = '--algdim={}'.format(dim)
# #        if seed >= 100 and seed < 1000: seed = 10
# #        elif seed >= 1000 and seed < 10000: seed = 100
# #        elif seed >= 10000 and seed < 100000: seed = 1000
        
#         finalargs = defaultargs + [savepath, demofile, logpath, perturb, algdim, '--seed={}'.format(seed)]
        
#         run.main(finalargs)
     
# defaultargs = ['--alg=her','--env=FetchPickAndPlaceFragile-v1', '--num_timesteps=1e5']
# for dim in [4]:
#     for seed in [10,500,1000]:
#         savepath = '--save_path=./models/block/test/fpp_demo25bad{}dim_{}'.format(dim,seed)
#         demofile = '--demo_file=./gym_adjustments/data_fetch_random_25_bad{}dim.npz'.format(dim)
#         logpath = '--log_path=./models/block/test/fpp_demo25bad{}dim_{}_log'.format(dim,seed)
#         perturb = '--perturb=delay'
#         algdim = '--algdim={}'.format(dim)
# #        if seed >= 100 and seed < 1000: seed = 10
# #        elif seed >= 1000 and seed < 10000: seed = 100
# #        elif seed >= 10000 and seed < 100000: seed = 1000
        
#         finalargs = defaultargs + [savepath, demofile, logpath, perturb, algdim, '--seed={}'.format(seed)]
        
#         run.main(finalargs)

# defaultargs = ['--alg=her','--env=FetchPickAndPlaceFragile-v5', '--num_timesteps=5e4']
# for dim in [2]:
#     for seed in [10,500,1000]:
#         savepath = '--save_path=./models/block/harsh_65/For_IM/RL/fpp_demo25bad{}dim_{}'.format(dim,seed)
#         demofile = '--demo_file=./gym_adjustments/data_fetch_random_25_bad{}dim.npz'.format(dim)
#         logpath = '--log_path=./models/block/harsh_65/For_IM/RL/fpp_demo25bad{}dim_{}_log'.format(dim,seed)
#         perturb = '--perturb=delay'
#         algdim = '--algdim={}'.format(dim)
# #        if seed >= 100 and seed < 1000: seed = 10
# #        elif seed >= 1000 and seed < 10000: seed = 100
# #        elif seed >= 10000 and seed < 100000: seed = 1000
        
#         finalargs = defaultargs + [savepath, logpath, perturb, algdim, '--seed={}'.format(seed)]
        
#         run.main(finalargs)

# defaultargs = ['--alg=her','--env=FetchPickAndPlaceFragile-v6', '--num_timesteps=1e5']
# for dim in [2]:
#     for seed in [10,500,1000]:
#         savepath = '--save_path=./models/chip/harsh_85/For_IM/RL/fpp_demo25bad{}dim_{}'.format(dim,seed)
#         demofile = '--demo_file=./gym_adjustments/data_chip_vel_random_25_bad{}dim.npz'.format(dim)
#         logpath = '--log_path=./models/chip/harsh_85/For_IM/RL/fpp_demo25bad{}dim_{}_log'.format(dim,seed)
#         perturb = '--perturb=delay'
#         algdim = '--algdim={}'.format(dim)
# #        if seed >= 100 and seed < 1000: seed = 10
# #        elif seed >= 1000 and seed < 10000: seed = 100
# #        elif seed >= 10000 and seed < 100000: seed = 1000
        
#         finalargs = defaultargs + [savepath, logpath, perturb, algdim, '--seed={}'.format(seed)]
        
#         run.main(finalargs)

# defaultargs = ['--alg=her','--env=NuFingers_Experiment', '--num_timesteps=1e6']
# for dim in [2]:
#     for seed in [10]:
#         savepath = '--save_path=./models/NuFingers/harsh_65/Strong/fpp_demo25bad{}dim_{}'.format(dim,seed)
#         loadpath = '--load_path=./models/NuFingers/harsh_65/Strong/Sim_NuFingers_bad{}dim_{}'.format(dim,seed)
#         demofile = '--demo_file=./NuFingers_{}D_Demo_SimPolicy.npz'.format(dim)
#         logpath = '--log_path=./models/NuFingers/harsh_65/Strong/fpp_demo25bad{}dim_{}_log'.format(dim,seed)
#         perturb = '--perturb=none'
#         algdim = '--algdim={}'.format(dim)
        
#         finalargs = defaultargs + [savepath, loadpath, logpath, perturb, algdim, '--seed={}'.format(seed)]
        
#         run.main(finalargs)

# defaultargs = ['--alg=her','--env=NuFingersRotate-v1', '--num_timesteps=1e5']
# for dim in [2]:
#     for seed in [10,500,1000]:
#         savepath = '--save_path=./models/NuFingers/harsh_65/For_IM/Sim_NuFingers_bad{}dim_{}'.format(dim,seed)
#         # loadpath = '--load_path=./models/NuFingers/harsh_65/Sim_NuFingers_bad{}dim_{}'.format(dim,seed)
#         demofile = '--demo_file=./data_NuFingers_domain_random_25_bad{}dim.npz'.format(dim)
#         logpath = '--log_path=./models/NuFingers/harsh_65/For_IM/NuFingers_bad{}dim_{}_log'.format(dim,seed)
#         perturb = '--perturb=delay'
#         algdim = '--algdim={}'.format(dim)
        
#         finalargs = defaultargs + [savepath, demofile, logpath, perturb, algdim, '--seed={}'.format(seed)]
        
#         run.main(finalargs)
        
# defaultargs = ['--alg=her','--env=NuFingersRotate-v2', '--num_timesteps=1e5']
# for dim in [2]:
#     for seed in [1000]:
#         savepath = '--save_path=./models/NuFingers/harsh_65/For_IM/RL/Sim_NuFingers_bad{}dim_{}'.format(dim,seed)
#         # loadpath = '--load_path=./models/NuFingers/harsh_65/Sim_NuFingers_bad{}dim_{}'.format(dim,seed)
#         demofile = '--demo_file=./data_NuFingers_domain_random_25_bad{}dim.npz'.format(dim)
#         logpath = '--log_path=./models/NuFingers/harsh_65/For_IM/RL/NuFingers_bad{}dim_{}_log'.format(dim,seed)
#         perturb = '--perturb=delay'
#         algdim = '--algdim={}'.format(dim)
        
#         finalargs = defaultargs + [savepath, logpath, perturb, algdim, '--seed={}'.format(seed)]
        
#         run.main(finalargs)
        
# defaultargs = ['--alg=her','--env=CheolFingersManipulate-v1', '--num_timesteps=4e5']
# for dim in [5]:
#     for seed in [100,1000]:
#         savepath = '--save_path=./PASCAPE/models/Dark/Manipulate/Delayed_env/PASCAPE_10_2/Sim_NuFingers_bad{}dim_{}'.format(dim,seed)
#         # loadpath = '--load_path=./models/Dark/ideal_65/Sim_NuFingers_bad{}dim_{}'.format(dim,seed)
#         demofile = '--demo_file=./PASCAPE/DarkManipulationStiffDemo_{}D.npz'.format(dim)
#         logpath = '--log_path=./PASCAPE/models/Dark/Manipulate/Delayed_env/PASCAPE_10_2/NuFingers_bad{}dim_{}_log'.format(dim,seed)
#         perturb = '--perturb=none'
#         algdim = '--algdim={}'.format(dim)
#         eval_env = '--eval_env=False'
        
#         finalargs = defaultargs + [savepath, demofile, logpath, perturb, algdim, eval_env, '--seed={}'.format(seed)]
        
#         run.main(finalargs)
        
defaultargs = ['--alg=her','--env=CheolFingersSearch-v1', '--num_timesteps=5e4']
for dim in [1]:
    for seed in [10,100,1000]:
        savepath = '--save_path=./models/Dark/Search/POS/Sim_NuFingers_bad{}dim_{}'.format(dim,seed)
        # loadpath = '--load_path=./models/Dark/ideal_65/Sim_NuFingers_bad{}dim_{}'.format(dim,seed)
        demofile = '--demo_file=./PASCAPE/DarkSearchDemo_{}D.npz'.format(dim)
        logpath = '--log_path=./models/Dark/Search/POS/NuFingers_bad{}dim_{}_log'.format(dim,seed)
        perturb = '--perturb=none'
        algdim = '--algdim={}'.format(dim)
        eval_env = '--eval_env=False'
        
        finalargs = defaultargs + [savepath, demofile, logpath, perturb, algdim, eval_env, '--seed={}'.format(seed)]
        
        run.main(finalargs)

# defaultargs = ['--alg=her','--env=CheolFingersLiquid-v1', '--num_timesteps=1e5']
# for dim in [1]:
#     for seed in [10,100,1000]:
#         savepath = '--save_path=./models/Liquid/Sim_NuFingers_bad{}dim_{}'.format(dim,seed)
#         # loadpath = '--load_path=./models/Dark/ideal_65/Sim_NuFingers_bad{}dim_{}'.format(dim,seed)
#         demofile = '--demo_file=./LiquidDemo_{}D.npz'.format(dim)
#         logpath = '--log_path=./models/Liquid/NuFingers_bad{}dim_{}_log'.format(dim,seed)
#         perturb = '--perturb=none'
#         algdim = '--algdim={}'.format(dim)
#         eval_env = '--eval_env=False'
        
#         finalargs = defaultargs + [savepath, demofile, logpath, perturb, algdim, eval_env, '--seed={}'.format(seed)]
        
#         run.main(finalargs)
        
# defaultargs = ['--alg=her','--env=NuFingersRotate-v1', '--num_timesteps=1e5']
# for dim in [2]:
#     for seed in [1000]:
#         savepath = '--save_path=./models/NuFingers/harsh_65/For_IM/RL/Sim_NuFingers_bad{}dim_{}'.format(dim,seed)
#         # loadpath = '--load_path=./models/NuFingers/harsh_65/Sim_NuFingers_bad{}dim_{}'.format(dim,seed)
#         demofile = '--demo_file=./data_NuFingers_domain_random_25_bad{}dim.npz'.format(dim)
#         logpath = '--log_path=./models/NuFingers/harsh_65/For_IM/RL/NuFingers_bad{}dim_{}_log'.format(dim,seed)
#         perturb = '--perturb=delay'
#         algdim = '--algdim={}'.format(dim)
        
#         finalargs = defaultargs + [demofile, perturb, algdim, '--seed={}'.format(seed)]
        
#         run.main(finalargs)