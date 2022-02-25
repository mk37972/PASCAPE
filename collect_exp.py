#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:36:15 2020

@author: mincheol
"""
from baselines import run

# defaultargs = ['--alg=her','--env=FetchPickAndPlaceFragile-v3', '--num_timesteps=0', '--play']

# if __name__ == '__main__':
#     for dim in [6]:
#         for seed in [10,750,1000]:
#             for pert in ['delay']:
#                 loadpath = '--load_path=./models/animate/harsh_65/PassiveCtrl_randomly_heavy/fpp_demo25bad{}dim_{}'.format(dim,seed)
#                 filename = '--filename=./models/animate/harsh_65/PassiveCtrl_randomly_heavy/force_dist_data/Force_Distance_data_random_{}_{}{}'.format(dim,pert,seed)
#                 perturb = '--perturb={}'.format(pert)
#                 algdim = '--algdim={}'.format(dim)
                
#                 finalargs = defaultargs + [loadpath, perturb, algdim]
#                 run.main(finalargs)

# defaultargs = ['--alg=her','--env=FetchPickAndPlaceFragile-v2', '--num_timesteps=0', '--play']

# if __name__ == '__main__':
#     for dim in [3]:
#         for seed in [10,500,1000]:
#             for pert in ['delay']:
#                 loadpath = '--load_path=./models/chip/harsh_85/For_IM/fpp_demo25bad{}dim_{}'.format(dim,seed)
#                 filename = '--filename=./models/chip/harsh_85/For_IM/force_dist_data_before/Force_Distance_data_random_{}_{}{}'.format(dim,pert,seed)
#                 perturb = '--perturb={}'.format(pert)
#                 algdim = '--algdim={}'.format(dim)
                
#                 finalargs = defaultargs + [loadpath, perturb, algdim]
#                 run.main(finalargs)

# defaultargs = ['--alg=her','--env=FetchPickAndPlaceFragile-v1', '--num_timesteps=0', '--play']

# if __name__ == '__main__':
#     for dim in [4]:
#         for seed in [10,500,1000]:
#             for pert in ['delay']:
#                 loadpath = '--load_path=./models/block/test/fpp_demo25bad{}dim_{}'.format(dim,seed)
#                 filename = '--filename=./models/block/test/force_dist_data/Force_Distance_data_random_{}_{}{}'.format(dim,pert,seed)
#                 perturb = '--perturb={}'.format(pert)
#                 algdim = '--algdim={}'.format(dim)
                
#                 finalargs = defaultargs + [loadpath, perturb, algdim]
#                 run.main(finalargs)

# defaultargs = ['--alg=her','--env=FetchPickAndPlaceFragile-v5', '--num_timesteps=0', '--play']

# if __name__ == '__main__':
#     for dim in [2]:
#         for seed in [10,500,1000]:
#             for pert in ['delay']:
#                 loadpath = '--load_path=./models/block/harsh_65/For_IM/RL/fpp_demo25bad{}dim_{}'.format(dim,seed)
#                 filename = '--filename=./models/block/harsh_65/For_IM/RL/force_dist_data/Force_Distance_data_random_{}_{}{}'.format(dim,pert,seed)
#                 perturb = '--perturb={}'.format(pert)
#                 algdim = '--algdim={}'.format(dim)
                
#                 finalargs = defaultargs + [loadpath, perturb, algdim]
#                 run.main(finalargs)

# defaultargs = ['--alg=her','--env=FetchPickAndPlaceFragile-v6', '--num_timesteps=0', '--play']

# if __name__ == '__main__':
#     for dim in [2]:
#         for seed in [10,500,1000]:
#             for pert in ['delay']:
#                 loadpath = '--load_path=./models/chip/harsh_85/For_IM/RL/fpp_demo25bad{}dim_{}'.format(dim,seed)
#                 filename = '--filename=./models/chip/harsh_85/For_IM/RL/force_dist_data_before/Force_Distance_data_random_{}_{}{}'.format(dim,pert,seed)
#                 perturb = '--perturb={}'.format(pert)
#                 algdim = '--algdim={}'.format(dim)
                
#                 finalargs = defaultargs + [loadpath, perturb, algdim]
#                 run.main(finalargs)

# defaultargs = ['--alg=her','--env=NuFingersRotate-v1', '--num_timesteps=0', '--play']
# if __name__ == '__main__':
#     for dim in [2]:
#         for seed in [10,500,1000]:
#             for pert in ['delay']:
#                 loadpath = '--load_path=./models/NuFingers/harsh_65/For_IM/Sim_NuFingers_bad{}dim_{}'.format(dim,seed)
#                 filename = '--filename=./models/NuFingers/harsh_65/IR/force_dist_data/Force_Distance_data_random_{}_{}{}'.format(dim,pert,seed)
#                 perturb = '--perturb={}'.format(pert)
#                 algdim = '--algdim={}'.format(dim)
            
#                 finalargs = defaultargs + [loadpath, perturb, algdim]
#                 run.main(finalargs)
            
# defaultargs = ['--alg=her','--env=NuFingersRotate-v2', '--num_timesteps=0', '--play']
# if __name__ == '__main__':
#     for dim in [2]:
#         for seed in [10,500,1000]:
#             for pert in ['delay']:
#                 loadpath = '--load_path=./models/NuFingers/harsh_65/For_IM/RL/Sim_NuFingers_bad{}dim_{}'.format(dim,seed)
#                 filename = '--filename=./models/NuFingers/harsh_65/For_IM/RL/force_dist_data/Force_Distance_data_random_{}_{}{}'.format(dim,pert,seed)
#                 perturb = '--perturb={}'.format(pert)
#                 algdim = '--algdim={}'.format(dim)
            
#                 finalargs = defaultargs + [loadpath, perturb, algdim]
#                 run.main(finalargs)

        
defaultargs = ['--alg=her','--env=CheolFingersManipulate-v1', '--num_timesteps=0', '--play']
if __name__ == '__main__':
    for dim in [5]:
        for seed in [1000]:
            for pert in ['none']:
                loadpath = '--load_path=./models/Dark/Manipulate/Delayed_env/PASCAPE_10/Sim_NuFingers_bad{}dim_{}'.format(dim,seed)
                perturb = '--perturb={}'.format(pert)
                algdim = '--algdim={}'.format(dim)
                eval_env = '--eval_env=True'
                
                finalargs = defaultargs + [loadpath, perturb, algdim, eval_env]
                
                run.main(finalargs)
                
# defaultargs = ['--alg=her','--env=CheolFingersSearch-v1', '--num_timesteps=0', '--play']
# if __name__ == '__main__':
#     for dim in [3]:
#         for seed in [10]:
#             for pert in ['none']:
#                 loadpath = '--load_path=./models/Dark/Search/SCAPE_heavy_penalty_low_stiffness/Sim_NuFingers_bad{}dim_{}'.format(dim,seed)
#                 perturb = '--perturb={}'.format(pert)
#                 algdim = '--algdim={}'.format(dim)
#                 eval_env = '--eval_env=False'
                
#                 finalargs = defaultargs + [loadpath, perturb, algdim, eval_env]
                
#                 run.main(finalargs)