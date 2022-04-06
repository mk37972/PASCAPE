# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 18:32:55 2021

@author: mk37972
"""

import numpy as np
import gym

actions = []
observations = []
infos = []
num_ep = 100
lengh_ep = 50
action = [0]

a = gym.make('CheolFingersSearch-v1', n_actions=1, eval_env=False)
render = 0
ep = 0
while ep < num_ep:
    print("Episode: {}".format(ep+1))
    a.reset()
    if render: a.render()
    
    collision = False
    
    episodeAcs = []
    episodeObs = []
    episodeInfo = []
    
    for i in range(lengh_ep):
        action = [1]
        obsDataNew, reward, done, info = a.step(action)
        # print(reward)
        
        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)
        
        if render: a.render()
    
    if info['is_success'] == 1.0:
        actions.append(episodeAcs)
        observations.append(episodeObs)
        infos.append(episodeInfo)
        print('Recorded..')
        ep += 1
    else:
        print('Failed!')
    

if render != 1: 
    for i in [1,3]:
        fileName = "DarkSearchDemo"
        if i == 1:
            fileName += "_1D"
        if i == 3:
            for j in range(num_ep):
                for k in range(lengh_ep):
                    observations[j][k]['observation'] = np.concatenate([observations[j][k]['observation'], [0.8, 0.8]])
                    actions[j][k] = np.concatenate([actions[j][k], [0.0, 0.0]])
            fileName += "_3D"
        fileName += ".npz"
        np.savez_compressed(fileName, acs=actions, obs=observations, info=infos) # save the file