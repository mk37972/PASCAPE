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

a = gym.make('CheolFingersLiquid-v1', n_actions=3, eval_env=False)
render = 0
ep = 0
while ep < num_ep:
    print("Episode: {}".format(ep+1))
    a.reset()
    if render: a.render()
    obsDataNew, reward, done, info = a.step([0,0,0])
    collision = False
    obj_location = a.est_obj_pose.copy()[:2]
    # obj_location[1,0] += 0.005
    cen_location = a.des_p[1:3]
    
    episodeAcs = []
    episodeObs = []
    episodeInfo = []
    for i in range(lengh_ep):
        if np.linalg.norm(np.array([(a.goal[:2].reshape(-1,1)-obj_location)[0,0],(a.goal[:2].reshape(-1,1)-obj_location)[1,0]]))<0.005: # hold it still
            action = [-0,0,0]
            
            obsDataNew, reward, done, info = a.step(action)
            
            episodeAcs.append(action)
            episodeInfo.append(info)
            episodeObs.append(obsDataNew)
            if render: print("STAGE4", a.prev_oforce, a.obj_acc)
        elif np.linalg.norm(obj_location - cen_location) > 0.01: # move to the object
            obj_location = a.est_obj_pose.copy()[:2]
            # obj_location[1,0] += 0.005
            cen_location = a.des_p[1:3]
            
            move_to_obj = 10*np.array([(obj_location-cen_location)[0,0],(obj_location-cen_location)[1,0]])
            decrease_l = -0.5
            
            action = [decrease_l,move_to_obj[0],move_to_obj[1]]
            obsDataNew, reward, done, info = a.step(action)
            
            episodeAcs.append(action)
            episodeInfo.append(info)
            episodeObs.append(obsDataNew)
            
            start_grasp = i
            if render: print("STAGE1", a.prev_oforce, a.obj_acc)
        elif a.prev_oforce < a.min_grip/2. and np.linalg.norm(obj_location - cen_location) < 0.01: # grasp the object
            obj_location = a.est_obj_pose.copy()[:2]
            # obj_location[1,0] += 0.005
            cen_location = a.des_p[1:3]
            
            # print(cen_location, obj_location, a.p[1:3], a.goal[:2].reshape(-1,1))
            
            move_to_obj = 6*np.array([(obj_location-cen_location)[0,0],(obj_location-cen_location)[1,0]])
            move_to_goal = 6*np.array([(a.goal[:2].reshape(-1,1)-cen_location)[0,0],(a.goal[:2].reshape(-1,1)-cen_location)[1,0]])
            decrease_l = -0.5
            
            action = [decrease_l,move_to_obj[0],move_to_obj[1]]
            obsDataNew, reward, done, info = a.step(action)
            
            episodeAcs.append(action)
            episodeInfo.append(info)
            episodeObs.append(obsDataNew)
            if render: print("STAGE2", a.prev_oforce, a.obj_acc)
        elif np.linalg.norm(np.array([(a.goal[:2].reshape(-1,1)-obj_location)[0,0],(a.goal[:2].reshape(-1,1)-obj_location)[1,0]]))>0.001: # manipulate the object
            cen_location = a.des_p[1:3]
            
            move_to_goal = 10*np.array([(a.goal[:2].reshape(-1,1)-obj_location)[0,0],(a.goal[:2].reshape(-1,1)-obj_location)[1,0]])
            decrease_l = -1.
            
            action = [decrease_l,move_to_goal[0],move_to_goal[1]]
            obsDataNew, reward, done, info = a.step(action)
            
            episodeAcs.append(action)
            episodeInfo.append(info)
            episodeObs.append(obsDataNew)
            if render: print("STAGE3", a.prev_oforce, a.obj_acc)
        if render: a.render()
        real_object_pos = a.sim.data.get_body_xpos('object2')
        # print(reward)
        # print(a.max_vel_L,a.max_vel_R,a.max_ext_torques_L,a.max_ext_torques_R,)

    if info['is_success'] == 1.0:
        actions.append(episodeAcs)
        observations.append(episodeObs)
        infos.append(episodeInfo)
        print('Recorded..')
        ep += 1
    else:
        print('Failed!')
    

if render != 1: 
    for i in [3,5]:
        fileName = "ToolManipulationDRDemo"
        if i == 3:
            fileName += "_3D"
        if i == 5:
            for j in range(num_ep):
                for k in range(lengh_ep):
                    observations[j][k]['observation'] = np.concatenate([observations[j][k]['observation'], [0.8, 0.8]])
                    actions[j][k] = np.concatenate([actions[j][k], [0.0, 0.0]])
            fileName += "_5D"
        fileName += ".npz"
        np.savez_compressed(fileName, acs=actions, obs=observations, info=infos) # save the file