from collections import deque

import numpy as np
import pickle

from baselines.her.util import convert_episode_to_batch_major, store_args

from nifpga import Session
import time
import click

class RolloutWorker:

    @store_args
    def __init__(self, policy, dims, logger, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, monitor=False, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            venv: vectorized gym environments.
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            rollout_batch_size (int): the number of parallel rollouts that should be used
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            use_target_net (boolean): whether or not to use the target net for rollouts
            compute_Q (boolean): whether or not to compute the Q values alongside the actions
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a completely random action
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
        """

        self.info_keys = [key.replace('info_', '') for key in self.dims.keys() if key.startswith('info_')]

        self.success_history = deque(maxlen=history_len)
        self.success_history2 = deque(maxlen=history_len)
        self.success_history3 = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)
        self.F_history = deque(maxlen=history_len)
        self.K_history = deque(maxlen=history_len)
        
        self.R_j = np.matrix([[0.01575,0],
                          [-0.01575, 0.01575]])
        self.R_j_inv = np.linalg.inv(self.R_j)
        self.R_j_L = np.matrix([[0.01575,0],
                          [0.01575, 0.01575]])
        self.R_j_inv_L = np.linalg.inv(self.R_j_L)
        self.R_e = np.matrix([[0.0034597,0],
                          [0, 0.0034597]])
        self.L1 = 0.1
        self.L2 = 0.075
        
        self.Ksc = 700
        
        self.Rm = 0.0285
                        
        self.Pc_R = np.array([-0.0635, 0.12])
        self.Pc_L = np.array([0.0635, 0.12])
        self.max_kj_R = np.transpose(self.R_j) * np.matrix([[2*self.Ksc, 0],[0, 2*self.Ksc]]) * self.R_j
        self.max_kj_L = np.transpose(self.R_j_L) * np.matrix([[2*self.Ksc, 0],[0, 2*self.Ksc]]) * self.R_j_L
        
        self.stiffness = 1.0
        self.stiffness_lim = 1.0
        
        self.l_limit = 0.08
        self.th_limit = np.pi/3.0
        
        self.l_step_limit = 0.02
        self.th_step_limit = np.pi/60.0
        
        self.vel_limit = 0.15
        
        self.object_fragility = 2
        
        self.offset = np.array([-290.8079833984375,-346.9042053222656,579.5279541015625,-16115.4892578125])
        
        self.actions_list = []
        self.obs_list =[]
        self.infos_list = []
        self.forces_list = []

        self.n_episodes = 0
        self.reset_all_rollouts()
        self.clear_history()

    def reset_all_rollouts(self):
        
        with Session(bitfile="SCPC-lv-noFIFO_FPGATarget_FPGAmainepos_1XvgQEcJVeE.lvbitx", resource="rio://10.157.23.150/RIO0") as session:
            act_Rj1=session.registers['Mod3/AO0']
            enc_Rj1=session.registers['Rj1']
            act_Rj2=session.registers['Mod3/AO1']
            enc_Rj2=session.registers['Rj2']
            
            act_Lj1=session.registers['Mod3/AO3']
            enc_Lj1=session.registers['Lj1']
            act_Lj2=session.registers['Mod3/AO4']
            enc_Lj2=session.registers['Lj2']
            
            sen_f=session.registers['fsensor']
            sen_e=session.registers['fencoder']
            
            # resetting the offset for encoder malfunction
            for i in range(201):
                act_Rj1.write(-3 + 3*i/200.0)
                act_Rj2.write(-3 + 3*i/200.0)
                act_Lj1.write(3 - 3*i/200.0)
                act_Lj2.write(-3 + 3*i/200.0)
            
            time.sleep(0.5)
            Re1 = enc_Rj1.read()
            Re2 = enc_Rj2.read()
            Le1 = enc_Lj1.read()
            Le2 = enc_Lj2.read()
            self.offset = np.array([Re1,Re2,Le1,Le2])
            
            
            # getting back to the starting pose
            des_l = 0.08
            des_th = 0.
            
            des_p = np.array([[np.min([np.max([des_l, -self.l_limit]), self.l_limit])], [np.min([np.max([des_th, -self.th_limit]), self.th_limit])]])#0.7854
            des_p_R = np.array([des_p[0]/2.0, des_p[1]])
            des_p_L = des_p_R
            
            r = np.array([[1.0], [1.0]])
            
            if self.compute_Q: new_goal = np.pi/16.0#(3*np.pi/32.0 + (np.random.random()*np.pi/16.0 - np.pi/32.0)) * np.sign(np.random.random() - 0.5)
            else: new_goal = np.random.random()*np.pi/4.0 - np.pi/8.0
            
            for i in range(25):
                
                Re1 = enc_Rj1.read()
                Re2 = enc_Rj2.read()
                Le1 = enc_Lj1.read()
                Le2 = enc_Lj2.read()
                
                f_sensor = 5.1203 * sen_f.read() - 5.2506
                e_sensor = (((sen_e.read()) - (f_sensor / 100.0 * 0.15)) -2.9440)/0.0148
                
                Rj = self.R_j_inv * self.R_e * np.array([[Re1-self.offset[0]],[-Re2+self.offset[1]]]) * np.pi/180.0
                Lj = self.R_j_inv_L * self.R_e * np.array([[Le1-self.offset[2]],[Le2-self.offset[3]]]) * np.pi/180.0
                
                xR = self.L1 * np.cos(Rj[0,0] + np.pi/2.0) + self.L2 * np.cos(Rj[0,0]-Rj[1,0] + np.pi/2.0)
                yR = self.L1 * np.sin(Rj[0,0] + np.pi/2.0) + self.L2 * np.sin(Rj[0,0]-Rj[1,0] + np.pi/2.0)
                xL = self.L1 * np.cos(Lj[0,0] + np.pi/2.0) + self.L2 * np.cos(Lj[0,0]+Lj[1,0] + np.pi/2.0)
                yL = self.L1 * np.sin(Lj[0,0] + np.pi/2.0) + self.L2 * np.sin(Lj[0,0]+Lj[1,0] + np.pi/2.0)
                P_R = np.array([xR, yR])
                P_L = np.array([xL, yL])
                Prel_R = self.Pc_R - P_R
                Prel_L = self.Pc_L - P_L
                l_R = np.sqrt(Prel_R[0]*Prel_R[0] + Prel_R[1]*Prel_R[1])
                l_L = np.sqrt(Prel_L[0]*Prel_L[0] + Prel_L[1]*Prel_L[1])
                p_R = np.array([[l_R],[np.arctan2(-Prel_R[1],-Prel_R[0])]])
                p_L = np.array([[l_L],[np.arctan2(Prel_L[1],Prel_L[0])]])
                
                # change_l_R = np.clip(des_l/2.0 - p_R[0,0], -self.l_step_limit, self.l_step_limit)
                # change_th_R = np.clip(des_th - p_R[1,0], -8.0*self.th_step_limit, self.th_step_limit)
                # change_l_L = np.clip(des_l/2.0 - p_L[0,0], -self.l_step_limit, self.l_step_limit)
                # change_th_L = np.clip(des_th - p_L[1,0], -self.th_step_limit, 8.0*self.th_step_limit)
                # des_p_R = np.array([[np.min([np.max([p_R[0,0] + change_l_R, -self.l_limit/2.0]), self.l_limit/2.0])], [np.min([np.max([p_R[1,0] + change_th_R, -self.th_limit]), self.th_limit])]])
                # des_p_L = np.array([[np.min([np.max([p_L[0,0] + change_l_L, -self.l_limit/2.0]), self.l_limit/2.0])], [np.min([np.max([p_L[1,0] + change_th_L, -self.th_limit]), self.th_limit])]])
                des_p_R = np.array([[des_l/2.0], [des_th]])
                des_p_L = des_p_R
                
                Jp_R = np.matrix([[-Prel_R[0]/l_R, -Prel_R[1]/l_R],[Prel_R[1]/l_R/l_R, -Prel_R[0]/l_R/l_R]])
                Jp_L = np.matrix([[-Prel_L[0]/l_L, -Prel_L[1]/l_L],[Prel_L[1]/l_L/l_L, -Prel_L[0]/l_L/l_L]])
                Jp_inv_R = np.matrix([[Jp_R[1,1] / (Jp_R[0,0]*Jp_R[1,1] - Jp_R[0,1]*Jp_R[1,0]), -Jp_R[0,1] / (Jp_R[0,0]*Jp_R[1,1] - Jp_R[0,1]*Jp_R[1,0])], [-Jp_R[1,0] / (Jp_R[0,0]*Jp_R[1,1] - Jp_R[0,1]*Jp_R[1,0]), Jp_R[0,0] / (Jp_R[0,0]*Jp_R[1,1] - Jp_R[0,1]*Jp_R[1,0])]])
                Jp_inv_L = np.matrix([[Jp_L[1,1] / (Jp_L[0,0]*Jp_L[1,1] - Jp_L[0,1]*Jp_L[1,0]), -Jp_L[0,1] / (Jp_L[0,0]*Jp_L[1,1] - Jp_L[0,1]*Jp_L[1,0])], [-Jp_L[1,0] / (Jp_L[0,0]*Jp_L[1,1] - Jp_L[0,1]*Jp_L[1,0]), Jp_L[0,0] / (Jp_L[0,0]*Jp_L[1,1] - Jp_L[0,1]*Jp_L[1,0])]])
                J_R = np.matrix([[-yR, self.L2 * np.cos(Rj[0,0]-Rj[1,0])], 
                                 [xR, self.L2 * np.sin(Rj[0,0]-Rj[1,0])]])
                J_L = np.matrix([[-yL, -self.L2 * np.cos(Lj[0,0]+Lj[1,0])], 
                                 [xL, -self.L2 * np.sin(Lj[0,0]+Lj[1,0])]])
                J_inv_R = np.matrix([[J_R[1,1] / (J_R[0,0]*J_R[1,1] - J_R[0,1]*J_R[1,0]), -J_R[0,1] / (J_R[0,0]*J_R[1,1] - J_R[0,1]*J_R[1,0])], [-J_R[1,0] / (J_R[0,0]*J_R[1,1] - J_R[0,1]*J_R[1,0]), J_R[0,0] / (J_R[0,0]*J_R[1,1] - J_R[0,1]*J_R[1,0])]])
                J_inv_L = np.matrix([[J_L[1,1] / (J_L[0,0]*J_L[1,1] - J_L[0,1]*J_L[1,0]), -J_L[0,1] / (J_L[0,0]*J_L[1,1] - J_L[0,1]*J_L[1,0])], [-J_L[1,0] / (J_L[0,0]*J_L[1,1] - J_L[0,1]*J_L[1,0]), J_L[0,0] / (J_L[0,0]*J_L[1,1] - J_L[0,1]*J_L[1,0])]])
                
                max_k_R = np.transpose(J_inv_R) * self.max_kj_R * J_inv_R
                max_k_L = np.transpose(J_inv_L) * self.max_kj_L * J_inv_L
                max_kp_R = np.transpose(Jp_inv_R) * max_k_R * Jp_inv_R
                max_kp_L = np.transpose(Jp_inv_L) * max_k_L * Jp_inv_L
                max_kp_R[0,1] = 0.0
                max_kp_R[1,0] = 0.0
                max_kp_L[0,1] = 0.0
                max_kp_L[1,0] = 0.0
                des_Fp_R = max_kp_R * (r * (des_p_R - p_R))
                des_Fp_L = max_kp_L * (r * (des_p_L - p_L))
                
                des_F_R = np.transpose(Jp_R) * des_Fp_R
                des_F_L = np.transpose(Jp_L) * des_Fp_L
                des_tau_R = np.transpose(J_R) * des_F_R if Rj[1,0] < 0 else np.array([[0.0],[-0.1]])
                des_tau_L = np.transpose(J_L) * des_F_L if Lj[1,0] < 0 else np.array([[0.0],[-0.1]])
                des_tau_R = np.clip(des_tau_R, -0.1, 0.1)
                des_tau_L = np.clip(des_tau_L, -0.1, 0.1)
                des_mR = (np.transpose(self.R_j_inv)*des_tau_R / (2*self.Ksc) + self.R_j * Rj) / self.Rm 
                des_mL = (np.transpose(self.R_j_inv_L)*des_tau_L / (2*self.Ksc) + self.R_j_L * Lj) / self.Rm
                
                # Position control mode
                act_Rj1.write(np.min([np.max([des_mR[0,0] * 180.0 / np.pi * 0.117258, -10.0]),10.0]))
                act_Rj2.write(np.min([np.max([des_mR[1,0] * 180.0 / np.pi * 0.117541, -10.0]),10.0]))
                act_Lj1.write(np.min([np.max([des_mL[0,0] * 180.0 / np.pi * 0.117729, -10.0]),10.0]))
                act_Lj2.write(np.min([np.max([des_mL[1,0] * 180.0 / np.pi * 0.117679, -10.0]),10.0]))
                
                time.sleep(0.2)
                
            r = np.array([[2.0], [2.0]])
            
            for i in range(25):
                
                Re1 = enc_Rj1.read()
                Re2 = enc_Rj2.read()
                Le1 = enc_Lj1.read()
                Le2 = enc_Lj2.read()
                
                f_sensor = 5.1203 * sen_f.read() - 5.2506
                e_sensor = (((sen_e.read()) - (f_sensor / 100.0 * 0.15)) -2.9440)/0.0148
                
                Rj = self.R_j_inv * self.R_e * np.array([[Re1-self.offset[0]],[-Re2+self.offset[1]]]) * np.pi/180.0
                Lj = self.R_j_inv_L * self.R_e * np.array([[Le1-self.offset[2]],[Le2-self.offset[3]]]) * np.pi/180.0
                
                xR = self.L1 * np.cos(Rj[0,0] + np.pi/2.0) + self.L2 * np.cos(Rj[0,0]-Rj[1,0] + np.pi/2.0)
                yR = self.L1 * np.sin(Rj[0,0] + np.pi/2.0) + self.L2 * np.sin(Rj[0,0]-Rj[1,0] + np.pi/2.0)
                xL = self.L1 * np.cos(Lj[0,0] + np.pi/2.0) + self.L2 * np.cos(Lj[0,0]+Lj[1,0] + np.pi/2.0)
                yL = self.L1 * np.sin(Lj[0,0] + np.pi/2.0) + self.L2 * np.sin(Lj[0,0]+Lj[1,0] + np.pi/2.0)
                P_R = np.array([xR, yR])
                P_L = np.array([xL, yL])
                Prel_R = self.Pc_R - P_R
                Prel_L = self.Pc_L - P_L
                l_R = np.sqrt(Prel_R[0]*Prel_R[0] + Prel_R[1]*Prel_R[1])
                l_L = np.sqrt(Prel_L[0]*Prel_L[0] + Prel_L[1]*Prel_L[1])
                p_R = np.array([[l_R],[np.arctan2(-Prel_R[1],-Prel_R[0])]])
                p_L = np.array([[l_L],[np.arctan2(Prel_L[1],Prel_L[0])]])
                
                # change_l_R = np.clip(des_l/2.0 - p_R[0,0], -self.l_step_limit, self.l_step_limit)
                # change_th_R = np.clip(des_th - p_R[1,0], -8.0*self.th_step_limit, self.th_step_limit)
                # change_l_L = np.clip(des_l/2.0 - p_L[0,0], -self.l_step_limit, self.l_step_limit)
                # change_th_L = np.clip(des_th - p_L[1,0], -self.th_step_limit, 8.0*self.th_step_limit)
                # des_p_R = np.array([[np.min([np.max([p_R[0,0] + change_l_R, -self.l_limit/2.0]), self.l_limit/2.0])], [np.min([np.max([p_R[1,0] + change_th_R, -self.th_limit]), self.th_limit])]])
                # des_p_L = np.array([[np.min([np.max([p_L[0,0] + change_l_L, -self.l_limit/2.0]), self.l_limit/2.0])], [np.min([np.max([p_L[1,0] + change_th_L, -self.th_limit]), self.th_limit])]])
                des_p_R = np.array([[des_l/2.0], [des_th]])
                des_p_L = des_p_R
                
                Jp_R = np.matrix([[-Prel_R[0]/l_R, -Prel_R[1]/l_R],[Prel_R[1]/l_R/l_R, -Prel_R[0]/l_R/l_R]])
                Jp_L = np.matrix([[-Prel_L[0]/l_L, -Prel_L[1]/l_L],[Prel_L[1]/l_L/l_L, -Prel_L[0]/l_L/l_L]])
                Jp_inv_R = np.matrix([[Jp_R[1,1] / (Jp_R[0,0]*Jp_R[1,1] - Jp_R[0,1]*Jp_R[1,0]), -Jp_R[0,1] / (Jp_R[0,0]*Jp_R[1,1] - Jp_R[0,1]*Jp_R[1,0])], [-Jp_R[1,0] / (Jp_R[0,0]*Jp_R[1,1] - Jp_R[0,1]*Jp_R[1,0]), Jp_R[0,0] / (Jp_R[0,0]*Jp_R[1,1] - Jp_R[0,1]*Jp_R[1,0])]])
                Jp_inv_L = np.matrix([[Jp_L[1,1] / (Jp_L[0,0]*Jp_L[1,1] - Jp_L[0,1]*Jp_L[1,0]), -Jp_L[0,1] / (Jp_L[0,0]*Jp_L[1,1] - Jp_L[0,1]*Jp_L[1,0])], [-Jp_L[1,0] / (Jp_L[0,0]*Jp_L[1,1] - Jp_L[0,1]*Jp_L[1,0]), Jp_L[0,0] / (Jp_L[0,0]*Jp_L[1,1] - Jp_L[0,1]*Jp_L[1,0])]])
                J_R = np.matrix([[-yR, self.L2 * np.cos(Rj[0,0]-Rj[1,0])], 
                                 [xR, self.L2 * np.sin(Rj[0,0]-Rj[1,0])]])
                J_L = np.matrix([[-yL, -self.L2 * np.cos(Lj[0,0]+Lj[1,0])], 
                                 [xL, -self.L2 * np.sin(Lj[0,0]+Lj[1,0])]])
                J_inv_R = np.matrix([[J_R[1,1] / (J_R[0,0]*J_R[1,1] - J_R[0,1]*J_R[1,0]), -J_R[0,1] / (J_R[0,0]*J_R[1,1] - J_R[0,1]*J_R[1,0])], [-J_R[1,0] / (J_R[0,0]*J_R[1,1] - J_R[0,1]*J_R[1,0]), J_R[0,0] / (J_R[0,0]*J_R[1,1] - J_R[0,1]*J_R[1,0])]])
                J_inv_L = np.matrix([[J_L[1,1] / (J_L[0,0]*J_L[1,1] - J_L[0,1]*J_L[1,0]), -J_L[0,1] / (J_L[0,0]*J_L[1,1] - J_L[0,1]*J_L[1,0])], [-J_L[1,0] / (J_L[0,0]*J_L[1,1] - J_L[0,1]*J_L[1,0]), J_L[0,0] / (J_L[0,0]*J_L[1,1] - J_L[0,1]*J_L[1,0])]])
                
                max_k_R = np.transpose(J_inv_R) * self.max_kj_R * J_inv_R
                max_k_L = np.transpose(J_inv_L) * self.max_kj_L * J_inv_L
                max_kp_R = np.transpose(Jp_inv_R) * max_k_R * Jp_inv_R
                max_kp_L = np.transpose(Jp_inv_L) * max_k_L * Jp_inv_L
                max_kp_R[0,1] = 0.0
                max_kp_R[1,0] = 0.0
                max_kp_L[0,1] = 0.0
                max_kp_L[1,0] = 0.0
                des_Fp_R = max_kp_R * (r * (des_p_R - p_R))
                des_Fp_L = max_kp_L * (r * (des_p_L - p_L))
                
                des_F_R = np.transpose(Jp_R) * des_Fp_R
                des_F_L = np.transpose(Jp_L) * des_Fp_L
                des_tau_R = np.transpose(J_R) * des_F_R if Rj[1,0] < 0 else np.array([[0.0],[-0.1]])
                des_tau_L = np.transpose(J_L) * des_F_L if Lj[1,0] < 0 else np.array([[0.0],[-0.1]])
                des_tau_R = np.clip(des_tau_R, -0.1, 0.1)
                des_tau_L = np.clip(des_tau_L, -0.1, 0.1)
                des_mR = (np.transpose(self.R_j_inv)*des_tau_R / (2*self.Ksc) + self.R_j * Rj) / self.Rm 
                des_mL = (np.transpose(self.R_j_inv_L)*des_tau_L / (2*self.Ksc) + self.R_j_L * Lj) / self.Rm
                
                # Position control mode
                act_Rj1.write(np.min([np.max([des_mR[0,0] * 180.0 / np.pi * 0.117258, -10.0]),10.0]))
                act_Rj2.write(np.min([np.max([des_mR[1,0] * 180.0 / np.pi * 0.117541, -10.0]),10.0]))
                act_Lj1.write(np.min([np.max([des_mL[0,0] * 180.0 / np.pi * 0.117729, -10.0]),10.0]))
                act_Lj2.write(np.min([np.max([des_mL[1,0] * 180.0 / np.pi * 0.117679, -10.0]),10.0]))
                
                time.sleep(0.02)
            
            p_R[1,0] += -0.15
            p_L[1,0] += 0.15
            
            if self.dims['u'] > 2:
                observation = np.array([[p_R[0,0] * 10 - 1.0, p_L[0,0] * 10 - 1.0, p_R[1,0], p_L[1,0], 
                                              (e_sensor* np.pi / 180.0 - p_R[1,0]) , (e_sensor* np.pi / 180.0 - p_L[1,0]), 
                                              (new_goal - e_sensor * np.pi / 180.0),
                                              des_Fp_R[0,0] * 0.1, des_Fp_L[0,0] * 0.1, 
                                               0.0, 0.0, 0.0, 0.0,
                                              self.stiffness, self.stiffness_lim]])
            else:
                observation = np.array([[p_R[0,0] * 10 - 1.0, p_L[0,0] * 10 - 1.0, p_R[1,0], p_L[1,0], 
                                              (e_sensor* np.pi / 180.0 - p_R[1,0]) , (e_sensor* np.pi / 180.0 - p_L[1,0]), 
                                              (new_goal - e_sensor * np.pi / 180.0),
                                              des_Fp_R[0,0] * 0.1, des_Fp_L[0,0] * 0.1, 
                                               0.0, 0.0, 0.0, 0.0]])
        
            self.obs_dict = dict(observation = observation,
                                 achieved_goal = np.array([[e_sensor * np.pi/ 180.0, 0.0, 0.0, 0.0, 0.0, des_Fp_R[0,0] * 0.1, des_Fp_L[0,0] * 0.1]]),
                                 desired_goal = np.array([[new_goal, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                                 )
            
            self.initial_o = self.obs_dict['observation']
            self.initial_ag = self.obs_dict['achieved_goal']
            self.g = self.obs_dict['desired_goal']
            self.stiffness = 1.0
            self.stiffness_lim = 1.0

    def generate_rollouts(self):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        self.reset_all_rollouts()
        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # generate episodes
        obs, achieved_goals, acts, goals, successes, successes2 = [], [], [], [], [], []
        dones = []
        info_values = [np.empty((3*self.T-1, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        Qs = []
        Fs = []
        Ks = []
        
        with Session(bitfile="SCPC-lv-noFIFO_FPGATarget_FPGAmainepos_1XvgQEcJVeE.lvbitx", resource="rio://10.157.23.150/RIO0") as session:
            act_Rj1=session.registers['Mod3/AO0']
            enc_Rj1=session.registers['Rj1']
            act_Rj2=session.registers['Mod3/AO1']
            enc_Rj2=session.registers['Rj2']
            
            act_Lj1=session.registers['Mod3/AO3']
            enc_Lj1=session.registers['Lj1']
            act_Lj2=session.registers['Mod3/AO4']
            enc_Lj2=session.registers['Lj2']
            
            sen_f=session.registers['fsensor']
            sen_e=session.registers['fencoder']
            
            emergency = False
            
            Re1 = enc_Rj1.read()
            Re2 = enc_Rj2.read()
            Le1 = enc_Lj1.read()
            Le2 = enc_Lj2.read()
            
            f_sensor = 5.1203 * sen_f.read() - 5.2506
            e_sensor = (((sen_e.read()) - (f_sensor / 100.0 * 0.15)) -2.9440)/0.0148
            
            Rj = self.R_j_inv * self.R_e * np.array([[Re1-self.offset[0]],[-Re2+self.offset[1]]]) * np.pi/180.0
            Lj = self.R_j_inv_L * self.R_e * np.array([[Le1-self.offset[2]],[Le2-self.offset[3]]]) * np.pi/180.0
            
            Prev_Rj = Rj
            Prev_Lj = Lj
            
            des_l = 0.08
            des_th = 0.
            
            xR = self.L1 * np.cos(Rj[0,0] + np.pi/2.0) + self.L2 * np.cos(Rj[0,0]-Rj[1,0] + np.pi/2.0)
            yR = self.L1 * np.sin(Rj[0,0] + np.pi/2.0) + self.L2 * np.sin(Rj[0,0]-Rj[1,0] + np.pi/2.0)
            xL = self.L1 * np.cos(Lj[0,0] + np.pi/2.0) + self.L2 * np.cos(Lj[0,0]+Lj[1,0] + np.pi/2.0)
            yL = self.L1 * np.sin(Lj[0,0] + np.pi/2.0) + self.L2 * np.sin(Lj[0,0]+Lj[1,0] + np.pi/2.0)
            
            P_R = np.array([xR, yR])
            P_L = np.array([xL, yL])
            Prel_R = self.Pc_R - P_R
            Prel_L = self.Pc_L - P_L
            l_R = np.sqrt(Prel_R[0]*Prel_R[0] + Prel_R[1]*Prel_R[1])
            l_L = np.sqrt(Prel_L[0]*Prel_L[0] + Prel_L[1]*Prel_L[1])
            p_R = np.array([[l_R],[np.arctan2(-Prel_R[1],-Prel_R[0])]])
            p_L = np.array([[l_L],[np.arctan2(Prel_L[1],Prel_L[0])]])
            
            if self.compute_Q == True: print("Evaluating... \n Goal: {}".format(int(self.g[0][0] * 180.0 / np.pi)))
            max_force = 0
            
            episodeAct = []
            episodeObs = []
            episodeInfo = []
            episodeForce = []
            
            for t in range(self.T*3):
                policy_output = self.policy.get_actions(
                    o, ag, self.g,
                    compute_Q=self.compute_Q,
                    noise_eps=self.noise_eps if not self.exploit else 0.,
                    random_eps=self.random_eps if not self.exploit else 0.,
                    use_target_net=self.use_target_net)
    
                if self.compute_Q:
                    u, Q = policy_output
                    Qs.append(Q)
                    Fs.append(f_sensor)
                    Ks.append(self.stiffness)
                else:
                    u = policy_output
                if u.ndim == 1:
                    # The non-batched case should still have a reasonable shape.
                    u = u.reshape(1, -1)
    
                o_new = np.empty((self.rollout_batch_size, self.dims['o']))
                ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
                success = np.zeros(self.rollout_batch_size)
                success2 = np.zeros(self.rollout_batch_size)
                
                # compute new states and observations
                if self.dims['u'] > 2:
                    self.stiffness_lim = np.clip(self.stiffness_lim + 0.2 * u[0][3], 0.1, 1.0)
                    self.stiffness = np.clip(self.stiffness + 0.2 * u[0][2], 0, self.stiffness_lim)
                    # print(self.stiffness)
                change_l = np.clip(u[0][0]/50.0, -self.l_step_limit, self.l_step_limit)
                change_th = np.clip(u[0][1]/20.0, -self.th_step_limit, self.th_step_limit)
                vel_R = Rj - Prev_Rj
                vel_L = Lj - Prev_Lj
                
                if emergency == False:
                    if vel_R[0,0] > self.vel_limit or vel_R[0,0] < -self.vel_limit or vel_L[0,0] > self.vel_limit or vel_L[0,0] < -self.vel_limit:
                        emergency = True
                        self.stiffness = 0.0
                        r = np.array([[self.stiffness], [0.0]])
                        print("***************Robot going insane! Safety on!***************")
                        print(vel_R, vel_L)
                    else:
                        r = np.array([[self.stiffness], [1.0]])
                else:
                      self.stiffness = 0.0
                      r = np.array([[self.stiffness], [0.0]])
                des_p_R = np.array([[np.min([np.max([p_R[0,0] + u[0][0]/14.0, -self.l_limit/2.0]), self.l_limit/2.0])], [np.min([np.max([p_R[1,0] + u[0][1]/2.0, -self.th_limit]), self.th_limit])]])
                des_p_L = np.array([[np.min([np.max([p_L[0,0] + u[0][0]/14.0, -self.l_limit/2.0]), self.l_limit/2.0])], [np.min([np.max([p_L[1,0] + u[0][1]/2.0, -self.th_limit]), self.th_limit])]])
                
                des_l = des_l + change_l
                des_th = des_th + change_th
                
                des_p_R = np.array([[np.min([np.max([des_l/2.0, -self.l_limit]), self.l_limit])], [np.min([np.max([des_th, -self.th_limit]), self.th_limit])]])
                
                des_p_L = des_p_R
                
                Jp_R = np.matrix([[-Prel_R[0]/l_R, -Prel_R[1]/l_R],[Prel_R[1]/l_R/l_R, -Prel_R[0]/l_R/l_R]])
                Jp_L = np.matrix([[-Prel_L[0]/l_L, -Prel_L[1]/l_L],[Prel_L[1]/l_L/l_L, -Prel_L[0]/l_L/l_L]])
                Jp_inv_R = np.matrix([[Jp_R[1,1] / (Jp_R[0,0]*Jp_R[1,1] - Jp_R[0,1]*Jp_R[1,0]), -Jp_R[0,1] / (Jp_R[0,0]*Jp_R[1,1] - Jp_R[0,1]*Jp_R[1,0])], [-Jp_R[1,0] / (Jp_R[0,0]*Jp_R[1,1] - Jp_R[0,1]*Jp_R[1,0]), Jp_R[0,0] / (Jp_R[0,0]*Jp_R[1,1] - Jp_R[0,1]*Jp_R[1,0])]])
                Jp_inv_L = np.matrix([[Jp_L[1,1] / (Jp_L[0,0]*Jp_L[1,1] - Jp_L[0,1]*Jp_L[1,0]), -Jp_L[0,1] / (Jp_L[0,0]*Jp_L[1,1] - Jp_L[0,1]*Jp_L[1,0])], [-Jp_L[1,0] / (Jp_L[0,0]*Jp_L[1,1] - Jp_L[0,1]*Jp_L[1,0]), Jp_L[0,0] / (Jp_L[0,0]*Jp_L[1,1] - Jp_L[0,1]*Jp_L[1,0])]])
                J_R = np.matrix([[-yR, self.L2 * np.cos(Rj[0,0]-Rj[1,0])], 
                                 [xR, self.L2 * np.sin(Rj[0,0]-Rj[1,0])]])
                J_L = np.matrix([[-yL, -self.L2 * np.cos(Lj[0,0]+Lj[1,0])], 
                                 [xL, -self.L2 * np.sin(Lj[0,0]+Lj[1,0])]])
                J_inv_R = np.matrix([[J_R[1,1] / (J_R[0,0]*J_R[1,1] - J_R[0,1]*J_R[1,0]), -J_R[0,1] / (J_R[0,0]*J_R[1,1] - J_R[0,1]*J_R[1,0])], [-J_R[1,0] / (J_R[0,0]*J_R[1,1] - J_R[0,1]*J_R[1,0]), J_R[0,0] / (J_R[0,0]*J_R[1,1] - J_R[0,1]*J_R[1,0])]])
                J_inv_L = np.matrix([[J_L[1,1] / (J_L[0,0]*J_L[1,1] - J_L[0,1]*J_L[1,0]), -J_L[0,1] / (J_L[0,0]*J_L[1,1] - J_L[0,1]*J_L[1,0])], [-J_L[1,0] / (J_L[0,0]*J_L[1,1] - J_L[0,1]*J_L[1,0]), J_L[0,0] / (J_L[0,0]*J_L[1,1] - J_L[0,1]*J_L[1,0])]])
                
                max_kj_R = np.transpose(self.R_j) * np.matrix([[2*self.Ksc, 0],[0, 2*self.Ksc]]) * self.R_j
                max_kj_L = np.transpose(self.R_j_L) * np.matrix([[2*self.Ksc, 0],[0, 2*self.Ksc]]) * self.R_j_L
                max_k_R = np.transpose(J_inv_R) * max_kj_R * J_inv_R
                max_k_L = np.transpose(J_inv_L) * max_kj_L * J_inv_L
                max_kp_R = np.transpose(Jp_inv_R) * max_k_R * Jp_inv_R
                max_kp_L = np.transpose(Jp_inv_L) * max_k_L * Jp_inv_L
                max_kp_R[0,1] = 0.0
                max_kp_R[1,0] = 0.0
                max_kp_L[0,1] = 0.0
                max_kp_L[1,0] = 0.0
                des_Fp_R = max_kp_R * (r * (des_p_R - p_R))
                des_Fp_L = max_kp_L * (r * (des_p_L - p_L))
                des_F_R = np.transpose(Jp_R) * des_Fp_R
                des_F_L = np.transpose(Jp_L) * des_Fp_L
                des_tau_R = np.transpose(J_R) * des_F_R
                des_tau_L = np.transpose(J_L) * des_F_L
                # if Rj[1,0] > -0.1: 
                #     if Rj[0,0] < -1.0: des_tau_R = np.clip(des_tau_R,np.array([[0.05],[-1.0]]),np.array([[1.0],[-0.05]])) 
                #     elif Rj[0,0] > 0.1: des_tau_R = np.clip(des_tau_R,np.array([[-1.0],[-1.0]]),np.array([[-0.05],[-0.05]])) 
                #     else: des_tau_R = np.clip(des_tau_R,np.array([[des_tau_R[0,0]],[-1.0]]),np.array([[des_tau_R[0,0]],[-0.05]]))
                # elif Rj[1,0] < -2.0: 
                #     if Rj[0,0] < -1.0: des_tau_R = np.clip(des_tau_R,np.array([[0.05],[0.05]]),np.array([[1.0],[1.0]])) 
                #     elif Rj[0,0] > 0.1: des_tau_R = np.clip(des_tau_R,np.array([[-1.0],[0.05]]),np.array([[-0.05],[1.0]])) 
                #     else: des_tau_R = np.clip(des_tau_R,np.array([[des_tau_R[0,0]],[0.05]]),np.array([[des_tau_R[0,0]],[1.0]]))
                # elif Rj[0,0] < -1.0: des_tau_R = np.clip(des_tau_R,np.array([[0.05],[des_tau_R[1,0]]]),np.array([[1.0],[des_tau_R[1,0]]])) 
                # elif Rj[0,0] > 0.1: des_tau_R = np.clip(des_tau_R,np.array([[-1.0],[des_tau_R[1,0]]]),np.array([[-0.05],[des_tau_R[1,0]]])) 
                   
                # if Lj[1,0] > -0.1: 
                #     if Lj[0,0] < 0.1: des_tau_L = np.clip(des_tau_L,np.array([[0.05],[-1.0]]),np.array([[1.0],[-0.05]])) 
                #     elif Lj[0,0] > 1.0: des_tau_L = np.clip(des_tau_L,np.array([[-1.0],[-1.0]]),np.array([[-0.05],[-0.05]])) 
                #     else: des_tau_L = np.clip(des_tau_L,np.array([[des_tau_L[0,0]],[-1.0]]),np.array([[des_tau_L[0,0]],[-0.05]]))
                # elif Lj[1,0] < -2.0: 
                #     if Lj[0,0] < 0.1: des_tau_L = np.clip(des_tau_L,np.array([[0.05],[0.05]]),np.array([[1.0],[1.0]])) 
                #     elif Lj[0,0] > 1.0: des_tau_L = np.clip(des_tau_L,np.array([[-1.0],[0.05]]),np.array([[-0.05],[1.0]])) 
                #     else: des_tau_L = np.clip(des_tau_L,np.array([[des_tau_L[0,0]],[0.05]]),np.array([[des_tau_L[0,0]],[1.0]]))
                # elif Lj[0,0] < 0.1: des_tau_L = np.clip(des_tau_L,np.array([[0.05],[des_tau_L[1,0]]]),np.array([[1.0],[des_tau_L[1,0]]])) 
                # elif Lj[0,0] > 1.0: des_tau_L = np.clip(des_tau_L,np.array([[-1.0],[des_tau_L[1,0]]]),np.array([[-0.05],[des_tau_L[1,0]]])) 
                    
            
                des_mR = (np.transpose(self.R_j_inv)*des_tau_R / (2*self.Ksc) + self.R_j * Rj) / self.Rm 
                des_mL = (np.transpose(self.R_j_inv_L)*des_tau_L / (2*self.Ksc) + self.R_j_L * Lj) / self.Rm
                
                act_Rj1.write(np.min([np.max([des_mR[0,0] * 180.0 / np.pi * 0.117258, -10.0]),10.0]))
                act_Rj2.write(np.min([np.max([des_mR[1,0] * 180.0 / np.pi * 0.117541, -10.0]),10.0]))
                act_Lj1.write(np.min([np.max([des_mL[0,0] * 180.0 / np.pi * 0.117729, -10.0]),10.0]))
                act_Lj2.write(np.min([np.max([des_mL[1,0] * 180.0 / np.pi * 0.117679, -10.0]),10.0]))
                
                time.sleep(0.2)
                
                Re1 = enc_Rj1.read()
                Re2 = enc_Rj2.read()
                Le1 = enc_Lj1.read()
                Le2 = enc_Lj2.read()
                f_sensor = 5.1203 * sen_f.read() - 5.2506
                e_sensor = (((sen_e.read()) - (f_sensor / 100.0 * 0.15)) -2.9440)/0.0148
                
                Prev_Rj = Rj
                Prev_Lj = Lj
                
                Rj = self.R_j_inv * self.R_e * np.array([[Re1-self.offset[0]],[-Re2+self.offset[1]]]) * np.pi/180.0
                Lj = self.R_j_inv_L * self.R_e * np.array([[Le1-self.offset[2]],[Le2-self.offset[3]]]) * np.pi/180.0
                
                xR = self.L1 * np.cos(Rj[0,0] + np.pi/2.0) + self.L2 * np.cos(Rj[0,0]-Rj[1,0] + np.pi/2.0)
                yR = self.L1 * np.sin(Rj[0,0] + np.pi/2.0) + self.L2 * np.sin(Rj[0,0]-Rj[1,0] + np.pi/2.0)
                xL = self.L1 * np.cos(Lj[0,0] + np.pi/2.0) + self.L2 * np.cos(Lj[0,0]+Lj[1,0] + np.pi/2.0)
                yL = self.L1 * np.sin(Lj[0,0] + np.pi/2.0) + self.L2 * np.sin(Lj[0,0]+Lj[1,0] + np.pi/2.0)
                
                P_R = np.array([xR, yR])
                P_L = np.array([xL, yL])
                Prel_R = self.Pc_R - P_R
                Prel_L = self.Pc_L - P_L
                l_R = np.sqrt(Prel_R[0]*Prel_R[0] + Prel_R[1]*Prel_R[1])
                l_L = np.sqrt(Prel_L[0]*Prel_L[0] + Prel_L[1]*Prel_L[1])
                p_R = np.array([[l_R],[np.arctan2(-Prel_R[1],-Prel_R[0])]])
                p_L = np.array([[l_L],[np.arctan2(Prel_L[1],Prel_L[0])]])
                
                
                p_R[1,0] += -0.15
                p_L[1,0] += 0.15
                
                if self.dims['u'] > 2:
                    observation = np.array([[p_R[0,0] * 10 - 1.0, p_L[0,0] * 10 - 1.0, p_R[1,0], p_L[1,0], 
                                                  (e_sensor* np.pi / 180.0 - p_R[1,0]) , (e_sensor* np.pi / 180.0 - p_L[1,0]), 
                                                  (self.g[0][0] - e_sensor * np.pi / 180.0),
                                                  des_Fp_R[0,0] * 0.1, des_Fp_L[0,0] * 0.1, 
                                                   vel_R[0,0], vel_R[1,0], vel_L[0,0], vel_L[1,0],
                                                  self.stiffness, self.stiffness_lim]])
                else:
                    observation = np.array([[p_R[0,0] * 10 - 1.0, p_L[0,0] * 10 - 1.0, p_R[1,0], p_L[1,0], 
                                                  (e_sensor* np.pi / 180.0 - p_R[1,0]) , (e_sensor* np.pi / 180.0 - p_L[1,0]), 
                                                  (self.g[0][0] - e_sensor * np.pi / 180.0),
                                                  des_Fp_R[0,0] * 0.1, des_Fp_L[0,0] * 0.1, 
                                                   vel_R[0,0], vel_R[1,0], vel_L[0,0], vel_L[1,0]]])
                # print(des_l)
                
                obs_dict_new = dict(observation=observation, 
                                    achieved_goal=np.array([[e_sensor* np.pi / 180.0, vel_R[0,0], vel_R[1,0], vel_L[0,0], vel_L[1,0], des_Fp_R[0,0] * 0.1, des_Fp_L[0,0] * 0.1]]), 
                                    desired_goal = self.g)
                
                done = [False] if t < 3*self.T-1 else [True]
                info = [{
                    'is_success': self._is_success(obs_dict_new['achieved_goal'][0][0], obs_dict_new['desired_goal'][0][0]),
                }]
                
                episodeInfo.append(info.copy()[0])
                episodeAct.append(u.copy()[0])
                episodeObs.append(dict(observation=o.copy()[0],
                                        achieved_goal=ag.copy()[0],
                                        desired_goal=self.g.copy()[0]))
                episodeForce.append(f_sensor)
                print(f_sensor)
                o_new = obs_dict_new['observation']
                ag_new = obs_dict_new['achieved_goal']
                success = np.array([i.get('is_success', 0.0) for i in info])
                success2 = (np.float32(f_sensor < self.object_fragility))
                if f_sensor > max_force: max_force = f_sensor
    
                if any(done):
                    # here we assume all environments are done is ~same number of steps, so we terminate rollouts whenever any of the envs returns done
                    # trick with using vecenvs is not to add the obs from the environments that are "done", because those are already observations
                    # after a reset
                    break
    
                for i, info_dict in enumerate(info):
                    for idx, key in enumerate(self.info_keys):
                        info_values[idx][t, i] = info[i][key]
    
                # if np.isnan(o_new).any():
                #     self.logger.warn('NaN caught during rollout generation. Trying again...')
                #     self.reset_all_rollouts()
                #     return self.generate_rollouts()
    
                dones.append(done)
                obs.append(o.copy())
                achieved_goals.append(ag.copy())
                successes.append(success.copy())
                successes2.append(success2.copy())
                acts.append(u.copy())
                goals.append(self.g.copy())
                o[...] = o_new
                ag[...] = ag_new
    #        print("--------------------New Rollout--------------------")
    
            
            
            self.actions_list.append(episodeAct)
            self.obs_list.append(episodeObs)
            self.infos_list.append(episodeInfo)
            self.forces_list.append(episodeForce)
            
            obs.append(o.copy())
            achieved_goals.append(ag.copy())
    
            episode = dict(o=obs,
                           u=acts,
                           g=goals,
                           ag=achieved_goals)
            for key, value in zip(self.info_keys, info_values):
                episode['info_{}'.format(key)] = value
            if self.compute_Q == True: 
                print("Maximum Force: {}".format(max_force))
                print("Achieved: {}".format(int(e_sensor)))
            # stats
            successful = np.array(successes)[-1, :]
            successful2 = np.array(successes2)
            assert successful.shape == (self.rollout_batch_size,)
            success_rate = np.mean(successful)
            success_rate2 = np.mean(successful2.mean(axis=0))
            success_rate3 = np.mean(successful2.min(axis=0) * successful)
            self.success_history.append(success_rate)
            self.success_history2.append(success_rate2)
            self.success_history3.append(success_rate3)
            if self.compute_Q:
                self.Q_history.append(np.mean(Qs))
                self.F_history.append(np.mean(Fs))
                self.K_history.append(np.mean(Ks))
            self.n_episodes += self.rollout_batch_size

        return convert_episode_to_batch_major(episode)

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.success_history2.clear()
        self.success_history3.clear()
        self.Q_history.clear()
        self.F_history.clear()
        self.K_history.clear()

    def current_success_rate(self):
        return np.mean(self.success_history)

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
            logs += [('mean_F', np.mean(self.F_history))]
            logs += [('mean_K', np.mean(self.K_history))]
        logs += [('success_rate2', np.mean(self.success_history2))]
        logs += [('success_rate3', np.mean(self.success_history3))]
        logs += [('episode', self.n_episodes)]

        if prefix != '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs
        
    def _is_success(self, ag, g):
        return (np.linalg.norm(ag - g) < np.pi/32.0).astype(np.float32)

