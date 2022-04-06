import numpy as np

from gym.envs.robotics import rotations, robot_env, utils

R_j = np.matrix([[0.01575,0],
                  [-0.01575, 0.01575]])
R_j_inv = np.linalg.inv(R_j)
R_j_L = np.matrix([[0.01575,0],
                  [0.01575, 0.01575]])
R_j_inv_L = np.linalg.inv(R_j_L)
R_e = np.matrix([[0.0034597,0],
                  [0, 0.0034597]])
L1 = 0.1
L2 = 0.075

Ksc = 700

Rm = 0.0285

def ToQuaternion(yaw, pitch, roll): # yaw (Z), pitch (Y), roll (X)
    #// Abbreviations for the various angular functions
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class NuFingersEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, target_range,
        distance_threshold, initial_qpos, reward_type, pert_type='none', n_actions=2
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.model_path = model_path
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.broken_table = False
        self.broken_object = False
        self.max_stiffness = 1.0
        self.prev_stiffness = self.max_stiffness
        self.prev_stiffness_limit = self.max_stiffness
        self.object_fragility = 4.5
        self.min_grip = 0.0
        self.fric_mu = 0.2
        self.grav_const = 9.81
        self.prev_force = 0.0
        self.prev_lforce = 0.0
        self.prev_rforce = 0.0
        self.prev_oforce = 0.0
        self.previous_input = 0
        self.remaining_timestep = 75
        self.des_l = 0.08
        self.des_th = 0.
        self.des_Fp_R = np.array([[0.0],[0.0]])
        self.des_Fp_L = np.array([[0.0],[0.0]])
        self.Rj = np.array([[initial_qpos['Joint_1_R']],[initial_qpos['Joint_2_R']]])
        self.Lj = np.array([[initial_qpos['Joint_1_L']],[initial_qpos['Joint_2_L']]])
        self.Prev_Rj = np.array([[initial_qpos['Joint_1_R']],[initial_qpos['Joint_2_R']]])
        self.Prev_Lj = np.array([[initial_qpos['Joint_1_L']],[initial_qpos['Joint_2_L']]])
        self.Pc_R = np.array([-0.0635, 0.12])
        self.Pc_L = np.array([0.0635, 0.12])
        self.P_R = np.array([L1 * np.cos(self.Rj[0,0] + np.pi/2.0) + L2 * np.cos(self.Rj[0,0]-self.Rj[1,0] + np.pi/2.0), L1 * np.sin(self.Rj[0,0] + np.pi/2.0) + L2 * np.sin(self.Rj[0,0]-self.Rj[1,0] + np.pi/2.0)])
        self.P_L = np.array([L1 * np.cos(self.Lj[0,0] + np.pi/2.0) + L2 * np.cos(self.Lj[0,0]+self.Lj[1,0] + np.pi/2.0), L1 * np.sin(self.Lj[0,0] + np.pi/2.0) + L2 * np.sin(self.Lj[0,0]+self.Lj[1,0] + np.pi/2.0)])
        self.Prel_R = self.Pc_R - self.P_R
        self.Prel_L = self.Pc_L - self.P_L
        l_R = np.sqrt(self.Prel_R[0]*self.Prel_R[0] + self.Prel_R[1]*self.Prel_R[1])
        l_L = np.sqrt(self.Prel_L[0]*self.Prel_L[0] + self.Prel_L[1]*self.Prel_L[1])
        self.p_R = np.array([[l_R],[np.arctan2(-self.Prel_R[1],-self.Prel_R[0])]])
        self.p_L = np.array([[l_L],[np.arctan2(self.Prel_L[1],self.Prel_L[0])]])
        self.Prev_p_R = self.p_R
        self.Prev_p_L = self.p_L
        self.l_step_limit = 0.02
        self.th_step_limit = np.pi/60.0
        self.l_limit = 0.08
        self.th_limit = np.pi/3.0
        self.pert_type = pert_type
        self.n_actions = n_actions
        self.goal_dim = 1
        self.velgoal_dim = 4

        super(NuFingersEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, initial_qpos=initial_qpos, n_actions=n_actions)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        try: 
            d = goal_distance(achieved_goal[:,:self.goal_dim], goal[:,:self.goal_dim])
            fragile_goal = 1 * np.linalg.norm(achieved_goal[:,self.goal_dim : self.goal_dim + self.velgoal_dim] - goal[:,self.goal_dim : self.goal_dim + self.velgoal_dim], axis=-1) + 0.0 * np.linalg.norm((achieved_goal[:,self.goal_dim + self.velgoal_dim:] - goal[:,self.goal_dim + self.velgoal_dim:])*((achieved_goal[:,self.goal_dim + self.velgoal_dim:] - goal[:,self.goal_dim + self.velgoal_dim:]) < 0), axis=-1)
        except: 
            d = goal_distance(achieved_goal[:self.goal_dim], goal[:self.goal_dim])
            fragile_goal = 1 * np.linalg.norm(achieved_goal[self.goal_dim : self.goal_dim + self.velgoal_dim] - goal[self.goal_dim : self.goal_dim + self.velgoal_dim], axis=-1) + 0.0 * np.linalg.norm((achieved_goal[self.goal_dim + self.velgoal_dim:] - goal[self.goal_dim + self.velgoal_dim:])*((achieved_goal[self.goal_dim + self.velgoal_dim:] - goal[self.goal_dim + self.velgoal_dim:]) < 0), axis=-1)
        
        return -(d > self.distance_threshold).astype(np.float32) - np.float32(fragile_goal)
        

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        self.sim.forward()

    def _set_action(self, action):
        action = action.copy()  # ensure that we don't change the action outside of this scope
        
        pos_ctrl = action[:2].copy()
        change_l = np.clip(pos_ctrl[0]/50.0, -self.l_step_limit, self.l_step_limit)
        change_th = np.clip(pos_ctrl[1]/20.0, -self.th_step_limit, self.th_step_limit)
        stiffness_ctrl = 0.0
        stiffness_limit = 0.0
        
        if action.shape[0] > 2:
            stiffness_limit = 0.2 * self.max_stiffness * action[3]
            
            self.prev_stiffness_limit += stiffness_limit
            self.prev_stiffness_limit = np.max([np.min([self.prev_stiffness_limit, self.max_stiffness]), self.max_stiffness / 25.0])
            
            stiffness_ctrl = 0.2 * self.max_stiffness * action[2]
            
            self.prev_stiffness += stiffness_ctrl
            self.prev_stiffness = np.max([np.min([self.prev_stiffness, self.prev_stiffness_limit]), 0.0])
            
        r = np.array([[self.prev_stiffness], [1.0]])
        
        self.des_l = self.des_l + change_l
        self.des_th = self.des_th + change_th
        
        des_p_R = np.array([[np.min([np.max([self.des_l/2.0, -self.l_limit]), self.l_limit])], [np.min([np.max([self.des_th, -self.th_limit]), self.th_limit])]])
        
        des_p_L = des_p_R
        
        Jp_R = np.matrix([[-self.Prel_R[0]/self.p_R[0,0], -self.Prel_R[1]/self.p_R[0,0]],[self.Prel_R[1]/self.p_R[0,0]/self.p_R[0,0], -self.Prel_R[0]/self.p_R[0,0]/self.p_R[0,0]]])
        Jp_L = np.matrix([[-self.Prel_L[0]/self.p_L[0,0], -self.Prel_L[1]/self.p_L[0,0]],[self.Prel_L[1]/self.p_L[0,0]/self.p_L[0,0], -self.Prel_L[0]/self.p_L[0,0]/self.p_L[0,0]]])
        Jp_inv_R = np.matrix([[Jp_R[1,1] / (Jp_R[0,0]*Jp_R[1,1] - Jp_R[0,1]*Jp_R[1,0]), -Jp_R[0,1] / (Jp_R[0,0]*Jp_R[1,1] - Jp_R[0,1]*Jp_R[1,0])], [-Jp_R[1,0] / (Jp_R[0,0]*Jp_R[1,1] - Jp_R[0,1]*Jp_R[1,0]), Jp_R[0,0] / (Jp_R[0,0]*Jp_R[1,1] - Jp_R[0,1]*Jp_R[1,0])]])
        Jp_inv_L = np.matrix([[Jp_L[1,1] / (Jp_L[0,0]*Jp_L[1,1] - Jp_L[0,1]*Jp_L[1,0]), -Jp_L[0,1] / (Jp_L[0,0]*Jp_L[1,1] - Jp_L[0,1]*Jp_L[1,0])], [-Jp_L[1,0] / (Jp_L[0,0]*Jp_L[1,1] - Jp_L[0,1]*Jp_L[1,0]), Jp_L[0,0] / (Jp_L[0,0]*Jp_L[1,1] - Jp_L[0,1]*Jp_L[1,0])]])
        J_R = np.matrix([[-self.P_R[1], L2 * np.cos(self.Rj[0,0]-self.Rj[1,0])], 
                         [self.P_R[0], L2 * np.sin(self.Rj[0,0]-self.Rj[1,0])]])
        J_L = np.matrix([[-self.P_L[1], -L2 * np.cos(self.Lj[0,0]+self.Lj[1,0])], 
                         [self.P_L[0], -L2 * np.sin(self.Lj[0,0]+self.Lj[1,0])]])
        J_inv_R = np.matrix([[J_R[1,1] / (J_R[0,0]*J_R[1,1] - J_R[0,1]*J_R[1,0]), -J_R[0,1] / (J_R[0,0]*J_R[1,1] - J_R[0,1]*J_R[1,0])], [-J_R[1,0] / (J_R[0,0]*J_R[1,1] - J_R[0,1]*J_R[1,0]), J_R[0,0] / (J_R[0,0]*J_R[1,1] - J_R[0,1]*J_R[1,0])]])
        J_inv_L = np.matrix([[J_L[1,1] / (J_L[0,0]*J_L[1,1] - J_L[0,1]*J_L[1,0]), -J_L[0,1] / (J_L[0,0]*J_L[1,1] - J_L[0,1]*J_L[1,0])], [-J_L[1,0] / (J_L[0,0]*J_L[1,1] - J_L[0,1]*J_L[1,0]), J_L[0,0] / (J_L[0,0]*J_L[1,1] - J_L[0,1]*J_L[1,0])]])
        
        max_kj_R = np.transpose(R_j) * np.matrix([[1400, 0],[0, 1400]]) * R_j
        max_kj_L = np.transpose(R_j_L) * np.matrix([[1400, 0],[0, 1400]]) * R_j_L
        
        max_k_R = np.transpose(J_inv_R) * max_kj_R * J_inv_R
        max_k_L = np.transpose(J_inv_L) * max_kj_L * J_inv_L
        max_kp_R = np.transpose(Jp_inv_R) * max_k_R * Jp_inv_R
        max_kp_L = np.transpose(Jp_inv_L) * max_k_L * Jp_inv_L
        max_kp_R[0,1] = 0
        max_kp_R[1,0] = 0
        max_kp_L[0,1] = 0
        max_kp_L[1,0] = 0
        self.des_Fp_R = max_kp_R * (r * (des_p_R - self.p_R))
        self.des_Fp_L = max_kp_L * (r * (des_p_L - self.p_L))
        # self.sim.model.tendon_stiffness[self.sim.model.tendon_name2id('T1_R')] = 1400 * self.prev_stiffness_limit
        # self.sim.model.tendon_stiffness[self.sim.model.tendon_name2id('T2_R')] = 1400 * self.prev_stiffness_limit
        # self.sim.model.tendon_stiffness[self.sim.model.tendon_name2id('T1_L')] = 1400 * self.prev_stiffness_limit
        # self.sim.model.tendon_stiffness[self.sim.model.tendon_name2id('T2_L')] = 1400 * self.prev_stiffness_limit
        
        des_F_R = np.transpose(Jp_R) * self.des_Fp_R
        des_F_L = np.transpose(Jp_L) * self.des_Fp_L
        des_tau_R = np.transpose(J_R) * des_F_R
        des_tau_L = np.transpose(J_L) * des_F_L
        des_mR = ((np.matrix([[1/self.sim.model.tendon_stiffness[self.sim.model.tendon_name2id('T1_R')], 0],[0, 1/self.sim.model.tendon_stiffness[self.sim.model.tendon_name2id('T2_R')]]]) * np.transpose(R_j_inv)*des_tau_R) + R_j * self.Rj) / Rm 
        des_mL = ((np.matrix([[1/self.sim.model.tendon_stiffness[self.sim.model.tendon_name2id('T1_L')], 0],[0, 1/self.sim.model.tendon_stiffness[self.sim.model.tendon_name2id('T2_L')]]]) * np.transpose(R_j_inv_L)*des_tau_L) + R_j_L * self.Lj) / Rm
        
        prob = 0.1 if self.pert_type == 'delay' else -0.1
        if np.random.random() > prob:
            self.sim.data.ctrl[0] = des_mL[0,0]
            self.sim.data.ctrl[1] = des_mL[1,0]
            self.sim.data.ctrl[2] = des_mR[0,0]
            self.sim.data.ctrl[3] = des_mR[1,0]
            self.previous_input = self.sim.data.ctrl
        else:
            try: self.sim.data.ctrl = self.previous_input
            except: 
                self.sim.data.ctrl[0] = 0.3
                self.sim.data.ctrl[1] = -0.66
                self.sim.data.ctrl[2] = -0.3
                self.sim.data.ctrl[3] = -0.66
        
        # self.sim.data.ctrl[0] = des_mL[0,0]
        # self.sim.data.ctrl[1] = des_mL[1,0]
        # self.sim.data.ctrl[2] = des_mR[0,0]
        # self.sim.data.ctrl[3] = des_mR[1,0]

    def _get_obs(self):
        # positions
        self.remaining_timestep -= 1
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        l_finger_force = self.prev_lforce + (self.des_Fp_R[0,0] - self.prev_lforce) * dt / 0.5
        r_finger_force = self.prev_rforce + (self.des_Fp_L[0,0] - self.prev_rforce) * dt / 0.5  
        self.prev_oforce += (self.sim.data.sensordata[self.sim.model.sensor_name2id('object_frc')] - self.prev_oforce) * dt / 0.5 
        if (l_finger_force + r_finger_force < -self.object_fragility):
            self.sim.model.geom_rgba[-3][1:3] = 0.1
            self.sim.model.geom_rgba[-2][1:3] = 0.1
        self.Rj = np.array([[self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Joint_1_R')]]],
                       [self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Joint_2_R')]]]])
        self.Lj = np.array([[self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Joint_1_L')]]],
                       [self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Joint_2_L')]]]])
        
        vel_R = self.Rj - self.Prev_Rj
        vel_L = self.Lj - self.Prev_Lj
        vel_p_R = self.p_R - self.Prev_p_R
        vel_p_L = self.p_L - self.Prev_p_L
        self.Prev_Rj = self.Rj
        self.Prev_Lj = self.Lj
        self.Prev_p_R = self.p_R
        self.Prev_p_L = self.p_L
        
        xR = L1 * np.cos(self.Rj[0,0] + np.pi/2.0) + L2 * np.cos(self.Rj[0,0]-self.Rj[1,0] + np.pi/2.0)
        yR = L1 * np.sin(self.Rj[0,0] + np.pi/2.0) + L2 * np.sin(self.Rj[0,0]-self.Rj[1,0] + np.pi/2.0)
        xL = L1 * np.cos(self.Lj[0,0] + np.pi/2.0) + L2 * np.cos(self.Lj[0,0]+self.Lj[1,0] + np.pi/2.0)
        yL = L1 * np.sin(self.Lj[0,0] + np.pi/2.0) + L2 * np.sin(self.Lj[0,0]+self.Lj[1,0] + np.pi/2.0)
        
        self.P_R = np.array([xR, yR])
        self.P_L = np.array([xL, yL])
        
        self.Prel_R = self.Pc_R - self.P_R
        self.Prel_L = self.Pc_L - self.P_L
        l_R = np.sqrt(self.Prel_R[0]*self.Prel_R[0] + self.Prel_R[1]*self.Prel_R[1])
        l_L = np.sqrt(self.Prel_L[0]*self.Prel_L[0] + self.Prel_L[1]*self.Prel_L[1])
        self.p_R = np.array([[l_R],[np.arctan2(-self.Prel_R[1],-self.Prel_R[0])]])
        self.p_L = np.array([[l_L],[np.arctan2(self.Prel_L[1],self.Prel_L[0])]])
        
#        print(self.sim.data.sensordata[self.sim.model.sensor_name2id('object_frc')])
        if self.pert_type == 'none' or self.pert_type == 'pert':
            obj_rot = self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Sensor_joint')]]
        else:
            obj_rot = self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Sensor_joint')]] + 0.04 * (np.random.random() - 0.5)
        
        if self.pert_type != 'none' and self.pert_type != 'meas':
            if self.prev_oforce > 0.1: self.sim.data.qvel[self.sim.model.joint_name2id('Sensor_joint')] += 1 * (np.random.random()-0.5)
        if self.n_actions == 4:
            observation = np.array([self.p_R[0,0] * 10 - 1.0, self.p_L[0,0] * 10 - 1.0, self.p_R[1,0], self.p_L[1,0], 
                                    (obj_rot - self.p_R[1,0]), (obj_rot - self.p_L[1,0]), 
                                    (self.goal[0] - obj_rot),
                                    l_finger_force * 0.1, r_finger_force * 0.1, 
                                    vel_R[0,0], vel_R[1,0], vel_L[0,0], vel_L[1,0],
                                    #vel_p_R[0,0], vel_p_L[0,0],
                                    self.prev_stiffness, self.prev_stiffness_limit])
        else:
            observation = np.array([self.p_R[0,0] * 10 - 1.0, self.p_L[0,0] * 10 - 1.0, self.p_R[1,0], self.p_L[1,0], 
                                    (obj_rot - self.p_R[1,0]), (obj_rot - self.p_L[1,0]), 
                                    (self.goal[0] - obj_rot),
                                    l_finger_force * 0.1, r_finger_force * 0.1, 
                                    vel_R[0,0], vel_R[1,0], vel_L[0,0], vel_L[1,0],
                                    #vel_p_R[0,0], vel_p_L[0,0]]
                                   ])
        # print(observation)
        # modified_obs = dict(observation=observation, achieved_goal=np.array([obj_rot, 100.0*vel_p_R[0,0], 100.0*vel_p_L[0,0], l_finger_force * 0.1, r_finger_force * 0.1]), desired_goal = self.goal)
        modified_obs = dict(observation=observation, achieved_goal=np.array([obj_rot, vel_R[0,0], vel_R[1,0], vel_L[0,0], vel_L[1,0], l_finger_force * 0.1, r_finger_force * 0.1]), desired_goal = self.goal)
        # modified_obs = dict(observation=observation, achieved_goal=np.array([obj_rot, l_finger_force, r_finger_force]), desired_goal = self.goal)
        
        self.prev_lforce = l_finger_force
        self.prev_rforce = r_finger_force
        
        return modified_obs

    def _viewer_setup(self):
        self.viewer.cam.distance = 0.5
        self.viewer.cam.azimuth = 32.
        self.viewer.cam.elevation = -30.

    def _render_callback(self):
        # Visualize target.
        self.sim.model.body_quat[self.sim.model.body_name2id('target_body')] = ToQuaternion(0, -np.pi/2.0, self.goal[0])
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        
        # reset the broken objects
        self.broken_object = False
        
        self.sim.model.geom_rgba[-3][1:3] = 0.93
        self.sim.model.geom_rgba[-2][1:3] = 0.93
        
        # reset stiffness
        self.prev_stiffness = self.max_stiffness
        self.prev_stiffness_limit = self.max_stiffness
        
        # reset forces
        self.prev_lforce = 0.0
        self.prev_rforce = 0.0
        self.prev_oforce = 0.0
        
        self.remaining_timestep = 75

        # Randomize start position of object.
        self.sim.data.ctrl[0] = 0.3
        self.sim.data.ctrl[1] = -0.66
        self.sim.data.ctrl[2] = -0.3
        self.sim.data.ctrl[3] = -0.66
        self.des_l = 0.08
        self.des_th = 0.

        self.sim.forward()
        return True

    def _sample_goal(self):
        goal = np.array([(7*np.pi/32.0 + (np.random.random()*np.pi/16.0 - np.pi/32.0)) * np.sign(np.random.random() - 0.5)])#self.np_random.uniform(-self.target_range, self.target_range, size=1)
        self.sim.model.body_pos[self.sim.model.body_name2id('Sensor_base')][0] = -0.0327 + 0.01 * (np.random.random()- 0.5)
        self.sim.model.body_pos[self.sim.model.body_name2id('target_body')][0] = self.sim.model.body_pos[self.sim.model.body_name2id('Sensor_base')][0]
        self.sim.model.geom_size[self.sim.model.geom_name2id('Fake_object_geom')][1] = 0.02 + (np.random.random()-0.5)*0.01
        self.sim.model.site_size[self.sim.model.site_name2id('object_force')][1] =self.sim.model.geom_size[self.sim.model.geom_name2id('Fake_object_geom')][1] + 0.001
        self.sim.model.tendon_stiffness[self.sim.model.tendon_name2id('T0_L')] = (np.random.random())*1e2
        self.sim.model.tendon_stiffness[self.sim.model.tendon_name2id('T0_R')] = self.sim.model.tendon_stiffness[self.sim.model.tendon_name2id('T0_L')]
        return np.concatenate([goal.copy(), np.concatenate([np.zeros(self.velgoal_dim), [0.0, 0.0]])])
        # return np.concatenate([goal.copy(), np.array([0.0, 0.0])])

    def _is_success(self, achieved_goal, desired_goal):
        try: 
            d = goal_distance(achieved_goal[:,:self.goal_dim], desired_goal[:,:self.goal_dim])
        except: 
            d = goal_distance(achieved_goal[:self.goal_dim], desired_goal[:self.goal_dim])
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.data.ctrl[0] = 0.3
        self.sim.data.ctrl[1] = -0.66
        self.sim.data.ctrl[2] = -0.3
        self.sim.data.ctrl[3] = -0.66
                
        self.sim.forward()
        for _ in range(10):
            self.sim.step()

    def render(self, mode='human', width=500, height=500):
        return super(NuFingersEnv, self).render(mode, width, height)
