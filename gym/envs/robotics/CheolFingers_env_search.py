import numpy as np

from gym.envs.robotics import rotations, robot_env, utils

R_j = np.matrix([[0.012,0],
                  [-0.012, -0.012]])
R_j_inv = np.linalg.inv(R_j)
R_j_L = np.matrix([[0.012,0],
                  [0.012, 0.012]])
R_j_inv_L = np.linalg.inv(R_j_L)
R_e = np.matrix([[0.0034597,0],
                  [0, 0.0034597]])
L1 = 0.1
L2 = 0.075

m1 = 0.1
m2 = 0.05

# Ksc = 700

Rm = 0.012

def tansig(x):
    tansig = 2/(1+np.exp(-2*x))-1
    return tansig

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

def ToRPY(quat):
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]
    
    roll = np.arctan2(2*y*w - 2*x*z, 1 - 2*y*y - 2*z*z)
    pitch = np.arctan2(2*x*w - 2*y*z, 1 - 2*x*x - 2*z*z)
    yaw = np.arcsin(2*x*y + 2*z*w)
    
    return np.array([roll, pitch, yaw])

def QuatPosToTrot(Q,pos):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0] # real part
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 4x4 homogeneous matrix
    rot_matrix = np.matrix([[r00, r01, r02, pos[0]],
                           [r10, r11, r12, pos[1]],
                           [r20, r21, r22, pos[2]],
                           [0, 0, 0, 1]])
                            
    return rot_matrix

def trot(d,th,a,al):
    trot_mat = np.matrix([[np.cos(th), -np.sin(th)*np.cos(al), np.sin(th)*np.sin(al), a*np.cos(th)],
        [np.sin(th), np.cos(th)*np.cos(al), -np.cos(th)*np.sin(al), a*np.sin(th)],
        [0, np.sin(al), np.cos(al), d],
        [0, 0, 0, 1]]);
    return trot_mat

def trans(mat,dx,dy,dz):
    trans_mat = mat*np.matrix([[dx], [dy], [dz], [1]]);
    return trans_mat


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class CheolFingersEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, target_range,
        distance_threshold, initial_qpos, reward_type, pert_type='none', n_actions=3, eval_env=False
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
        self.max_stiffness = 0.8
        self.min_stiffness = 0.1
        self.prev_stiffness = 0.8
        self.prev_stiffness_limit = 0.8
        self.actual_max_stiffness = 2.30399e-2
        self.actual_max_stiffness2 = 1.15199e-2
        self.actual_stiffness = self.actual_max_stiffness
        self.actual_stiffness2 = self.actual_max_stiffness2
        self.object_fragility = 90.0
        self.min_grip = 0.0
        self.fric_mu = 0.7
        self.grav_const = 9.81
        self.prev_force = 0.0
        self.prev_lforce = 0.0
        self.prev_rforce = 0.0
        self.prev_oforce = 0.0
        self.actual_force = 0.0
        self.Rj = np.array([[initial_qpos['Joint_1_R']],[initial_qpos['Joint_2_R']]])
        self.Lj = np.array([[initial_qpos['Joint_1_L']],[initial_qpos['Joint_2_L']]])
        self.Prev_Rj = np.array([[initial_qpos['Joint_1_R']],[initial_qpos['Joint_2_R']]])
        self.Prev_Lj = np.array([[initial_qpos['Joint_1_L']],[initial_qpos['Joint_2_L']]])
        self.Pc = np.array([0., 0.])
        self.P_R = np.array([L1 * np.cos(self.Rj[0,0] + 3*np.pi/4.0) + L2 * np.cos(self.Rj[0,0]-self.Rj[1,0] + 3*np.pi/4.0), L1 * np.sin(self.Rj[0,0] + 3*np.pi/4.0) + L2 * np.sin(self.Rj[0,0]-self.Rj[1,0] + 3*np.pi/4.0)])
        self.P_L = np.array([L1 * np.cos(self.Lj[0,0] + 3*np.pi/4.0) + L2 * np.cos(self.Lj[0,0]+self.Lj[1,0] + 3*np.pi/4.0), L1 * np.sin(self.Lj[0,0] + 3*np.pi/4.0) + L2 * np.sin(self.Lj[0,0]+self.Lj[1,0] + 3*np.pi/4.0)])
        self.joint_pos = np.zeros([4,1])
        self.p = np.array([[0.],[0.],[0.]])
        self.des_p = np.array([[0.],[0.],[0.]])
        self.Prev_p = self.p
        self.vel_p = self.p - self.Prev_p
        self.des_Fp = np.array([[0.],[0.],[0.]])
        self.lower_limit = 0.0 if eval_env else -np.pi/2.0
        self.upper_limit = np.pi/2.0 if eval_env else 0.0
        self.pert_type = pert_type
        self.n_actions = n_actions
        self.des_tau = np.zeros([2,1])
        self.eval_env = eval_env
        self.est_dim = 1
        self.force_dim = 1
        self.vel_dim = 1
        self.flag_dim = 1
        self.th = np.zeros([2,1])
        self.Prev_th = np.zeros([2,1])
        self.des_th = np.zeros([2,1])
        self.vel_th = np.zeros([3,2,1])
        self.vel_threshold = 1e-2
        self.des_mR = np.zeros([2,1])
        self.des_mL = np.zeros([2,1])
        self.update_bool = False
        self.obj_weight = 2e3
        self.mocap_offset = np.zeros([2,1])

        super(CheolFingersEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, initial_qpos=initial_qpos, n_actions=n_actions)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        try: 
            intention = np.linalg.norm(achieved_goal[:,:self.est_dim]< goal[:,:self.est_dim], axis=-1)
            torque = np.linalg.norm(achieved_goal[:,self.est_dim:self.est_dim+self.force_dim] - goal[:,self.est_dim:self.est_dim+self.force_dim], axis=-1)
            velocity = np.linalg.norm((achieved_goal[:,self.est_dim+self.force_dim:self.est_dim+self.force_dim+self.vel_dim] - goal[:,self.est_dim+self.force_dim:self.est_dim+self.force_dim+self.vel_dim]), axis=-1)
        except: 
            intention = np.linalg.norm(achieved_goal[:self.est_dim]< goal[:self.est_dim], axis=-1)
            torque = np.linalg.norm(achieved_goal[self.est_dim:self.est_dim+self.force_dim] - goal[self.est_dim:self.est_dim+self.force_dim], axis=-1)
            velocity = np.linalg.norm((achieved_goal[self.est_dim+self.force_dim:self.est_dim+self.force_dim+self.vel_dim] - goal[self.est_dim+self.force_dim:self.est_dim+self.force_dim+self.vel_dim]), axis=-1)
        # print(-(intention).astype(np.float32), -(velocity > self.vel_threshold).astype(np.float32))
        penalty = -(intention).astype(np.float32) -(velocity > self.vel_threshold).astype(np.float32) -(torque < 1e-4).astype(np.float32)
        penalty = (penalty < 0).astype(np.float32)
        return -0.5*penalty -1.0e-3*(torque).astype(np.float32)
        

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        self.sim.forward()

    def _set_action(self, action):
        action = action.copy()  # ensure that we don't change the action outside of this scope
        
        pos_ctrl = action[:1].copy()
        
        change_th = pos_ctrl[0] * np.pi/9 if self.eval_env else -pos_ctrl[0] * np.pi/9
        
        stiffness_ctrl = 0.0
        stiffness_limit = 0.0
        
        if action.shape[0] > 1:
            stiffness_limit = 0.5 * self.max_stiffness * action[2]
            
            self.prev_stiffness_limit += stiffness_limit
            self.prev_stiffness_limit = np.max([np.min([self.prev_stiffness_limit, self.max_stiffness]), self.min_stiffness])
            self.actual_stiffness = self.actual_max_stiffness * self.prev_stiffness_limit
            self.actual_stiffness2 = self.actual_max_stiffness2 * self.prev_stiffness_limit
            
            stiffness_ctrl = 0.5 * self.max_stiffness * action[1]
            
            self.prev_stiffness += stiffness_ctrl
            self.prev_stiffness = np.max([np.min([self.prev_stiffness, self.prev_stiffness_limit]), 0.0])
            
            # print(self.prev_stiffness_limit)
            
        self.des_th[0,0] = np.clip(self.des_th[0,0] + change_th, self.lower_limit, self.upper_limit)
        # print(self.prev_stiffness, self.prev_stiffness_limit)
        # calculating desired joint positions
        
        r = np.array([[self.prev_stiffness],[self.max_stiffness]])
        mocap_th = self.th + r * (self.des_th - self.th) + self.mocap_offset
        if self.eval_env == True: # right finger
            x_r = L1 * np.cos(mocap_th[0,0] + 1*np.pi/4.0) + L2 * np.cos(mocap_th[0,0]-mocap_th[1,0] + 1*np.pi/4.0)
            y_r = L1 * np.sin(mocap_th[0,0] + 1*np.pi/4.0) + L2 * np.sin(mocap_th[0,0]-mocap_th[1,0] + 1*np.pi/4.0)
            
            x_l = L1 * np.cos(0. + 3*np.pi/4.0) + L2 * np.cos(0.+ 3*np.pi/4.0)
            y_l = L1 * np.sin(0. + 3*np.pi/4.0) + L2 * np.sin(0. + 3*np.pi/4.0)
        else: # left finger
            x_l = L1 * np.cos(mocap_th[0,0] + 3*np.pi/4.0) + L2 * np.cos(mocap_th[0,0]+mocap_th[1,0] + 3*np.pi/4.0)
            y_l = L1 * np.sin(mocap_th[0,0] + 3*np.pi/4.0) + L2 * np.sin(mocap_th[0,0]+mocap_th[1,0] + 3*np.pi/4.0)
            
            x_r = L1 * np.cos(0. + 1*np.pi/4.0) + L2 * np.cos(0.+ 1*np.pi/4.0)
            y_r = L1 * np.sin(0. + 1*np.pi/4.0) + L2 * np.sin(0. + 1*np.pi/4.0)
        
        # welding the bodies to the mocap
        new_pose_l = np.array([-y_l + 0.1, x_l - 0.0635])
        new_pose_r = np.array([-y_r + 0.1, x_r + 0.0635])
        new_pose = np.array([new_pose_l, new_pose_r])
        
        for eq_type, obj1_id, obj2_id in zip(self.sim.model.eq_type,
                                          self.sim.model.eq_obj1id,
                                          self.sim.model.eq_obj2id):

            mocap_id = self.sim.model.body_mocapid[obj1_id]
            if mocap_id != -1:
                # obj1 is the mocap, obj2 is the welded body
                body_idx = obj2_id
            else:
                # obj2 is the mocap, obj1 is the welded body
                mocap_id = self.sim.model.body_mocapid[obj2_id]
                body_idx = obj1_id
    
            assert (mocap_id != -1)
            self.sim.data.mocap_pos[mocap_id][:] = self.sim.data.body_xpos[body_idx]
            # self.sim.data.mocap_quat[mocap_id][:] = self.sim.data.body_xquat[body_idx]
        self.sim.data.mocap_pos[:,:2] = new_pose
        
        # tendon space stiffness
        Ksc = np.matrix([[self.actual_stiffness, 0],[0, self.actual_stiffness2]])/Rm/Rm
        Ksc_L = np.matrix([[self.actual_stiffness, 0],[0, self.actual_stiffness2]])/Rm/Rm
        
        r = np.array([[self.prev_stiffness],[0.8]])
        
        if self.eval_env == True:
            # joint space stiffness
            max_kj_R = np.transpose(R_j) * Ksc * R_j
            max_kj_diag_R = np.matrix([[max_kj_R[0,0], 0], [0, max_kj_R[1,1]]])
            # desired joint space torques
            self.des_tau = max_kj_diag_R * (r * (self.des_th - self.th))
            # desired actuator positions
            self.des_mR = ((np.matrix([[1/Ksc[0,0], 0],[0, 1/Ksc[1,1]]]) * np.transpose(R_j_inv)*self.des_tau) + R_j * self.th) / Rm 
            self.actual_force = max_kj_diag_R * (mocap_th - self.th)
        else:
            # joint space stiffness
            max_kj_L = np.transpose(R_j_L) * Ksc_L * R_j_L
            max_kj_diag_L = np.matrix([[max_kj_L[0,0], 0], [0, max_kj_L[1,1]]])
            # desired joint space torques
            self.des_tau = max_kj_diag_L * (r * (self.des_th - self.th))
            # desired actuator positions
            self.des_mL = ((np.matrix([[1/Ksc_L[0,0], 0],[0, 1/Ksc_L[1,1]]]) * np.transpose(R_j_inv_L)*self.des_tau) + R_j_L * self.th) / Rm
            self.actual_force = max_kj_diag_L * (mocap_th - self.th)
        self.prev_force = self.actual_force[0,0] if self.eval_env else -self.actual_force[0,0]
        
        # prob = 0.005 if self.pert_type == 'delay' else -0.1
        # if np.random.random() > prob:
        #     self.sim.data.ctrl[0] = self.des_mL[0,0]
        #     self.sim.data.ctrl[1] = self.des_mL[1,0]
        #     self.sim.data.ctrl[2] = self.des_mR[0,0]
        #     self.sim.data.ctrl[3] = self.des_mR[1,0]
        #     self.previous_input = self.sim.data.ctrl
        # else:
        #     try: self.sim.data.ctrl = self.previous_input
        #     except: 
        #         self.sim.data.ctrl[0] = 0.14097741
        #         self.sim.data.ctrl[1] = -1.79383202
        #         self.sim.data.ctrl[2] = -0.14176121
        #         self.sim.data.ctrl[3] = -1.79471042
                
                   

    def _get_obs(self):
        # print(self.des_tau[0,0])
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
            
        self.Rj = np.array([[self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Joint_1_R')]]],
                   [-self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Joint_2_R')]]]])
        self.Lj = np.array([[self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Joint_1_L')]]],
                       [self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Joint_2_L')]]]])
        
        if self.eval_env == True:
            self.th = self.Rj
        else:
            self.th = self.Lj
        
        self.vel_th[2] = self.vel_th[1]
        self.vel_th[1] = self.vel_th[0]
        self.vel_th[0] = self.th - self.Prev_th
        
        self.Prev_th = self.th
        self.prev_oforce = self.sim.data.sensordata[self.sim.model.sensor_name2id('object_frc')] if self.actual_stiffness == self.actual_max_stiffness else self.prev_stiffness_limit * self.sim.data.sensordata[self.sim.model.sensor_name2id('object_frc')]
        if self.prev_oforce > self.object_fragility:
            self.sim.model.geom_rgba[-1][0:3] = np.array([0.8000, 0., 0.2824])
        # if self.pert_type != 'none' and self.pert_type != 'meas':
        # if self.grasped_flag == 0.1: self.sim.data.qvel[self.sim.model.joint_name2id('object:joint')+1] += 0.5*(np.random.random()-0.5)
        if self.n_actions == 3:
            if self.eval_env:
                observation = np.array([self.th[0,0], 
                                        self.des_th[0,0]-self.th[0,0], self.vel_th[0,0,0],
                                        self.prev_force, 
                                        self.prev_stiffness, self.prev_stiffness_limit
                                        ])
            else:
                observation = np.array([-self.th[0,0],
                                        -self.des_th[0,0]+self.th[0,0], -self.vel_th[0,0,0],
                                        self.prev_force,
                                        self.prev_stiffness, self.prev_stiffness_limit
                                        ])
                    
        else:
            if self.eval_env:
                observation = np.array([self.th[0,0], 
                                        self.des_th[0,0]-self.th[0,0], self.vel_th[0,0,0],
                                        self.prev_force,
                                        ])
            else:
                observation = np.array([-self.th[0,0],
                                        -self.des_th[0,0]+self.th[0,0], -self.vel_th[0,0,0],
                                        self.prev_force,
                                        ])
        # print(observation[1:4])
        modified_obs = dict(observation=observation, achieved_goal=np.array([observation[1],self.prev_oforce,observation[2]]), desired_goal = self.goal)
        return modified_obs

    def _viewer_setup(self):
        self.viewer.cam.distance = 0.5
        self.viewer.cam.azimuth = 32.
        self.viewer.cam.elevation = -30.

    def _render_callback(self):
        # Visualize target.
        self.sim.model.body_pos[self.sim.model.body_name2id('target_body'), 0:2] = np.array([-self.goal[1] + 0.1, self.goal[0] - 0.0635])
        self.sim.model.body_quat[self.sim.model.body_name2id('target_body')] = ToQuaternion(0, -np.pi/2.0, 0)
        
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        
        # reset the broken objects
        self.broken_object = False
        self.sim.model.geom_rgba[-1][0] = 0.2
        self.sim.model.geom_rgba[-1][2] = 0.3
        
        self.Rj = np.array([[0.],[0.]])
        self.Lj = np.array([[0.],[0.]])
        # print(self.p)
        
        self.Prev_p = self.p.copy()
        self.vel_p = self.p - self.Prev_p
        self.Pc = np.array([(self.P_R[0]+self.P_L[0]+0.127)/2,(self.P_R[1]+self.P_L[1])/2])
        
        self.joint_pos = np.concatenate([self.Lj, self.Rj])
        self.joint_pos[3,0] = -self.joint_pos[3,0]
        self.est_obj_pose = np.zeros([2,1])
        self.th = np.array([[0.],[0.]])
        self.Prev_th = np.array([[0.],[0.]])
        self.des_th = np.array([[0.],[0.]])
        self.vel_th = np.zeros([3,2,1])
        self.des_tau = np.array([[0.],[0.]])
        
        # reset stiffness
        self.prev_stiffness = 0.8
        self.prev_stiffness_limit = 0.8
        self.actual_stiffness = self.actual_max_stiffness
        self.actual_stiffness2 = self.actual_max_stiffness2
        # self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('AJ1_R'), 0] = self.actual_max_stiffness * self.prev_stiffness_limit 
        # self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('AJ1_R'), 1] = -self.actual_max_stiffness * self.prev_stiffness_limit 
        # self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('AJ2_R'), 0] = self.actual_max_stiffness2 * self.prev_stiffness_limit 
        # self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('AJ2_R'), 1] = -self.actual_max_stiffness2 * self.prev_stiffness_limit 
        # self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('AJ1_L'), 0] = self.actual_max_stiffness * self.prev_stiffness_limit 
        # self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('AJ1_L'), 1] = -self.actual_max_stiffness * self.prev_stiffness_limit 
        # self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('AJ2_L'), 0] = self.actual_max_stiffness2 * self.prev_stiffness_limit 
        # self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('AJ2_L'), 1] = -self.actual_max_stiffness2 * self.prev_stiffness_limit 
        # self.sim.model.dof_damping[self.sim.model.joint_name2id('Joint_1_L')] = self.actual_max_friction * self.prev_stiffness_limit 
        # self.sim.model.dof_damping[self.sim.model.joint_name2id('Joint_1_R')] = self.actual_max_friction * self.prev_stiffness_limit 
        # self.sim.model.dof_damping[self.sim.model.joint_name2id('Joint_2_L')] = self.actual_max_friction2 * self.prev_stiffness_limit 
        # self.sim.model.dof_damping[self.sim.model.joint_name2id('Joint_2_R')] = self.actual_max_friction2 * self.prev_stiffness_limit 
        
        # reset forces
        self.prev_force = 0.0
        self.prev_lforce = 0.0
        self.prev_rforce = 0.0
        self.prev_oforce = 0.0
        self.actual_force = 0.0
        self.est_grasping_force = 0.0
        self.est_grasping_force_R = 0.0
        self.est_grasping_force_L = 0.0
        
        # reset flags
        self.grasped_flag = 0
        self.location_flag_l = 0
        self.location_flag_r = 0
        self.location_flag = 0
        self.obj_fallen = 0
        self.update_bool = False
        
        # minimum grip force
        self.min_grip = 9.81*(self.obj_weight * 1.5)/0.5
        self.sim.model.geom_size[self.sim.model.geom_name2id('object_bottom'),2] = 0.05
        self.sim.model.body_mass[self.sim.model.body_name2id('object')] = self.obj_weight/2.0
        self.sim.model.body_mass[self.sim.model.body_name2id('object2')] = self.obj_weight
        
        # reset values
        self.vel_R = np.zeros([3,2,1])
        self.vel_L = np.zeros([3,2,1])
        
        self.sim.forward()
        return True

    def _sample_goal(self):
        goal = np.array([0.2, 1e-4, 0.0])
        
        initial_qpos = self.sim.data.get_joint_qpos('object:joint').copy()
        initial_pos, initial_quat = initial_qpos[:3], initial_qpos[3:]
        initial_pos[:2] = np.array([-(np.random.random_sample()*0.03 + 0.09) + 0.1, np.random.random_sample()*0.07+0.02 - 0.0635])
        initial_quat = ToQuaternion(np.random.random_sample()*np.pi/4-np.pi/8, 0, 0)
        initial_qpos[:3] = initial_pos
        
        self.mocap_offset = (np.random.random((2,1))-0.5) * np.pi/18.0 # domain randomization
        
        self.sim.data.set_joint_qpos('object:joint', initial_qpos)
        return goal

    def _is_success(self, achieved_goal, desired_goal):
        try: 
            intention = np.linalg.norm(achieved_goal[:,:self.est_dim] > desired_goal[:,:self.est_dim], axis=-1)
            torque = np.linalg.norm((achieved_goal[:,self.est_dim:self.est_dim+self.force_dim] > desired_goal[:,self.est_dim:self.est_dim+self.force_dim]), axis=-1)
            velocity = np.linalg.norm((achieved_goal[:,self.est_dim+self.force_dim:self.est_dim+self.force_dim+self.vel_dim] - desired_goal[:,self.est_dim+self.force_dim:self.est_dim+self.force_dim+self.vel_dim]), axis=-1)
        except: 
            intention = np.linalg.norm(achieved_goal[:self.est_dim] > desired_goal[:self.est_dim], axis=-1)
            torque = np.linalg.norm((achieved_goal[self.est_dim:self.est_dim+self.force_dim] > desired_goal[self.est_dim:self.est_dim+self.force_dim]), axis=-1)
            velocity = np.linalg.norm((achieved_goal[self.est_dim+self.force_dim:self.est_dim+self.force_dim+self.vel_dim] - desired_goal[self.est_dim+self.force_dim:self.est_dim+self.force_dim+self.vel_dim]), axis=-1)

        return (intention * (velocity < self.vel_threshold) * torque).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        
        utils.reset_mocap_welds(self.sim)
                
        self.sim.forward()
        
        self.sim.data.set_mocap_pos('mocap_r', np.array([0.1 - (L1 * np.sin(1*np.pi/4.0) + L2 * np.sin(1*np.pi/4.0)), L1 * np.cos(1*np.pi/4.0) + L2 * np.cos(1*np.pi/4.0) + 0.0635, 0.056965]))
        self.sim.data.set_mocap_pos('mocap_l', np.array([0.1 - (L1 * np.sin(3*np.pi/4.0) + L2 * np.sin(3*np.pi/4.0)), L1 * np.cos(3*np.pi/4.0) + L2 * np.cos(3*np.pi/4.0) - 0.0635, 0.056965]))
        for _ in range(10):
            self.sim.step()

    def render(self, mode='human', width=500, height=500):
        return super(CheolFingersEnv, self).render(mode, width, height)
