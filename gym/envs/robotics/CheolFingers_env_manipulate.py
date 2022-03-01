import numpy as np

from gym.envs.robotics import rotations, robot_env, utils

R_j = np.matrix([[0.01575,0],
                  [-0.01575, -0.01575]])
R_j_inv = np.linalg.inv(R_j)
R_j_L = np.matrix([[0.01575,0],
                  [0.01575, 0.01575]])
R_j_inv_L = np.linalg.inv(R_j_L)
R_e = np.matrix([[0.0034597,0],
                  [0, 0.0034597]])
L1 = 0.1
L2 = 0.075

m1 = 0.1
m2 = 0.05

# Ksc = 700

Rm = 0.0285

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
        self.max_stiffness = 1.
        self.prev_stiffness = 1.
        self.prev_stiffness_limit = 1.
        self.actual_max_stiffness = 1.13715e-0
        self.actual_max_friction = 2.e-2
        self.object_fragility = 4.5
        self.min_grip = 0.0
        self.fric_mu = 0.7
        self.grav_const = 9.81
        self.prev_force = 0.0
        self.prev_lforce = 0.0
        self.prev_rforce = 0.0
        self.prev_oforce = 0.0
        self.previous_input = 0
        self.remaining_timestep = 75
        self.des_Fp_R = np.array([[0.0],[0.0]])
        self.des_Fp_L = np.array([[0.0],[0.0]])
        self.des_Fp = np.array([[0.0],[0.0],[0.0],[0.0]])
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
        self.lower_limit = np.array([[-0.15],[0.02],[0.09]])
        self.upper_limit = np.array([[0.40],[0.09],[0.14]])
        self.pert_type = pert_type
        self.n_actions = n_actions
        self.eval_env = eval_env
        self.est_dim = 3
        self.force_dim = 2
        self.vel_dim = 2
        self.flag_dim = 1
        self.vel_R = np.zeros([3,2,1])
        self.vel_L = np.zeros([3,2,1])
        self.des_tau = np.zeros([4,1])
        self.est_obj_pose_init = np.zeros([2,1])
        self.est_obj_pose = np.zeros([2,1])
        self.est_grasping_force = 0.0
        self.est_grasping_force_L = 0.0
        self.est_grasping_force_R = 0.0
        self.est_torques_R = np.zeros([2,1])
        self.est_torques_L = np.zeros([2,1])
        self.max_ext_torques_L = 0.0
        self.max_ext_torques_R = 0.0
        self.max_vel_L = 0.0
        self.max_vel_R = 0.0
        self.des_mR = np.zeros([2,1])
        self.des_mL = np.zeros([2,1])
        self.update_bool = False

        super(CheolFingersEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, initial_qpos=initial_qpos, n_actions=n_actions)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        try: 
            est_dist = goal_distance(achieved_goal[:,:self.est_dim], goal[:,:self.est_dim])
            force_rew = np.linalg.norm((achieved_goal[:,self.est_dim:self.est_dim+self.force_dim] - goal[:,self.est_dim:self.est_dim+self.force_dim]), axis=-1)
            vel_rew = np.linalg.norm((achieved_goal[:,self.est_dim+self.force_dim:self.est_dim+self.force_dim+self.vel_dim] - goal[:,self.est_dim+self.force_dim:self.est_dim+self.force_dim+self.vel_dim]), axis=-1)
            # flag_rew = np.linalg.norm((achieved_goal[:,self.est_dim+self.force_dim+self.vel_dim:self.est_dim+self.force_dim+self.vel_dim+self.flag_dim] - goal[:,self.est_dim+self.force_dim+self.vel_dim:self.est_dim+self.force_dim+self.vel_dim+self.flag_dim]), axis=-1)
        except: 
            est_dist = goal_distance(achieved_goal[:self.est_dim], goal[:self.est_dim])
            force_rew = np.linalg.norm((achieved_goal[self.est_dim:self.est_dim+self.force_dim] - goal[self.est_dim:self.est_dim+self.force_dim]), axis=-1)
            vel_rew = np.linalg.norm((achieved_goal[self.est_dim+self.force_dim:self.est_dim+self.force_dim+self.vel_dim] - goal[self.est_dim+self.force_dim:self.est_dim+self.force_dim+self.vel_dim]), axis=-1)
            # flag_rew = np.linalg.norm((achieved_goal[self.est_dim+self.force_dim+self.vel_dim:self.est_dim+self.force_dim+self.vel_dim+self.flag_dim] - goal[self.est_dim+self.force_dim+self.vel_dim:self.est_dim+self.force_dim+self.vel_dim+self.flag_dim]), axis=-1)
        # print(achieved_goal[1:self.est_dim])
        # print(goal[1:self.est_dim])
        # print((est_dist > self.distance_threshold).astype(np.float32))
        # print((est_angle > 0.17).astype(np.float32))
        # print(((est_dist > self.distance_threshold).astype(np.float32) + (est_angle > 0.17).astype(np.float32) > 0).astype(np.float32))
        # print(est_vel.astype(np.float32))
        return -(est_dist > self.distance_threshold).astype(np.float32) - 4.*force_rew - 1.*vel_rew #- 1.*flag_rew
        

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        self.sim.forward()

    def _set_action(self, action):
        action = action.copy()  # ensure that we don't change the action outside of this scope
        
        pos_ctrl = action[:3].copy()
        
        change_l = pos_ctrl[0]/50.0
        change_x = pos_ctrl[1]/50.0
        change_y = pos_ctrl[2]/50.0
        
        stiffness_ctrl = 0.0
        stiffness_limit = 0.0
        
        if action.shape[0] > 3:
            stiffness_limit = 0.2 * self.max_stiffness * action[4]
            
            self.prev_stiffness_limit += stiffness_limit
            self.prev_stiffness_limit = np.max([np.min([self.prev_stiffness_limit, self.max_stiffness]), self.max_stiffness / 10.0])
            self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('AJ1_R'), 0] = self.actual_max_stiffness * self.prev_stiffness_limit 
            self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('AJ1_R'), 1] = -self.actual_max_stiffness * self.prev_stiffness_limit 
            self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('AJ2_R'), 0] = self.actual_max_stiffness * self.prev_stiffness_limit 
            self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('AJ2_R'), 1] = -self.actual_max_stiffness * self.prev_stiffness_limit 
            self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('AJ1_L'), 0] = self.actual_max_stiffness * self.prev_stiffness_limit 
            self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('AJ1_L'), 1] = -self.actual_max_stiffness * self.prev_stiffness_limit 
            self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('AJ2_L'), 0] = self.actual_max_stiffness * self.prev_stiffness_limit 
            self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('AJ2_L'), 1] = -self.actual_max_stiffness * self.prev_stiffness_limit 
            self.sim.model.dof_damping[self.sim.model.joint_name2id('Joint_1_L')] = self.actual_max_friction * self.prev_stiffness_limit
            self.sim.model.dof_damping[self.sim.model.joint_name2id('Joint_1_R')] = self.actual_max_friction * self.prev_stiffness_limit
            self.sim.model.dof_damping[self.sim.model.joint_name2id('Joint_2_L')] = self.actual_max_friction * self.prev_stiffness_limit
            self.sim.model.dof_damping[self.sim.model.joint_name2id('Joint_2_R')] = self.actual_max_friction * self.prev_stiffness_limit
            
            stiffness_ctrl = 0.2 * self.max_stiffness * action[3]
            
            self.prev_stiffness += stiffness_ctrl
            self.prev_stiffness = np.max([np.min([self.prev_stiffness, self.prev_stiffness_limit]), 0.0])
            # print(self.prev_stiffness_limit)
        
        self.des_p = self.des_p + np.array([[change_l],[change_x],[change_y]])
        self.des_p = np.clip(self.des_p, self.lower_limit, self.upper_limit)
        
        Jp = np.matrix([[-self.Prel[0]/self.p[0,0], -self.Prel[1]/self.p[0,0], self.Prel[0]/self.p[0,0], self.Prel[1]/self.p[0,0]],
                          [0.5, 0, 0.5, 0],
                          [0, 0.5, 0, 0.5]])
        a, b = Jp[0,2], Jp[0,3]
        c = a**2 + b**2
        Jp_inv = np.matrix([[-0.5*a/c, 1, 0],
                            [-0.5*b/c, 0, 1],
                            [0.5*a/c, 1, 0],
                            [0.5*b/c, 0, 1],])
        
        J = np.matrix([[-self.P_L[1], -L2 * np.sin(self.Lj[0,0]+self.Lj[1,0] + 3*np.pi/4.0), 0, 0], 
                         [self.P_L[0], L2 * np.cos(self.Lj[0,0]+self.Lj[1,0] + 3*np.pi/4.0), 0, 0],
                         [0, 0, -self.P_R[1], -L2 * np.sin(self.Rj[0,0]-self.Rj[1,0] + np.pi/4.0)], 
                         [0, 0, self.P_R[0], L2 * np.cos(self.Rj[0,0]-self.Rj[1,0] + np.pi/4.0)]])
        J_inv_L = np.matrix([[J[1,1], -J[0,1]], [-J[1,0], J[0,0]]]) / (J[0,0]*J[1,1] - J[0,1]*J[1,0])
        J_inv_R = np.matrix([[J[3,3], -J[2,3]], [-J[3,2], J[2,2]]]) / (J[2,2]*J[3,3] - J[2,3]*J[3,2])
        
        # tendon space stiffness
        Ksc = np.matrix([[self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('AJ1_R'), 0], 0],[0, self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('AJ2_R'), 0]]])/Rm/Rm
        Ksc_L = np.matrix([[self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('AJ1_L'), 0], 0],[0, self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('AJ2_L'), 0]]])/Rm/Rm
        
        # joint space stiffness
        max_kj_R = np.transpose(R_j) * Ksc * R_j
        max_kj_L = np.transpose(R_j_L) * Ksc_L * R_j_L
        
        # Cartesian space stiffness
        max_k_R = np.transpose(J_inv_R) * max_kj_R * J_inv_R
        max_k_L = np.transpose(J_inv_L) * max_kj_L * J_inv_L
        max_k = np.matrix([[max_k_L[0,0], max_k_L[0,1], 0, 0],
                           [max_k_L[1,0], max_k_L[1,1], 0, 0],
                           [0, 0, max_k_R[0,0], max_k_R[0,1]],
                           [0, 0, max_k_R[1,0], max_k_R[1,1]]])
        
        # task space stiffness
        max_kp = np.transpose(Jp_inv) * max_k * Jp_inv
        
        max_kp_diag = np.matrix([[max_kp[0,0], 0, 0], [0, max_kp[1,1], 0], [0, 0, max_kp[2,2]]])
        max_kj_diag = np.matrix([[max_kj_L[0,0], 0, 0, 0], [0, max_kj_L[1,1], 0, 0], [0, 0, max_kj_R[0,0], 0], [0, 0, 0, max_kj_R[1,1]]])
        
        # desired task space force
        des_Fp = 0.8 * max_kp_diag * (self.prev_stiffness * (self.des_p - self.p))
        
        # desired Cartesian space force
        des_F = np.transpose(Jp) * des_Fp
        if action[0]*1e6 == 8.0: 
            self.update_bool = True
        elif np.linalg.norm(action) > 1.39e-5:
            self.max_ext_torques_L = 0.0
            self.max_ext_torques_R = 0.0
            self.max_vel_L = 0.0
            self.max_vel_R = 0.0
            
        # desired joint space torque
        self.des_tau = np.transpose(J) * des_F
        
        self.des_mL = ((np.matrix([[1/Ksc_L[0,0], 0],[0, 1/Ksc_L[1,1]]]) * np.transpose(R_j_inv_L)*self.des_tau[0:2]) + R_j_L * self.joint_pos[0:2]) / Rm
        self.des_mR = ((np.matrix([[1/Ksc[0,0], 0],[0, 1/Ksc[1,1]]]) * np.transpose(R_j_inv)*self.des_tau[2:4]) + R_j * self.joint_pos[2:4]) / Rm 
        
        prob = 0.005 if self.pert_type == 'delay' else -0.1
        if np.random.random() > prob:
            self.sim.data.ctrl[0] = self.des_mL[0,0]
            self.sim.data.ctrl[1] = self.des_mL[1,0]
            self.sim.data.ctrl[2] = self.des_mR[0,0]
            self.sim.data.ctrl[3] = self.des_mR[1,0]
            self.previous_input = self.sim.data.ctrl
        else:
            try: self.sim.data.ctrl = self.previous_input
            except: 
                self.sim.data.ctrl[0] = 0.07813083
                self.sim.data.ctrl[1] = -0.99159258
                self.sim.data.ctrl[2] = -0.07813083
                self.sim.data.ctrl[3] = -0.99159258      

    def _get_obs(self):
        Ksc = np.matrix([[self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('AJ1_R'), 0], 0],[0, self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('AJ2_R'), 0]]])/Rm/Rm
        Ksc_L = np.matrix([[self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('AJ1_L'), 0], 0],[0, self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('AJ2_L'), 0]]])/Rm/Rm
        # update only when robot can observe
        if self.update_bool == True: 
            self.update_bool = False
            
            self.Rj = np.array([[self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Joint_1_R')]]],
                       [self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Joint_2_R')]]]])
            self.Lj = np.array([[self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Joint_1_L')]]],
                           [self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Joint_2_L')]]]])
            
            self.joint_pos = np.concatenate([self.Lj, self.Rj])
            self.joint_pos[3,0] = -self.joint_pos[3,0]
            
            self.vel_R[2] = self.vel_R[1]
            self.vel_L[2] = self.vel_L[1]
            self.vel_R[1] = self.vel_R[0]
            self.vel_L[1] = self.vel_L[0]
            self.vel_R[0] = (self.Rj - self.Prev_Rj)/2
            self.vel_L[0] = (self.Lj - self.Prev_Lj)/2
            self.vel_R[0,1] = -self.vel_R[0,1]
            
            xR = L1 * np.cos(self.Rj[0,0] + np.pi/4.0) + L2 * np.cos(self.Rj[0,0]-self.Rj[1,0] + np.pi/4.0)
            yR = L1 * np.sin(self.Rj[0,0] + np.pi/4.0) + L2 * np.sin(self.Rj[0,0]-self.Rj[1,0] + np.pi/4.0)
            xL = L1 * np.cos(self.Lj[0,0] + 3*np.pi/4.0) + L2 * np.cos(self.Lj[0,0]+self.Lj[1,0] + 3*np.pi/4.0)
            yL = L1 * np.sin(self.Lj[0,0] + 3*np.pi/4.0) + L2 * np.sin(self.Lj[0,0]+self.Lj[1,0] + 3*np.pi/4.0)
            
            self.P_R = np.array([xR, yR])
            self.P_L = np.array([xL, yL])
            
            self.Prel = self.P_R - self.P_L + np.array([0.127, 0])
            l = np.sqrt(self.Prel[0]*self.Prel[0] + self.Prel[1]*self.Prel[1])
            self.p = np.array([[l],[(xR+xL+0.127)/2],[(yR+yL)/2]])
            self.vel_p = self.p - self.Prev_p
            
            self.est_torques_R = np.transpose(R_j) * Ksc * (self.des_mR * Rm - R_j * self.joint_pos[2:4])
            self.est_torques_L = np.transpose(R_j_L) * Ksc_L * (self.des_mL * Rm - R_j_L * self.joint_pos[0:2])
            
            self.Prev_Rj = self.Rj
            self.Prev_Lj = self.Lj
            self.Prev_p = self.p
            
            if self.max_ext_torques_L > self.est_torques_L[0,0]: 
                self.max_ext_torques_L = np.tanh(self.est_torques_L[0,0])
            if self.max_ext_torques_R < self.est_torques_R[0,0]: 
                self.max_ext_torques_R = np.tanh(self.est_torques_R[0,0])
            if self.max_vel_L < np.linalg.norm(self.vel_L[0]): 
                self.max_vel_L = np.tanh(np.linalg.norm(self.vel_L[0]))
            if self.max_vel_R < np.linalg.norm(self.vel_R[0]): 
                self.max_vel_R = np.tanh(np.linalg.norm(self.vel_R[0]))
        else:
            # positions
            unobserved_Rj = np.array([[self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Joint_1_R')]]],
                           [self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Joint_2_R')]]]])
            unobserved_Lj = np.array([[self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Joint_1_L')]]],
                           [self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Joint_2_L')]]]])
            
            unobserved_joint_pos = np.concatenate([unobserved_Lj, unobserved_Rj])
            unobserved_joint_pos[3,0] = -unobserved_joint_pos[3,0]
            
            unobserved_est_torques_R = np.transpose(R_j) * Ksc * (self.des_mR * Rm - R_j * unobserved_joint_pos[2:4])
            unobserved_est_torques_L = np.transpose(R_j_L) * Ksc_L * (self.des_mL * Rm - R_j_L * unobserved_joint_pos[0:2])
            unobserved_est_grasping_force_R = unobserved_est_torques_R[0,0]
            unobserved_est_grasping_force_L = unobserved_est_torques_L[0,0]
            
            unobserved_vel_R = (unobserved_Rj - self.Prev_Rj)/2
            unobserved_vel_L = (unobserved_Lj - self.Prev_Lj)/2
            unobserved_vel_R[1] = -unobserved_vel_R[1]
            
            if self.max_ext_torques_L > unobserved_est_grasping_force_L: 
                self.max_ext_torques_L = np.tanh(unobserved_est_grasping_force_L)
            if self.max_ext_torques_R < unobserved_est_grasping_force_R: 
                self.max_ext_torques_R = np.tanh(unobserved_est_grasping_force_R)
            if self.max_vel_L < np.linalg.norm(unobserved_vel_L): 
                self.max_vel_L = np.tanh(np.linalg.norm(unobserved_vel_L))
            if self.max_vel_R < np.linalg.norm(unobserved_vel_R): 
                self.max_vel_R = np.tanh(np.linalg.norm(unobserved_vel_R))
                
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        l_finger_force = self.prev_lforce + (self.sim.data.sensordata[self.sim.model.sensor_name2id('fingertip_l')] - self.prev_lforce) * dt / 0.5
        r_finger_force = self.prev_rforce + (self.sim.data.sensordata[self.sim.model.sensor_name2id('fingertip_r')] - self.prev_rforce) * dt / 0.5
        o_force = self.prev_oforce + (self.sim.data.sensordata[self.sim.model.sensor_name2id('object_frc')] - self.prev_oforce) * dt / 0.5
        
        self.sim.model.body_quat[self.sim.model.body_name2id('debug_body')] = ToQuaternion(0, -np.pi/2.0, np.pi/2.0)
        self.sim.model.body_pos[self.sim.model.body_name2id('debug_body'),0:2] = np.array([-self.p[2,0] + 0.0873, self.p[1,0] - 0.0635])
        self.sim.model.site_size[self.sim.model.site_name2id('debug'),1] = self.p[0,0]/2.0
        
        # if self.pert_type != 'none' and self.pert_type != 'meas':
        # if self.grasped_flag == 0.1: self.sim.data.qvel[self.sim.model.joint_name2id('object:joint')+1] += 0.5*(np.random.random()-0.5)
        
        object_pos = self.sim.data.get_body_xpos('object2').copy()
        object_pos_frame = np.array([[object_pos[1]+0.0635], [-(object_pos[0]-0.0873)], [object_pos[2]]])
        object_frc = self.sim.data.sensordata[self.sim.model.sensor_name2id('object_frc')]
        object_ori = ToRPY(self.sim.data.body_xquat[self.sim.model.body_name2id('object2')])[2]
            
        self.prev_lforce = l_finger_force
        self.prev_rforce = r_finger_force
        self.prev_oforce = o_force
        # Is object within grasp?
        if self.eval_env == True:
            if self.est_torques_R[0,0] - self.est_torques_L[0,0] > 0.14 and np.linalg.norm(self.vel_p[0,0]) < 1e-4 and self.p[0,0] > 0.025:
                self.est_obj_pose = self.p[1:3].copy()
                # self.sim.data.qvel[self.sim.model.joint_name2id('object:joint')+1] += 0.5*(np.random.random()-0.5)
        else:
            if self.p[0,0] > 0.025 and object_frc > self.min_grip and l_finger_force > 0.0 and r_finger_force > 0.0:# and l_finger_force * r_finger_force == 0.):
                self.est_obj_pose = self.p[1:3].copy()
                # self.sim.data.qvel[self.sim.model.joint_name2id('object:joint')+1] += 0.5*(np.random.random()-0.5)
        
        if self.p[0,0] > 0.025 and object_frc > self.min_grip and l_finger_force > 0.0 and r_finger_force > 0.0:
             self.sim.model.geom_size[self.sim.model.geom_name2id('object_bottom'),2] = 0.005
        else:
             self.sim.model.geom_size[self.sim.model.geom_name2id('object_bottom'),2] = 0.05
        
        if self.n_actions == 5:
            observation = np.array([self.p[0,0], self.p[1,0], self.p[2,0], # l, theta, cen_x, cen_y
                                    self.des_p[0,0]-self.p[0,0], self.des_p[1,0]-self.p[1,0], self.des_p[2,0]-self.p[2,0],
                                    self.goal[0]-self.est_obj_pose[0,0], self.goal[1]-self.est_obj_pose[1,0],
                                    self.est_obj_pose[0,0]-self.p[1,0], self.est_obj_pose[1,0]-self.p[2,0],
                                    self.est_torques_L[0,0], self.est_torques_R[0,0],
                                    self.vel_L[0,0,0], self.vel_L[0,1,0],self.vel_R[0,0,0], self.vel_R[0,1,0],
                                    self.prev_stiffness, self.prev_stiffness_limit
                                    ])
        else:
            observation = np.array([self.p[0,0], self.p[1,0], self.p[2,0],  # l, theta, cen_x, cen_y
                                    self.des_p[0,0]-self.p[0,0], self.des_p[1,0]-self.p[1,0], self.des_p[2,0]-self.p[2,0],
                                    self.goal[0]-self.est_obj_pose[0,0], self.goal[1]-self.est_obj_pose[1,0],
                                    self.est_obj_pose[0,0]-self.p[1,0], self.est_obj_pose[1,0]-self.p[2,0],
                                    self.est_torques_L[0,0], self.est_torques_R[0,0],
                                    self.vel_L[0,0,0], self.vel_L[0,1,0],self.vel_R[0,0,0], self.vel_R[0,1,0],
                                   ])
        
        modified_obs = dict(observation=observation, achieved_goal=np.array([object_pos_frame[0,0], object_pos_frame[1,0], object_pos_frame[2,0], self.max_ext_torques_L, self.max_ext_torques_R, self.max_vel_L, self.max_vel_R]), desired_goal = self.goal)
        return modified_obs

    def _viewer_setup(self):
        self.viewer.cam.distance = 0.5
        self.viewer.cam.azimuth = 32.
        self.viewer.cam.elevation = -30.

    def _render_callback(self):
        # Visualize target.
        self.sim.model.body_pos[self.sim.model.body_name2id('target_body'), 0:2] = np.array([-self.goal[1] + 0.0873, self.goal[0] - 0.0635])
        self.sim.model.body_quat[self.sim.model.body_name2id('target_body')] = ToQuaternion(0, -np.pi/2.0, 0)
        
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        
        # reset the broken objects
        self.broken_object = False
        
        self.Rj = np.array([[self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Joint_1_R')]]],
                   [self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Joint_2_R')]]]])
        self.Lj = np.array([[self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Joint_1_L')]]],
                       [self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Joint_2_L')]]]])
        
        self.P_R = np.array([L1 * np.cos(self.Rj[0,0] + 1*np.pi/4.0) + L2 * np.cos(self.Rj[0,0]-self.Rj[1,0] + 1*np.pi/4.0), L1 * np.sin(self.Rj[0,0] + 1*np.pi/4.0) + L2 * np.sin(self.Rj[0,0]-self.Rj[1,0] + 1*np.pi/4.0)])
        self.P_L = np.array([L1 * np.cos(self.Lj[0,0] + 3*np.pi/4.0) + L2 * np.cos(self.Lj[0,0]+self.Lj[1,0] + 3*np.pi/4.0), L1 * np.sin(self.Lj[0,0] + 3*np.pi/4.0) + L2 * np.sin(self.Lj[0,0]+self.Lj[1,0] + 3*np.pi/4.0)])
        
        self.Prel = self.P_R - self.P_L + np.array([0.127, 0])
        self.p = np.array([[0.16],[0.0635],[0.1]])
        self.des_p = np.array([[0.16],[0.0635],[0.1]])
        # print(self.p)
        
        self.Prev_p = self.p.copy()
        self.vel_p = self.p - self.Prev_p
        self.Pc = np.array([(self.P_R[0]+self.P_L[0]+0.127)/2,(self.P_R[1]+self.P_L[1])/2])
        
        self.joint_pos = np.concatenate([self.Lj, self.Rj])
        self.joint_pos[3,0] = -self.joint_pos[3,0]
        self.est_obj_pose = np.zeros([2,1])
        self.est_grasping_force = 0.0
        self.max_ext_torques_L = 0.0
        self.max_ext_torques_R = 0.0
        self.max_vel_L = 0.0
        self.max_vel_R = 0.0
        
        # reset stiffness
        self.prev_stiffness = 1.
        self.prev_stiffness_limit = 1.
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('AJ1_R'), 0] = self.actual_max_stiffness * self.prev_stiffness_limit 
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('AJ1_R'), 1] = -self.actual_max_stiffness * self.prev_stiffness_limit 
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('AJ2_R'), 0] = self.actual_max_stiffness * self.prev_stiffness_limit 
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('AJ2_R'), 1] = -self.actual_max_stiffness * self.prev_stiffness_limit 
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('AJ1_L'), 0] = self.actual_max_stiffness * self.prev_stiffness_limit 
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('AJ1_L'), 1] = -self.actual_max_stiffness * self.prev_stiffness_limit 
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('AJ2_L'), 0] = self.actual_max_stiffness * self.prev_stiffness_limit 
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('AJ2_L'), 1] = -self.actual_max_stiffness * self.prev_stiffness_limit 
        self.sim.model.dof_damping[self.sim.model.joint_name2id('Joint_1_L')] = self.actual_max_friction * self.prev_stiffness_limit 
        self.sim.model.dof_damping[self.sim.model.joint_name2id('Joint_1_R')] = self.actual_max_friction * self.prev_stiffness_limit 
        self.sim.model.dof_damping[self.sim.model.joint_name2id('Joint_2_L')] = self.actual_max_friction * self.prev_stiffness_limit 
        self.sim.model.dof_damping[self.sim.model.joint_name2id('Joint_2_R')] = self.actual_max_friction * self.prev_stiffness_limit 
        
        # reset forces
        self.prev_lforce = 0.0
        self.prev_rforce = 0.0
        self.prev_oforce = 0.0
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
        self.min_grip = 9.81*self.sim.model.body_mass[self.sim.model.body_name2id('object2')]/0.5
        self.sim.model.geom_size[self.sim.model.geom_name2id('object_bottom'),2] = 0.05
        self.sim.model.body_mass[self.sim.model.body_name2id('object')] = 1e-3
        
        # reset values
        self.vel_R = np.zeros([3,2,1])
        self.vel_L = np.zeros([3,2,1])
        
        self.sim.data.ctrl[0] = 0.07813083
        self.sim.data.ctrl[1] = -0.99159258
        self.sim.data.ctrl[2] = -0.07813083
        self.sim.data.ctrl[3] = -0.99159258
        
        self.sim.forward()
        return True

    def _sample_goal(self):
        goal = np.array([np.random.random_sample()*0.07+0.02, np.random.random_sample()*0.04+0.1, 0.05, 0.0, 0.0, 0.0, 0.0])
        
        initial_qpos = self.sim.data.get_joint_qpos('object:joint').copy()
        initial_pos, initial_quat = initial_qpos[:3], initial_qpos[3:]
        initial_pos[:2] = np.array([-(np.random.random_sample()*0.02 + 0.075) + 0.0873, np.random.random_sample()*0.07+0.02 - 0.0635])
        initial_quat = ToQuaternion(np.random.random_sample()*np.pi/4-np.pi/8, 0, 0)
        initial_qpos[:3] = initial_pos
        # initial_qpos[3:] = initial_quat
        
        self.sim.data.set_joint_qpos('object:joint', initial_qpos)
        self.est_obj_pose = np.array([[initial_qpos[1]+0.0635], [-(initial_qpos[0]-0.0873)]])
        # self.est_obj_pose_init = np.array([[-0.5], [-0.5]])
        
        self.est_obj_pose += np.array([[(np.random.random_sample()-0.5)*0.02], [(np.random.random_sample()-0.5)*0.02]])
        # self.sim.model.geom_size[self.sim.model.geom_name2id('object_top'),1] = (np.random.random_sample())*0.005 + 0.01
        # self.sim.model.site_size[self.sim.model.site_name2id('object_site'),1] = self.sim.model.geom_size[self.sim.model.geom_name2id('object_top'),0] + 0.005
        # self.sim.model.body_mass[self.sim.model.body_name2id('object2')] = np.random.random_sample()*0.05 + 0.1
        ## domain randomization
        # self.sim.model.body_pos[self.sim.model.body_name2id('Sensor_base')][0] = -0.0327 + 0.01 * (np.random.random()- 0.5)
        # self.sim.model.body_pos[self.sim.model.body_name2id('target_body')][0] = self.sim.model.body_pos[self.sim.model.body_name2id('Sensor_base')][0]
        # self.sim.model.geom_size[self.sim.model.geom_name2id('Fake_object_geom')][1] = 0.02 + (np.random.random()-0.5)*0.01
        # self.sim.model.site_size[self.sim.model.site_name2id('object_force')][1] =self.sim.model.geom_size[self.sim.model.geom_name2id('Fake_object_geom')][1] + 0.001
        # self.sim.model.tendon_stiffness[self.sim.model.tendon_name2id('T0_L')] = (np.random.random())*1e2
        # self.sim.model.tendon_stiffness[self.sim.model.tendon_name2id('T0_R')] = self.sim.model.tendon_stiffness[self.sim.model.tendon_name2id('T0_L')]
        
        self.sim.data.ctrl[0] = 0.07813083
        self.sim.data.ctrl[1] = -0.99159258
        self.sim.data.ctrl[2] = -0.07813083
        self.sim.data.ctrl[3] = -0.99159258
        # return np.concatenate([goal.copy(), np.concatenate([np.zeros(self.velest_dim), [0.0, 0.0]])])
        return goal

    def _is_success(self, achieved_goal, desired_goal):
        try: 
            est_dist = goal_distance(achieved_goal[:,:self.est_dim], desired_goal[:,:self.est_dim])
        except: 
            est_dist = goal_distance(achieved_goal[:self.est_dim], desired_goal[:self.est_dim])
        return (est_dist < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.data.ctrl[0] = 0.07813083
        self.sim.data.ctrl[1] = -0.99159258
        self.sim.data.ctrl[2] = -0.07813083
        self.sim.data.ctrl[3] = -0.99159258
                
        self.sim.forward()
        for _ in range(10):
            self.sim.step()

    def render(self, mode='human', width=500, height=500):
        return super(CheolFingersEnv, self).render(mode, width, height)
