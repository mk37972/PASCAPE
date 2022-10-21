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
        distance_threshold, initial_qpos, reward_type, pert_type='none', n_actions=4, eval_env=False
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
        self.max_stiffness = 0.2
        self.min_stiffness = 0.1
        self.prev_stiffness = 0.2
        self.prev_stiffness_limit = 0.2
        self.actual_max_stiffness = 2.30399e-2
        self.actual_max_stiffness2 = 1.15199e-2
        self.actual_stiffness = self.actual_max_stiffness
        self.actual_stiffness2 = self.actual_max_stiffness2
        self.object_fragility = 100. # 150.0 without DR 
        self.min_grip = 0.0
        self.fric_mu = 0.7
        self.grav_const = 9.81
        self.prev_force = 0.0
        self.prev_lforce = 0.0
        self.prev_rforce = 0.0
        self.prev_oforce = 0.0
        self.previous_input = 0
        
        self.des_Fp = np.array([[0.0],[0.0],[0.0],[0.0]])
        self.Rj = np.array([[initial_qpos['Joint_1_R']],[initial_qpos['Joint_2_R']]])
        self.Lj = np.array([[initial_qpos['Joint_1_L']],[initial_qpos['Joint_2_L']]])
        self.Prev_Rj = np.array([[initial_qpos['Joint_1_R']],[initial_qpos['Joint_2_R']]])
        self.Prev_Lj = np.array([[initial_qpos['Joint_1_L']],[initial_qpos['Joint_2_L']]])
        self.Pc = np.array([0., 0.])
        self.P_R = np.array([L1 * np.cos(self.Rj[0,0] + 3*np.pi/4.0) + L2 * np.cos(self.Rj[0,0]-self.Rj[1,0] + 3*np.pi/4.0), L1 * np.sin(self.Rj[0,0] + 3*np.pi/4.0) + L2 * np.sin(self.Rj[0,0]-self.Rj[1,0] + 3*np.pi/4.0)])
        self.P_L = np.array([L1 * np.cos(self.Lj[0,0] + 3*np.pi/4.0) + L2 * np.cos(self.Lj[0,0]+self.Lj[1,0] + 3*np.pi/4.0), L1 * np.sin(self.Lj[0,0] + 3*np.pi/4.0) + L2 * np.sin(self.Lj[0,0]+self.Lj[1,0] + 3*np.pi/4.0)])
        self.joint_pos = np.zeros([4,1])
        
        self.p = np.array([[0.],[0.],[0.],[0.]])
        self.des_p = np.array([[0.],[0.],[0.],[0.]])
        self.Prev_p = self.p
        self.vel_p = self.p - self.Prev_p
        self.des_Fp = np.array([[0.],[0.],[0.],[0.]])
        self.lower_limit = np.array([[-0.02],[0.0],[0.08],[-1.57]])
        self.upper_limit = np.array([[0.40],[0.125],[0.14],[1.57]])
        self.pert_type = pert_type
        self.n_actions = n_actions
        
        self.eval_env = eval_env
        self.est_dim = 4
        self.force_dim = 2
        self.vel_dim = 4
        
        self.vel_R = np.zeros([3,2,1])
        self.vel_L = np.zeros([3,2,1])
        self.des_tau = np.zeros([4,1])
        
        self.est_obj_pose = np.zeros([2,1])
        self.obj_weight = 2e-1
        self.mocap_offset = np.zeros(4)
        self.DR = True

        super(CheolFingersEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, initial_qpos=initial_qpos, n_actions=n_actions)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        try: 
            est_dist = goal_distance(achieved_goal[:,:self.est_dim], goal[:,:self.est_dim])
            force_rew = np.linalg.norm(achieved_goal[:,self.est_dim:self.est_dim+self.force_dim] - goal[:,self.est_dim:self.est_dim+self.force_dim], axis=-1)
            vel_rew = np.linalg.norm((achieved_goal[:,self.est_dim+self.force_dim:self.est_dim+self.force_dim+self.vel_dim] - goal[:,self.est_dim+self.force_dim:self.est_dim+self.force_dim+self.vel_dim]), axis=-1)
            # flag_rew = np.linalg.norm((achieved_goal[:,self.est_dim+self.force_dim+self.vel_dim:self.est_dim+self.force_dim+self.vel_dim+self.flag_dim] - goal[:,self.est_dim+self.force_dim+self.vel_dim:self.est_dim+self.force_dim+self.vel_dim+self.flag_dim]), axis=-1)
        except: 
            est_dist = goal_distance(achieved_goal[:self.est_dim], goal[:self.est_dim])
            force_rew = np.linalg.norm(achieved_goal[self.est_dim:self.est_dim+self.force_dim] - goal[self.est_dim:self.est_dim+self.force_dim], axis=-1)
            vel_rew = np.linalg.norm((achieved_goal[self.est_dim+self.force_dim:self.est_dim+self.force_dim+self.vel_dim] - goal[self.est_dim+self.force_dim:self.est_dim+self.force_dim+self.vel_dim]), axis=-1)
            # flag_rew = np.linalg.norm((achieved_goal[self.est_dim+self.force_dim+self.vel_dim:self.est_dim+self.force_dim+self.vel_dim+self.flag_dim] - goal[self.est_dim+self.force_dim+self.vel_dim:self.est_dim+self.force_dim+self.vel_dim+self.flag_dim]), axis=-1)
        # force reward = 2e-3 without DR 2e-5 with DR 
        return -(est_dist > self.distance_threshold).astype(np.float32) - 3e-3*force_rew - 1e-1*vel_rew #- 1.*flag_rew
        

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        self.sim.forward()

    def _set_action(self, action):
        action = action.copy()  # ensure that we don't change the action outside of this scope
        
        pos_ctrl = action[:4].copy()
        
        change_l = pos_ctrl[0]/50.0
        change_x = pos_ctrl[1]/50.0
        change_y = pos_ctrl[2]/50.0
        change_th = pos_ctrl[3]/5.0
        
        stiffness_ctrl = 0.0
        stiffness_limit = 0.0
        
        if action.shape[0] > 4:
            stiffness_limit = 0.2 * self.max_stiffness * action[5]
            
            self.prev_stiffness_limit += stiffness_limit
            self.prev_stiffness_limit = np.max([np.min([self.prev_stiffness_limit, self.max_stiffness]), self.min_stiffness])
            # self.actual_stiffness = self.actual_max_stiffness * self.prev_stiffness_limit
            # self.actual_stiffness2 = self.actual_max_stiffness2 * self.prev_stiffness_limit
            
            stiffness_ctrl = 0.2 * self.max_stiffness * action[4]
            
            self.prev_stiffness += stiffness_ctrl
            self.prev_stiffness = np.max([np.min([self.prev_stiffness, self.prev_stiffness_limit]), 0.0])
            
            # print(self.prev_stiffness_limit)
        
        # prob = 0.1/self.prev_stiffness_limit
        # if np.random.random() > prob:
        self.des_p = self.des_p + np.array([[change_l],[change_x],[change_y],[change_th]])
        self.des_p = np.clip(self.des_p, self.lower_limit, self.upper_limit)
        
        # calculating desired joint positions
        r = np.array([[self.prev_stiffness],[self.max_stiffness],[self.max_stiffness],[self.max_stiffness]])
        mocap_p = self.p + r * (self.des_p - self.p)
        x_l = mocap_p[1,0] - 0.5*mocap_p[0,0]*np.cos(mocap_p[3,0])
        y_l = mocap_p[2,0] - 0.5*mocap_p[0,0]*np.sin(mocap_p[3,0])
        # des_joint1_l = 2*np.arctan((2*L1*y - (L1**2*((- L1**2 + 2*L1*L2 - L2**2 + x_l**2 + y**2)*(L1**2 + 2*L1*L2 + L2**2 - x_l**2 - y**2))**(1/2))/(- L1**2 + 2*L1*L2 - L2**2 + x_l**2 + y**2) - (L2**2*((- L1**2 + 2*L1*L2 - L2**2 + x_l**2 + y**2)*(L1**2 + 2*L1*L2 + L2**2 - x_l**2 - y**2))**(1/2))/(- L1**2 + 2*L1*L2 - L2**2 + x_l**2 + y**2) + (x_l**2*((- L1**2 + 2*L1*L2 - L2**2 + x_l**2 + y**2)*(L1**2 + 2*L1*L2 + L2**2 - x_l**2 - y**2))**(1/2))/(- L1**2 + 2*L1*L2 - L2**2 + x_l**2 + y**2) + (y**2*((- L1**2 + 2*L1*L2 - L2**2 + x_l**2 + y**2)*(L1**2 + 2*L1*L2 + L2**2 - x_l**2 - y**2))**(1/2))/(- L1**2 + 2*L1*L2 - L2**2 + x_l**2 + y**2) + (2*L1*L2*((- L1**2 + 2*L1*L2 - L2**2 + x_l**2 + y**2)*(L1**2 + 2*L1*L2 + L2**2 - x_l**2 - y**2))**(1/2))/(- L1**2 + 2*L1*L2 - L2**2 + x_l**2 + y**2))/(L1**2 + 2*L1*x_l - L2**2 + x_l**2 + y**2)) - 3.*np.pi/4.
        # des_joint2_l = -2*np.arctan(((- L1**2 + 2*L1*L2 - L2**2 + x_l**2 + y**2)*(L1**2 + 2*L1*L2 + L2**2 - x_l**2 - y**2))**(1/2)/(- L1**2 + 2*L1*L2 - L2**2 + x_l**2 + y**2))
        
        x_r = mocap_p[1,0] + 0.5*mocap_p[0,0]*np.cos(mocap_p[3,0])
        y_r = mocap_p[2,0] + 0.5*mocap_p[0,0]*np.sin(mocap_p[3,0])
        # des_joint1_r = 2*np.arctan((2*L1*y - (L1**2*((- L1**2 + 2*L1*L2 - L2**2 + x_r**2 + y**2)*(L1**2 + 2*L1*L2 + L2**2 - x_r**2 - y**2))**(1/2))/(- L1**2 + 2*L1*L2 - L2**2 + x_r**2 + y**2) - (L2**2*((- L1**2 + 2*L1*L2 - L2**2 + x_r**2 + y**2)*(L1**2 + 2*L1*L2 + L2**2 - x_r**2 - y**2))**(1/2))/(- L1**2 + 2*L1*L2 - L2**2 + x_r**2 + y**2) + (x_r**2*((- L1**2 + 2*L1*L2 - L2**2 + x_r**2 + y**2)*(L1**2 + 2*L1*L2 + L2**2 - x_r**2 - y**2))**(1/2))/(- L1**2 + 2*L1*L2 - L2**2 + x_r**2 + y**2) + (y**2*((- L1**2 + 2*L1*L2 - L2**2 + x_r**2 + y**2)*(L1**2 + 2*L1*L2 + L2**2 - x_r**2 - y**2))**(1/2))/(- L1**2 + 2*L1*L2 - L2**2 + x_r**2 + y**2) + (2*L1*L2*((- L1**2 + 2*L1*L2 - L2**2 + x_r**2 + y**2)*(L1**2 + 2*L1*L2 + L2**2 - x_r**2 - y**2))**(1/2))/(- L1**2 + 2*L1*L2 - L2**2 + x_r**2 + y**2))/(L1**2 + 2*L1*x_r - L2**2 + x_r**2 + y**2)) - 1.*np.pi/4.
        # des_joint2_r = 2*np.arctan(((- L1**2 + 2*L1*L2 - L2**2 + x_r**2 + y**2)*(L1**2 + 2*L1*L2 + L2**2 - x_r**2 - y**2))**(1/2)/(- L1**2 + 2*L1*L2 - L2**2 + x_r**2 + y**2))
        
        # welding the bodies to the mocap
        new_pose_l = np.array([-y_l + 0.1 + self.mocap_offset[0], x_l - 0.0625 + self.mocap_offset[1]])
        new_pose_r = np.array([-y_r + 0.1 + self.mocap_offset[2], x_r - 0.0625 + self.mocap_offset[3]])
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
        
        Jp = np.matrix([[-self.Prel[0]/self.p[0,0], -self.Prel[1]/self.p[0,0], self.Prel[0]/self.p[0,0], self.Prel[1]/self.p[0,0]],
                          [self.Prel[1]/self.p[0,0]/self.p[0,0], -self.Prel[0]/self.p[0,0]/self.p[0,0], -self.Prel[1]/self.p[0,0]/self.p[0,0], self.Prel[0]/self.p[0,0]/self.p[0,0]],
                          [0.5, 0, 0.5, 0],
                          [0, 0.5, 0, 0.5]])
        Jp_det = (Jp[0,0]*Jp[1,1] - Jp[0,1]*Jp[1,0] - Jp[0,0]*Jp[1,3] + Jp[0,1]*Jp[1,2] - Jp[0,2]*Jp[1,1] + Jp[0,3]*Jp[1,0] + Jp[0,2]*Jp[1,3] - Jp[0,3]*Jp[1,2]);
        Jp_inv = np.matrix([[Jp[1,1] - Jp[1,3], Jp[0,3] - Jp[0,1], 2*Jp[0,1]*Jp[1,2] - 2*Jp[0,2]*Jp[1,1] + 2*Jp[0,2]*Jp[1,3] - 2*Jp[0,3]*Jp[1,2], 2*Jp[0,1]*Jp[1,3] - 2*Jp[0,3]*Jp[1,1]], 
                            [Jp[1,2] - Jp[1,0], Jp[0,0] - Jp[0,2], 2*Jp[0,2]*Jp[1,0] - 2*Jp[0,0]*Jp[1,2], 2*Jp[0,3]*Jp[1,0] - 2*Jp[0,0]*Jp[1,3] + 2*Jp[0,2]*Jp[1,3] - 2*Jp[0,3]*Jp[1,2]],
                            [Jp[1,3] - Jp[1,1], Jp[0,1] - Jp[0,3], 2*Jp[0,0]*Jp[1,1] - 2*Jp[0,1]*Jp[1,0] - 2*Jp[0,0]*Jp[1,3] + 2*Jp[0,3]*Jp[1,0], 2*Jp[0,3]*Jp[1,1] - 2*Jp[0,1]*Jp[1,3]],
                            [Jp[1,0] - Jp[1,2], Jp[0,2] - Jp[0,0], 2*Jp[0,0]*Jp[1,2] - 2*Jp[0,2]*Jp[1,0], 2*Jp[0,0]*Jp[1,1] - 2*Jp[0,1]*Jp[1,0] + 2*Jp[0,1]*Jp[1,2] - 2*Jp[0,2]*Jp[1,1]]])/Jp_det
        J = np.matrix([[-self.P_L[1], -L2 * np.sin(self.Lj[0,0]+self.Lj[1,0] + 3*np.pi/4.0), 0, 0], 
                         [self.P_L[0], L2 * np.cos(self.Lj[0,0]+self.Lj[1,0] + 3*np.pi/4.0), 0, 0],
                         [0, 0, -self.P_R[1], -L2 * np.sin(self.Rj[0,0]-self.Rj[1,0] + np.pi/4.0)], 
                         [0, 0, self.P_R[0], L2 * np.cos(self.Rj[0,0]-self.Rj[1,0] + np.pi/4.0)]])
        J_inv_L = np.matrix([[J[1,1], -J[0,1]], [-J[1,0], J[0,0]]]) / (J[0,0]*J[1,1] - J[0,1]*J[1,0])
        J_inv_R = np.matrix([[J[3,3], -J[2,3]], [-J[3,2], J[2,2]]]) / (J[2,2]*J[3,3] - J[2,3]*J[3,2])
        
        # tendon space stiffness
        Ksc = np.matrix([[self.actual_stiffness, 0],[0, self.actual_stiffness2]])/Rm/Rm
        Ksc_L = np.matrix([[self.actual_stiffness, 0],[0, self.actual_stiffness2]])/Rm/Rm
        
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
        
        max_kp_diag = np.matrix([[max_kp[0,0], 0, 0, 0], [0, max_kp[1,1], 0, 0], [0, 0, max_kp[2,2], 0], [0, 0, 0, max_kp[3,3]]])
        max_kj_diag = np.matrix([[max_kj_L[0,0], 0, 0, 0], [0, max_kj_L[1,1], 0, 0], [0, 0, max_kj_R[0,0], 0], [0, 0, 0, max_kj_R[1,1]]])
        
        # desired task space force
        self.des_Fp = max_kp_diag * (r * (self.des_p - self.p))
        
        # desired Cartesian space force
        des_F = np.transpose(Jp) * self.des_Fp
            
        # desired joint space torque
        self.des_tau = np.transpose(J) * des_F
        
        self.des_mL = ((np.matrix([[1/Ksc_L[0,0], 0],[0, 1/Ksc_L[1,1]]]) * np.transpose(R_j_inv_L)*self.des_tau[0:2]) + R_j_L * self.joint_pos[0:2]) / Rm
        self.des_mR = ((np.matrix([[1/Ksc[0,0], 0],[0, 1/Ksc[1,1]]]) * np.transpose(R_j_inv)*self.des_tau[2:4]) + R_j * self.joint_pos[2:4]) / Rm 
                
                   

    def _get_obs(self):
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
            
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
        
        self.Prel = self.P_R - self.P_L + np.array([0.125, 0])
        l = np.sqrt(self.Prel[0]*self.Prel[0] + self.Prel[1]*self.Prel[1])
        self.p = np.array([[l],[(xR+xL+0.125)/2],[(yR+yL)/2],[np.arctan2(yR-yL,xR+0.125-xL)]])
        self.vel_p = self.p - self.Prev_p
        
        self.Prev_Rj = self.Rj
        self.Prev_Lj = self.Lj
        self.Prev_p = self.p
        
        self.prev_force = self.prev_force + (self.des_Fp[0,0] - self.prev_force) * dt / 0.5
        self.prev_lforce = self.prev_lforce + (self.sim.data.sensordata[self.sim.model.sensor_name2id('fingertip_l')] - self.prev_lforce) * dt / 0.5
        self.prev_rforce = self.prev_rforce + (self.sim.data.sensordata[self.sim.model.sensor_name2id('fingertip_r')] - self.prev_rforce) * dt / 0.5
        # self.prev_oforce = self.prev_oforce + (self.sim.data.sensordata[self.sim.model.sensor_name2id('object_frc')] - self.prev_oforce) * dt / 0.5
            
        self.sim.model.body_quat[self.sim.model.body_name2id('debug_body')] = ToQuaternion(0, -np.pi/2.0, self.p[3,0] + np.pi/2.0)
        self.sim.model.body_pos[self.sim.model.body_name2id('debug_body'),0:2] = np.array([-self.p[2,0] + 0.1, self.p[1,0] - 0.0625])
        self.sim.model.site_size[self.sim.model.site_name2id('debug'),1] = self.p[0,0]/2.0
        
        object_pos = self.sim.data.get_body_xpos('object2').copy()
        object_pos_frame = np.array([[object_pos[1]+0.0625], [-(object_pos[0]-0.1)], [object_pos[2]]])
        # self.est_obj_pose = object_pos_frame + np.concatenate(((np.random.random(2)-0.5)*0.02,[0.])).reshape(-1,1) if self.DR and np.linalg.norm(object_pos_frame - self.goal[:3]) > 0.02 else object_pos_frame
        self.est_obj_pose = object_pos_frame
        
        self.prev_oforce = self.sim.data.sensordata[self.sim.model.sensor_name2id('object_frc')] if self.actual_stiffness == self.actual_max_stiffness else self.prev_stiffness_limit * self.sim.data.sensordata[self.sim.model.sensor_name2id('object_frc')]
        object_ori = ToRPY(self.sim.data.get_joint_qpos('object:joint')[3:].copy())[1]
        self.obj_acc = self.sim.data.qacc[self.sim.model.joint_name2id('object3:joint')] * 500 if self.actual_stiffness == self.actual_max_stiffness else self.prev_stiffness_limit * self.sim.data.qacc[self.sim.model.joint_name2id('object3:joint')] * 500
        
        if self.p[0,0] > 0.025 and self.prev_oforce > self.min_grip/2. and self.prev_lforce > 0.0 and self.prev_rforce > 0.0:
             self.sim.model.geom_size[self.sim.model.geom_name2id('object_bottom'),2] = 0.045
             if self.DR and np.linalg.norm(object_pos_frame - self.goal[:3]) > 0.02: self.sim.data.qvel[self.sim.model.joint_name2id('object:joint')+1] += 0.1*(np.random.random()-0.5)
        else:
             self.sim.model.geom_size[self.sim.model.geom_name2id('object_bottom'),2] = 0.05
            
        if self.prev_oforce > self.object_fragility:
            self.sim.model.geom_rgba[-1][0:3] = np.array([0.8000, 0.2627, 0.2824])
            
        if self.n_actions == 6:
            observation = np.array([self.p[0,0], self.p[1,0], self.p[2,0], self.p[3,0], # l, cen_x, cen_y, theta
                                    # self.des_p[0,0]-self.p[0,0], self.des_p[1,0]-self.p[1,0], self.des_p[2,0]-self.p[2,0], self.des_p[3,0]-self.p[3,0],#intention
                                    self.goal[0]-self.est_obj_pose[0,0], self.goal[1]-self.est_obj_pose[1,0],
                                    self.est_obj_pose[0,0]-self.p[1,0], self.est_obj_pose[1,0]-self.p[2,0],
                                    self.prev_force,
                                    self.vel_L[0,0,0], self.vel_L[0,1,0],self.vel_R[0,0,0], self.vel_R[0,1,0],
                                    self.prev_stiffness, self.prev_stiffness_limit
                                    ])
        else:
            observation = np.array([self.p[0,0], self.p[1,0], self.p[2,0], self.p[3,0], # l, cen_x, cen_y, theta
                                    # self.des_p[0,0]-self.p[0,0], self.des_p[1,0]-self.p[1,0], self.des_p[2,0]-self.p[2,0], self.des_p[3,0]-self.p[3,0],#intention
                                    self.goal[0]-self.est_obj_pose[0,0], self.goal[1]-self.est_obj_pose[1,0],
                                    self.est_obj_pose[0,0]-self.p[1,0], self.est_obj_pose[1,0]-self.p[2,0],
                                    self.prev_force,
                                    self.vel_L[0,0,0], self.vel_L[0,1,0],self.vel_R[0,0,0], self.vel_R[0,1,0],
                                   ])
        
        modified_obs = dict(observation=observation, 
                            achieved_goal=np.array([object_pos_frame[0,0], object_pos_frame[1,0], object_pos_frame[2,0], 0.5 * object_ori, # 0.5 for NoDR
                                                    self.prev_oforce, self.obj_acc, 
                                                    self.vel_L[0,0,0], self.vel_L[1,0,0], self.vel_R[0,0,0], self.vel_R[1,0,0]]), 
                            desired_goal = self.goal)
        return modified_obs

    def _viewer_setup(self):
        self.viewer.cam.distance = 0.5
        self.viewer.cam.azimuth = 32.
        self.viewer.cam.elevation = -30.

    def _render_callback(self):
        # Visualize target.
        self.sim.model.body_pos[self.sim.model.body_name2id('target_body'), 0:2] = np.array([-self.goal[1] + 0.1, self.goal[0] - 0.0625])
        self.sim.model.body_quat[self.sim.model.body_name2id('target_body')] = ToQuaternion(0, -np.pi/2.0, 0)
        
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        
        # reset the broken objects
        self.broken_object = False
        self.sim.model.geom_rgba[-1][0:3] = np.array([0.4118, 0.6941, 0.8941])
        
        self.Rj = np.array([[self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Joint_1_R')]]],
                   [self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Joint_2_R')]]]])
        self.Lj = np.array([[self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Joint_1_L')]]],
                       [self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Joint_2_L')]]]])
        
        self.P_R = np.array([L1 * np.cos(self.Rj[0,0] + 1*np.pi/4.0) + L2 * np.cos(self.Rj[0,0]-self.Rj[1,0] + 1*np.pi/4.0), L1 * np.sin(self.Rj[0,0] + 1*np.pi/4.0) + L2 * np.sin(self.Rj[0,0]-self.Rj[1,0] + 1*np.pi/4.0)])
        self.P_L = np.array([L1 * np.cos(self.Lj[0,0] + 3*np.pi/4.0) + L2 * np.cos(self.Lj[0,0]+self.Lj[1,0] + 3*np.pi/4.0), L1 * np.sin(self.Lj[0,0] + 3*np.pi/4.0) + L2 * np.sin(self.Lj[0,0]+self.Lj[1,0] + 3*np.pi/4.0)])
        
        self.Prel = self.P_R - self.P_L + np.array([0.125, 0])
        self.p = np.array([[0.16],[0.0625],[0.1],[0.]])
        self.des_p = np.array([[0.16],[0.0625],[0.1],[0.]])
        self.des_Fp = np.array([[0.],[0.],[0.],[0.]])
        # print(self.p)
        
        self.Prev_p = self.p.copy()
        self.vel_p = self.p - self.Prev_p
        self.Pc = np.array([(self.P_R[0]+self.P_L[0]+0.125)/2,(self.P_R[1]+self.P_L[1])/2])
        
        self.joint_pos = np.concatenate([self.Lj, self.Rj])
        self.joint_pos[3,0] = -self.joint_pos[3,0]
        self.est_obj_pose = np.zeros([2,1])
        
        # reset stiffness
        self.prev_stiffness = 0.2
        self.prev_stiffness_limit = 0.2
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
        self.est_grasping_force = 0.0
        self.est_grasping_force_R = 0.0
        self.est_grasping_force_L = 0.0
        self.obj_acc = 0.0
        
        # minimum grip force
        self.min_grip = 9.81*(self.obj_weight * 1.5)/0.5
        self.sim.model.geom_size[self.sim.model.geom_name2id('object_bottom'),2] = 0.05
        # self.sim.model.body_mass[self.sim.model.body_name2id('object')] = self.obj_weight/2.0
        # self.sim.model.body_mass[self.sim.model.body_name2id('object2')] = self.obj_weight
        
        # reset values
        self.vel_R = np.zeros([3,2,1])
        self.vel_L = np.zeros([3,2,1])
        
        self.sim.forward()
        return True

    def _sample_goal(self):
        goal = np.array([np.random.random_sample()*0.08+0.02, np.random.random_sample()*0.06+0.09, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        initial_qpos = self.sim.data.get_joint_qpos('object:joint').copy()
        initial_pos, initial_quat = initial_qpos[:3], initial_qpos[3:]
        obj_location = np.array([np.random.random_sample()*0.08+0.02, np.random.random_sample()*0.06+0.09])
        while(abs(obj_location[0] - goal[0]) < 0.03 or abs(obj_location[1] - goal[1]) < 0.03):
            obj_location = np.array([np.random.random_sample()*0.08+0.02, np.random.random_sample()*0.06+0.09])
        initial_pos[:2] = np.array([-obj_location[1] + 0.1, obj_location[0] - 0.0625])
        
        self.sim.data.set_joint_qpos('object:joint', initial_qpos)
        
        if self.DR: self.mocap_offset = (np.random.random(4)-0.5) * 0.01 # domain randomization
        
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
        
        utils.reset_mocap_welds(self.sim)
                
        self.sim.forward()
        
        self.sim.data.set_mocap_pos('mocap_l', np.array([0., -0.08, 0.056965]))
        self.sim.data.set_mocap_pos('mocap_r', np.array([0., 0.08, 0.056965]))
        for _ in range(10):
            self.sim.step()

    def render(self, mode='human', width=500, height=500):
        return super(CheolFingersEnv, self).render(mode, width, height)
