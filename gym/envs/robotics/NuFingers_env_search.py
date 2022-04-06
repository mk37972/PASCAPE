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

frictionData_L = np.load('C:/Users/mk37972/Coding/gym_adjustments/friction_data_L.npz', allow_pickle=True) #load the demonstration data from data file
frictionData_R = np.load('C:/Users/mk37972/Coding/gym_adjustments/friction_data_R.npz', allow_pickle=True) #load the demonstration data from data file

fric_in_L = frictionData_L['input_data']
fric_out_L = frictionData_L['force']
fric_in_R = frictionData_R['input_data']
fric_out_R = frictionData_R['force']
processed_input_L = []
processed_output_L = []
processed_input_R = []
processed_output_R = []

for epsd in range(1): # we initialize the whole demo buffer at the start of the training
    for transition in range(6435):
        processed_input_L.append([fric_in_L[epsd][transition]])
        processed_output_L.append([fric_out_L[epsd][transition]])
        processed_input_R.append([fric_in_L[epsd][transition]])
        processed_output_R.append([fric_out_L[epsd][transition]])
input_data_L = np.array(processed_input_L).reshape([6435,4])
force_data_L = np.array(processed_output_L).reshape([6435,2])
input_data_R = np.array(processed_input_R).reshape([6435,4])
force_data_R = np.array(processed_output_R).reshape([6435,2])

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
        self.reward_type = reward_type
        self.vel_threshold = 1e-4
        self.broken_table = False
        self.broken_object = False
        self.max_stiffness = 1.
        self.prev_stiffness = self.max_stiffness
        self.prev_stiffness_limit = self.max_stiffness
        self.actual_max_stiffness = 1.13715e-0
        self.actual_max_friction = 2.e-2
        self.prev_oforce = 0.0
        self.object_fragility = 4.5
        self.min_grip = 0.0
        self.fric_mu = 0.7
        self.grav_const = 9.81
        self.previous_input = 0
        self.th = np.zeros([2,1])
        self.Prev_th = np.zeros([2,1])
        self.des_th = np.zeros([2,1])
        self.vel_th = np.zeros([3,2,1])
        self.lower_limit = 0.0 if eval_env else -np.pi/2.0
        self.upper_limit = np.pi/2.0 if eval_env else 0.0
        self.pert_type = pert_type
        self.n_actions = n_actions
        self.eval_env = eval_env
        self.des_mR = np.zeros([2,1])
        self.des_mL = np.zeros([2,1])
        self.est_torques = np.zeros([2,1])

        super(CheolFingersEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, initial_qpos=initial_qpos, n_actions=n_actions)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        try: 
            intention = np.linalg.norm(achieved_goal[:,:1] < goal[:,:1], axis=-1)
            velocity = goal_distance(achieved_goal[:,1:2], goal[:,1:2])
            torque = np.linalg.norm((achieved_goal[:,2:3] - goal[:,2:3])*((achieved_goal[:,2:3] - goal[:,2:3]) > 0), axis=-1)
        except: 
            intention = np.linalg.norm(achieved_goal[:1] < goal[:1], axis=-1)
            velocity = goal_distance(achieved_goal[1:2], goal[1:2])
            torque = np.linalg.norm((achieved_goal[2:3] - goal[2:3])*((achieved_goal[2:3] - goal[2:3]) > 0), axis=-1)
        # print("Partly: {}".format(-10*(torque).astype(np.float32)))
        return -(intention).astype(np.float32) -(velocity > self.vel_threshold).astype(np.float32) -10*(torque).astype(np.float32)
        

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        self.sim.forward()

    def _set_action(self, action):
        if self.eval_env == True: # right finger
            # calculating instant forces
            # tendon space stiffness
            Ksc = np.matrix([[self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('AJ1_R'), 0], 0],[0, self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('AJ2_R'), 0]]])/Rm/Rm
            
            # current joint angles (after settling)
            current_Rj = np.array([[self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Joint_1_R')]]],
                           [-self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Joint_2_R')]]]])
            
            # currently experienced torques
            self.est_torques = np.transpose(R_j) * Ksc * (self.des_mR * Rm - R_j * current_Rj)
        else: # left finger
            # calculating instant forces
            # tendon space stiffness
            Ksc_L = np.matrix([[self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('AJ1_L'), 0], 0],[0, self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('AJ2_L'), 0]]])/Rm/Rm
            
            # current joint angles (after settling)
            current_Lj = np.array([[self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Joint_1_L')]]],
                           [self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Joint_2_L')]]]])            
            
            # currently experienced torques
            self.est_torques = np.transpose(R_j_L) * Ksc_L * (self.des_mL * Rm - R_j_L * current_Lj)
        self.prev_oforce = self.sim.data.sensordata[self.sim.model.sensor_name2id('object_frc')]
            
        action = action.copy()  # ensure that we don't change the action outside of this scope
        
        pos_ctrl = action[:1].copy()
        
        # scale action
        change_th = pos_ctrl[0] * np.pi/9 if self.eval_env else -pos_ctrl[0] * np.pi/9
        
        stiffness_ctrl = 0.0
        stiffness_limit = 0.0
        
        if action.shape[0] > 1:
            stiffness_limit = 0.5 * self.max_stiffness * action[2]
            
            self.prev_stiffness_limit += stiffness_limit
            self.prev_stiffness_limit = np.max([np.min([self.prev_stiffness_limit, self.max_stiffness]), self.max_stiffness / 10.0])
            # self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('AJ1_R'), 0] = self.actual_max_stiffness * self.prev_stiffness_limit 
            # self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('AJ1_R'), 1] = -self.actual_max_stiffness * self.prev_stiffness_limit 
            # self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('AJ2_R'), 0] = self.actual_max_stiffness * self.prev_stiffness_limit 
            # self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('AJ2_R'), 1] = -self.actual_max_stiffness * self.prev_stiffness_limit 
            # self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('AJ1_L'), 0] = self.actual_max_stiffness * self.prev_stiffness_limit 
            # self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('AJ1_L'), 1] = -self.actual_max_stiffness * self.prev_stiffness_limit 
            # self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('AJ2_L'), 0] = self.actual_max_stiffness * self.prev_stiffness_limit 
            # self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('AJ2_L'), 1] = -self.actual_max_stiffness * self.prev_stiffness_limit 
            # self.sim.model.dof_damping[self.sim.model.joint_name2id('Joint_1_L')] = self.actual_max_friction * self.prev_stiffness_limit 
            # self.sim.model.dof_damping[self.sim.model.joint_name2id('Joint_1_R')] = self.actual_max_friction * self.prev_stiffness_limit  
            # self.sim.model.dof_damping[self.sim.model.joint_name2id('Joint_2_L')] = self.actual_max_friction * self.prev_stiffness_limit 
            # self.sim.model.dof_damping[self.sim.model.joint_name2id('Joint_2_R')] = self.actual_max_friction * self.prev_stiffness_limit  
            
            stiffness_ctrl = 0.5 * self.max_stiffness * action[1]
            
            self.prev_stiffness += stiffness_ctrl
            self.prev_stiffness = np.max([np.min([self.prev_stiffness, self.prev_stiffness_limit]), self.max_stiffness / 10.0])
            # print(self.prev_stiffness_limit)
        
        self.des_th[0,0] = np.clip(self.des_th[0,0] + change_th, self.lower_limit, self.upper_limit)
        
        # tendon space stiffness
        Ksc = np.matrix([[self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('AJ1_R'), 0], 0],[0, self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('AJ2_R'), 0]]])/Rm/Rm
        Ksc_L = np.matrix([[self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('AJ1_L'), 0], 0],[0, self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('AJ2_L'), 0]]])/Rm/Rm
        
        # joint space stiffness
        max_kj_R = np.transpose(R_j) * Ksc * R_j
        max_kj_L = np.transpose(R_j_L) * Ksc_L * R_j_L
        max_kj_diag_R = np.matrix([[max_kj_R[0,0], 0], [0, max_kj_R[1,1]]])
        max_kj_diag_L = np.matrix([[max_kj_L[0,0], 0], [0, max_kj_L[1,1]]])
        
        # desired joint space torque
        if self.eval_env == True:
            self.des_tau = 0.8 * max_kj_diag_R * (np.array([[self.prev_stiffness],[1.0]]) * (self.des_th - self.th))
            self.des_mR = ((np.matrix([[1/Ksc[0,0], 0],[0, 1/Ksc[1,1]]]) * np.transpose(R_j_inv)*self.des_tau) + R_j * self.th) / Rm 
        else:
            self.des_tau = 0.8 * max_kj_diag_L * (np.array([[self.prev_stiffness],[1.0]]) * (self.des_th - self.th))
            self.des_mL = ((np.matrix([[1/Ksc_L[0,0], 0],[0, 1/Ksc_L[1,1]]]) * np.transpose(R_j_inv_L)*self.des_tau) + R_j_L * self.th) / Rm
        
        prob = 0.1 if self.pert_type == 'delay' else -0.1
        if np.random.random() > prob:
            self.sim.data.ctrl[0] = self.des_mL[0,0]
            self.sim.data.ctrl[1] = self.des_mL[1,0]
            self.sim.data.ctrl[2] = self.des_mR[0,0]
            self.sim.data.ctrl[3] = self.des_mR[1,0]
            self.previous_input = self.sim.data.ctrl
        else:
            try: self.sim.data.ctrl = self.previous_input
            except: 
                self.sim.data.ctrl[0] = 0.
                self.sim.data.ctrl[1] = 0.
                self.sim.data.ctrl[2] = 0.
                self.sim.data.ctrl[3] = 0.

    def _get_obs(self):
        # positions
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
        
        
        # if self.pert_type != 'none' and self.pert_type != 'meas':
        # if self.grasped_flag == 0.1: self.sim.data.qvel[self.sim.model.joint_name2id('object:joint')+1] += 0.5*(np.random.random()-0.5)
        if self.n_actions == 3:
            if self.eval_env:
                observation = np.array([self.th[0,0], 
                                        self.des_th[0,0]-self.th[0,0], self.vel_th[0,0,0],
                                        10 * self.est_torques[0,0], 
                                        0.25 * self.prev_stiffness, 0.25 * self.prev_stiffness_limit
                                        ])
            else:
                observation = np.array([-self.th[0,0],
                                        -self.des_th[0,0]+self.th[0,0], -self.vel_th[0,0,0],
                                        -10 * self.est_torques[0,0],
                                        0.25 * self.prev_stiffness, 0.25 * self.prev_stiffness_limit
                                        ])
                    
        else:
            if self.eval_env:
                observation = np.array([self.th[0,0], 
                                        self.des_th[0,0]-self.th[0,0], self.vel_th[0,0,0],
                                        10 * self.est_torques[0,0],
                                        ])
            else:
                observation = np.array([-self.th[0,0],
                                        -self.des_th[0,0]+self.th[0,0], -self.vel_th[0,0,0],
                                        -10 * self.est_torques[0,0],
                                        ])
        
        modified_obs = dict(observation=observation, achieved_goal=np.array([observation[1],observation[2],observation[3]]), desired_goal = self.goal)
        
        return modified_obs

    def _viewer_setup(self):
        self.viewer.cam.distance = 0.5
        self.viewer.cam.azimuth = 32.
        self.viewer.cam.elevation = -30.

    def _render_callback(self):
        # Visualize target.
        
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        
        # reset the broken objects
        self.broken_object = False
        self.Rj = np.array([[0.],[0.]])
        self.Lj = np.array([[0.],[0.]])
        self.des_mR = np.zeros([2,1])
        self.des_mL = np.zeros([2,1])
        
        self.th = np.array([[0.],[0.]])
        self.Prev_th = np.array([[0.],[0.]])
        self.des_th = np.array([[0.],[0.]])
        self.vel_th = np.zeros([3,2,1])
        self.prev_oforce = 0.0
        
        # reset stiffness
        self.prev_stiffness = self.max_stiffness
        self.prev_stiffness_limit = self.max_stiffness
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
        
        self.sim.forward()
        return True

    def _sample_goal(self):
        goal = np.array([0.2, 0.0, 1e-3])
        
        initial_qpos = self.sim.data.get_joint_qpos('object:joint').copy()
        initial_pos, initial_quat = initial_qpos[:3], initial_qpos[3:]
        initial_pos[:2] = np.array([-(np.random.random_sample()*0.02 + 0.085) + 0.0873, np.random.random_sample()*0.07+0.02 - 0.0635])
        initial_quat = ToQuaternion(np.random.random_sample()*np.pi/4-np.pi/8, 0, 0)
        initial_qpos[:3] = initial_pos
        
        self.sim.data.set_joint_qpos('object:joint', initial_qpos)
        return goal

    def _is_success(self, achieved_goal, desired_goal):
        try: 
            intention = np.linalg.norm(achieved_goal[:,:1] > desired_goal[:,:1], axis=-1)
            velocity = goal_distance(achieved_goal[:,1:2], desired_goal[:,1:2])
            torque = np.linalg.norm(achieved_goal[:,2:3] > desired_goal[:,2:3], axis=-1)
        except: 
            intention = np.linalg.norm(achieved_goal[:1] > desired_goal[:1], axis=-1)
            velocity = goal_distance(achieved_goal[1:2], desired_goal[1:2])
            torque = np.linalg.norm(achieved_goal[2:3] > desired_goal[2:3], axis=-1)
        return (intention * (velocity < self.vel_threshold) * torque).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
                
        self.sim.forward()
        for _ in range(10):
            self.sim.step()

    def render(self, mode='human', width=500, height=500):
        return super(CheolFingersEnv, self).render(mode, width, height)
