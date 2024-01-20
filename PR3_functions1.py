import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2 as cv2
from pr3_utils import *
import autograd.numpy as jnp
from autograd import grad
import transforms3d as tf3d
import scipy
from scipy.sparse import csr_matrix

class dataset:
    index=0
    tstamp=[]
    features=[]
    linear_velocity=[]
    angular_velocity=[]
    K_leftcam=[] # intrinsic parameters of left cam
    stereo_baseline=0
    imu_T_cam = [] # extrinsix transformation left cam to IMU frame

    pose_mean = []
    pose_covariance = []

    slam_pose_trajectory = []
    slam_landmarks = []
    slam_covariance = []
 
    ######################################## IN CLASS FUNCTIONS ##########################################
    def DeadReckon_IMU_Localization(self):
        V = self.linear_velocity # Vx, Vy, Vz
        W = self.angular_velocity # Roll, pitch, yaw rates?
        ts = self.tstamp
        tau = np.diff(ts, n=1)
        init_axis_anlge = np.zeros((6,))
        mu = axangle2pose(init_axis_anlge) # Initial pose
        poseT_traject = np.zeros((4, 4, tau.shape[-1]+1))
        poseT_traject[...,0] = mu 
        for t in range(tau.shape[-1]):
            mu = Motion_mean_predict(tau[:, t], V[:, t], W[:, t], mu)
            poseT_traject[...,t+1] = mu
        return poseT_traject
    
    def IMU_EKF_Predict(self, V_var, W_var):
        V = self.linear_velocity # Vx, Vy, Vz
        W = self.angular_velocity # Roll, pitch, yaw rates?
        ts = self.tstamp
        tau = np.diff(ts, n=1)
        noise_V = np.random.normal(0, V_var, V.shape) # Noise for linear velocity
        noise_W = np.random.normal(0, W_var, W.shape) # Noise angular velocity
        tp_ = np.concatenate((V_var * np.ones([3,]), W_var * np.ones([3,])))
        noise_cov = np.diag(tp_) # noise covariance matrix
        init_axis_anlge = np.zeros((6,))
        mu = axangle2pose(init_axis_anlge) # Initial pose mean
        Eps = noise_cov # initial covariance of perturbation

        mu_traject = np.zeros((4, 4, tau.shape[-1]+1))
        Eps_traject = np.zeros((6, 6, tau.shape[-1]+1))
        mu_traject[...,0] = mu
        Eps_traject[...,0] = Eps
        for t in range(tau.shape[-1]):
            mu = Motion_mean_predict(tau[:, t], V[:, t], W[:, t], mu)  # Mean of pose propogation
            Eps, dummy = Motion_cov_predict(tau[:, t], V[:, t], W[:, t], Eps, noise_cov) # Covaraiance of pose propogation
            mu_traject[...,t+1] = mu
            Eps_traject[...,t+1] = Eps
        self.pose_mean = mu_traject
        self.pose_covariance = Eps_traject

    def Mapping_EKF_Predict_Update(self, pixel_var, num_skip_ftr, num_skip_meas_update):
        ts = self.tstamp
        features = self.features[:,::num_skip_ftr+1,:] # leaving some features
        K_left  = self.K_leftcam
        imu_T_cam = self.imu_T_cam
        stereo_b = self.stereo_baseline
        pose_mean = self.pose_mean
        pose_cov = self.pose_covariance
        M = features.shape[1]
        lmark = np.zeros((3, M))
        lmark_visit = np.zeros((M,))
        lmark_mu = np.zeros((3, M))
        I = np.eye(3*M) # covariance initialized
        lmark_cov = I 
        skip_update = num_skip_meas_update + 1

        #Sparse matrices
        csr_lmark_cov = csr_matrix(lmark_cov)
        csr_I = csr_matrix(I)

        for t in tqdm(range(ts.shape[-1])):
            T_mu = pose_mean[...,t] # current world_T_imu
            z_all = features[...,t]
            z, z_ids = DiscardMissing_z(z_all)
            Nt = len(z_ids) # current observed number of features
            V = pixel_var*np.eye(4*Nt) # Obs model noise covariance (in pixels)
            first_visit = np.squeeze(np.where(lmark_visit[z_ids] == 0)) # Initialize for these
            if first_visit != []:
                lmark_mu[:, z_ids[first_visit]] = stereoPxl2World(K_left, stereo_b, imu_T_cam, T_mu, z[:, first_visit]) # Mean initialized
                lmark[:, z_ids[first_visit]] = stereoPxl2World(K_left, stereo_b, imu_T_cam, T_mu, z[:, first_visit])
            lmark_visit[z_ids] = 1 # Marking visited landmarks
            never_visited = np.squeeze(np.where(lmark_visit==0))
            if len(never_visited) == 0:
                cd = 8

            if np.remainder(t, skip_update) == 0:
                # Update for landmarks
                visited = np.squeeze(np.where(lmark_visit == 1)) # Need Jacobian for these
                H = ObsModel_Jacobian_wrt_lmark(K_left, stereo_b, T_mu, imu_T_cam, lmark_mu, z_ids) # 4Nt x 3M
                K_gain = Map_KalmanGain(csr_lmark_cov, H, V) # Pass sparse cov matrix
                z_mu = ObsModel(K_left, stereo_b, T_mu, imu_T_cam, lmark_mu, z_ids)
                
                z_rs = z.reshape((4*Nt,))
                z_mu_rs = z_mu.reshape((4*Nt,))
                lmark_mu = lmark_mu.reshape((3*M,))
                lmark_mu = lmark_mu + K_gain @ (z_rs - z_mu_rs) # Mean update
                lmark_mu = lmark_mu.reshape((3, M))
                csr_K_gain = csr_matrix(K_gain)
                csr_H = csr_matrix(H)
                tp1 = csr_K_gain.dot(csr_H) # K_gain x H
                I_Kg_H = csr_I - tp1 # I - K_gain x H
                csr_lmark_cov = I_Kg_H.dot(csr_lmark_cov) # cov update = (I - K_gain x H)cov 
        lmark_cov = csr_lmark_cov.toarray()
        
        # plot_MAP(lmark_mu, pose_mean)
        # plot_MAP(lmark, pose_mean)
        self.mapping_landmarks = lmark 

    def VisualInertial_SLAM(self, V_var, W_var, pixel_var, num_skip_ftr, num_skip_update):
        V = self.linear_velocity # Vx, Vy, Vz
        W = self.angular_velocity # Roll, pitch, yaw rates
        ts = self.tstamp
        tau = np.diff(ts, n=1)
        tp_ = np.concatenate((V_var * np.ones([3,]), W_var * np.ones([3,])))
        pose_noise_cov = np.diag(tp_) # noise covariance matrix
        init_axis_anlge = np.zeros((6,))
        pose_mu = axangle2pose(init_axis_anlge) # Initial pose mean
        pose_Eps = pose_noise_cov # initial covariance of perturbation

        features = self.features[:,::num_skip_ftr+1,:] # leaving some features
        K_left  = self.K_leftcam
        imu_T_cam = self.imu_T_cam
        stereo_b = self.stereo_baseline
        M = features.shape[1]
        lmark = np.zeros((3, M))
        lmark_visit = np.zeros((M,))
        lmark_mu = np.zeros((3, M))
        lmark_Eps = np.eye(3*M)  # covariance initialized
        skip_update = num_skip_update + 1

        # Combining pose and landmarks
        I = np.eye(3*M+6) 
        Eps = np.zeros((3*M+6, 3*M+6))
        Eps[0:3*M, 0:3*M] = lmark_Eps
        Eps[3*M:3*M+6, 3*M:3*M+6] = pose_Eps 

        #Sparse matrices
        lil_Eps = scipy.sparse.lil_matrix((3*M+6, 3*M+6), dtype=np.float64) # for efficiently updating  matrix
        lil_Eps[:,:] = Eps[:,:]
        csr_I = csr_matrix(I, dtype=np.float64) # csr_matrix for efficient multiplication

        pose_mu_traject = np.zeros((4, 4, tau.shape[-1]+1))
        # First obsevration of landmarks
        T_mu = pose_mu # initial world_T_imu
        z_all = features[...,0]
        z, z_ids = DiscardMissing_z(z_all)
        Nt = len(z_ids) # current observed number of features
        lmark_noise_cov = pixel_var*np.eye(4*Nt) # Obs model noise covariance (in pixels)
        first_visit = np.squeeze(np.where(lmark_visit[z_ids] == 0)) # Initialize for these
        lmark_mu[:, z_ids[first_visit]] = stereoPxl2World(K_left, stereo_b, imu_T_cam, T_mu, z[:, first_visit]) # Mean initialized
        lmark_visit[z_ids] = 1 # Marking visited landmarks

        pose_mu_traject[...,0] = pose_mu
        for t in tqdm(range(tau.shape[-1])):
            # PREDICT
            pose_mu = Motion_mean_predict(tau[:, t], V[:, t], W[:, t], pose_mu)  # Mean of pose propogation
            pose_Eps, F = Motion_cov_predict(tau[:, t], V[:, t], W[:, t], pose_Eps, pose_noise_cov) # Covaraiance of pose propogation
            lil_Eps[3*M:3*M+6, 3*M:3*M+6] = pose_Eps 
            csr_Eps = lil_Eps.tocsr()
            csr_Eps = Slam_cross_covaraiance_predict(lil_Eps, csr_Eps, F, M)

            z_all = features[...,t+1]
            z, z_ids = DiscardMissing_z(z_all)
            #Nt = len(z_ids) # current observed number of features
            #lmark_noise_cov = pixel_var*np.eye(4*Nt) # Obs model noise covariance (in pixels)
            first_visit = np.squeeze(np.where(lmark_visit[z_ids] == 0)) # Initialize for these
            already_visited = np.squeeze(np.where(lmark_visit[z_ids] == 1)) # Use these for update
            if first_visit != []:
                lmark_mu[:, z_ids[first_visit]] = stereoPxl2World(K_left, stereo_b, imu_T_cam, pose_mu, z[:, first_visit]) # Mean loc initialized
            lmark_visit[z_ids] = 1 # Marking visited landmarks
            ##************************************************##

            # UPDATE
            if np.remainder(t+1, skip_update) == 0 and already_visited != []:
                # Update mean pose
                obs_z_ids = z_ids[already_visited]
                #sel_ids = already_visited
                sel_ids = reject_outlier_landmarks(lmark_mu, obs_z_ids, pose_mu, thresh=1000)
                sel_z_ids = z_ids[sel_ids]
                Nt = len(sel_ids)
                lmark_noise_cov = pixel_var*np.eye(4*Nt).astype(np.float64) # Obs model noise covariance (in pixels)
                H_pose = ObsModel_Jacobian_wrt_pose(K_left, stereo_b, pose_mu, imu_T_cam, lmark_mu, sel_z_ids) # 4Ntx6
                H_lmark = ObsModel_Jacobian_wrt_lmark(K_left, stereo_b, pose_mu, imu_T_cam, lmark_mu, sel_z_ids) # 4Ntx3M
                hmax = np.max(H_lmark)
                hmin = np.min(H_lmark)
                H_slam = np.zeros((4*Nt, 3*M+6)).astype(np.float64)
                H_slam[:, 0:3*M] = H_lmark
                H_slam[:, 3*M:] = H_pose
                K_gain_slam = Slam_Kalmangain(csr_Eps, H_slam, lmark_noise_cov) # Pass sparse cov matrix
                z_tilda = ObsModel(K_left, stereo_b, pose_mu, imu_T_cam, lmark_mu, sel_z_ids)
                z_rs = z[:, sel_ids].reshape((4*Nt,1), order='F')
                z_tilda_rs = z_tilda.reshape((4*Nt,1), order='F')
                delta_mu = K_gain_slam @ (z_rs - z_tilda_rs)
                delta_mu = np.squeeze(delta_mu)

                delta_mu_pose = delta_mu[3*M:]
                delta_mu_pos_hat = axangle2twist(delta_mu_pose)
                exp_delta_mu_pos_hat = scipy.linalg.expm(delta_mu_pos_hat)
                pose_mu = pose_mu @ exp_delta_mu_pos_hat # Pose mean UPDATE

                lmark_mu = lmark_mu.reshape((3*M,), order='F')
                lmark_mu = lmark_mu + delta_mu[0:3*M] # Landmark mean UPDATE
                lmark_mu = lmark_mu.reshape((3, M), order='F')

                csr_K_gain_slam = csr_matrix(K_gain_slam)
                csr_H_slam = csr_matrix(H_slam)
                tp1 = csr_K_gain_slam.dot(csr_H_slam) # K_gain x H
                I_Kg_H = csr_I - tp1 # I - K_gain x H
                csr_Eps = I_Kg_H.dot(csr_Eps) # Covariance UPDATE
                csr_pose_Eps = csr_Eps[3*M:3*M+6, 3*M:3*M+6]
                pose_Eps = csr_pose_Eps.toarray()  # Updated pose cov to be feeded into next predict step 
            pose_mu_traject[...,t+1] = pose_mu  
        Eps = csr_Eps.toarray() 
        self.slam_pose_trajectory = pose_mu_traject
        self.slam_landmarks = lmark_mu
        self.slam_covariance = csr_Eps.toarray() 

######################################## GENERAL FUNCTIONS ##########################################
def plot_Slam_results(slam_landmarks, slam_pose, Ddrkn_pose):
    fig,ax = plt.subplots(figsize=(5,5))
    #tp = np.logical_and(np.abs(slam_landmarks[0,:])<1000, np.abs(slam_landmarks[1,:]<1000))
    landmarks1 = slam_landmarks
    ax.plot(slam_pose[0,3,:],slam_pose[1,3,:],'r-',label='SLAM')
    ax.scatter(slam_pose[0,3,0],slam_pose[1,3,0],marker='s',label="start")
    ax.scatter(slam_pose[0,3,-1],slam_pose[1,3,-1],marker='o',label="end")

    ax.plot(Ddrkn_pose[0,3,:],Ddrkn_pose[1,3,:],'g-',label='Dead-reckon')
    ax.scatter(Ddrkn_pose[0,3,0],Ddrkn_pose[1,3,0],marker='s',label="start")
    ax.scatter(Ddrkn_pose[0,3,-1],Ddrkn_pose[1,3,-1],marker='o',label="end")
    ax.scatter(landmarks1[0,:], landmarks1[1,:], s=5, c='b', label='slam_landmarks')
    ax.legend()
    plt.show(block=True)

def reject_outlier_landmarks(lmark, obs_z_ids, robot_pose, thresh):
    xy_robot = robot_pose[0:2, 3]
    xy_robot = xy_robot.reshape((2, 1))
    delta = lmark[0:2, obs_z_ids] - xy_robot
    dist = np.linalg.norm(delta, axis=0)
    pick_ids = np.squeeze(np.where(dist <= thresh))
    bh=1
    return pick_ids

def plot_MAP(landmarks, pose):
    fig,ax = plt.subplots(figsize=(5,5))
    n_pose = pose.shape[2]
    tp = np.logical_and(np.abs(landmarks[0,:])<1000, np.abs(landmarks[1,:]<1000))
    landmarks1 = landmarks[:, tp]
    ax.plot(pose[0,3,:],pose[1,3,:],'r-',label='Dead-reckon')
    ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
    ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
    ax.scatter(landmarks1[0,:], landmarks1[1,:], s=5, c='b', label='landmarks')
    ax.legend()
    plt.show(block=True)

def Slam_Kalmangain(csr_cov, H, V):
    csr_H = csr_matrix(H)
    H_t = np.transpose(H)
    csr_H_t = csr_matrix(H_t)
    csr_cov_H_t = csr_cov.dot(csr_H_t)
    cov_H_t = csr_cov_H_t.toarray()
    csr_tp = csr_H.dot(csr_cov_H_t) 
    tp = csr_tp.toarray()
    tp_V = (tp + V)
    csr_H_cov = csr_H.dot(csr_cov)
    H_cov = csr_H_cov.toarray()
    K_gain1_T = np.linalg.solve(tp_V, H_cov) 
    K_gain1 = np.transpose(K_gain1_T)
    return K_gain1

def Map_KalmanGain(csr_lmark_cov, H, V):
    csr_H = csr_matrix(H)
    H_t = np.transpose(H)
    csr_H_t = csr_matrix(H_t)
    csr_cov_H_t = csr_lmark_cov.dot(csr_H_t)
    cov_H_t = csr_cov_H_t.toarray()
    csr_tp = csr_H.dot(csr_cov_H_t) 
    tp = csr_tp.toarray()
    tp_V = tp + V 
    tp_inv = np.linalg.inv(tp_V)
    K_gain = cov_H_t @ tp_inv
    return K_gain

def stereoPxl2World(K_left, stereo_b, imu_T_cam, wrld_T_imu, pixel_uv):
    d = pixel_uv[0,...] - pixel_uv[2,...]
    fsu_b = K_left[0, 0] * stereo_b
    z = fsu_b / d
    fsu = K_left[0, 0]
    fsv = K_left[1, 1]
    cu = K_left[0, 2]
    cv = K_left[1, 2]
    uL = pixel_uv[0,...]
    vL = pixel_uv[1,...]
    opt_xyz = np.ones((4, pixel_uv.shape[-1])) # homogeneous form
    opt_xyz[0,...] = (uL - cu)*z/fsu
    opt_xyz[1,...] = (vL- cv)*z/fsv
    opt_xyz[2,...] = z
    R_opt2reg = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    reg_xyz = np.ones((4, pixel_uv.shape[-1])) # homogeneous form
    reg_xyz[0:3,...] = R_opt2reg @ opt_xyz[0:3,...] # Optical to regular camera frame
    imu_xyz = imu_T_cam @ reg_xyz  # TO DO: Flip to shift from IMU axes to regular world??
    wrld_xyz = wrld_T_imu @ imu_xyz
    return wrld_xyz[0:3,:]

def ObsModel_Jacobian_wrt_pose(K, b, T, imu_T_cam, m, obs_id):
    m_obs = m[:, obs_id]
    m_ = np.ones((4, m_obs.shape[1]))  # Homogeneous coordinates. Append 1
    if m.shape[0] == 3:
        m_[0:3, :] = m_obs
    else:
        m_ = m_obs
    M = m.shape[1]
    Nt = m_obs.shape[1]
    H_pose = np.zeros((4*Nt, 6))
    Ks = np.zeros((4, 4))
    Ks[0:3, 0:3] = K
    fsu_b = K[0, 0] * b
    Ks[2, 0:3] = K[0, :]
    Ks[2, 3] = -fsu_b
    Ks[3, 0:3] = K[1, :]

    T_inv = inversePose(T) # world to imu
    cam_T_imu = inversePose(imu_T_cam) # imu to cam regular
    opt_R_reg = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    opt_T_reg = np.zeros((4, 4))
    opt_T_reg[0:3, 0:3] = opt_R_reg
    opt_T_reg[3, 3] = 1
    opt_T_imu = opt_T_reg @ cam_T_imu # imu to cam optical

    for i in range(Nt): # each observed landmark
        T_inv_m = T_inv @ m_[:, i] # 4x1
        T_inv_m_dot = vect2dotmat(T_inv_m) # 4x6
        Topt_T_inv_m = opt_T_imu @ T_inv_m # 4x1
        pJac = projectionJacobian(np.transpose(Topt_T_inv_m)) # 4x4
        Ks_pJac = -Ks @ pJac # 4x4
        H_pose_i = Ks_pJac @ opt_T_imu @ T_inv_m_dot # 4x6
        H_pose[i*4:(i+1)*4, :] = H_pose_i # 4Ntx6
    return H_pose.astype(np.float64)

def vect2dotmat(s_):
    s_hat = axangle2skew(s_[0:3])
    s_dot = np.zeros((4, 6))
    s_dot[0:3, 0:3] = np.eye(3)
    s_dot[0:3, 3:] = -s_hat
    return s_dot

def ObsModel_Jacobian_wrt_lmark(K, b, T, imu_T_cam, m, obs_id):
    m_obs = m[:, obs_id]
    m_ = np.ones((4, m_obs.shape[1]))  # Homogeneous coordinates. Append 1
    if m.shape[0] == 3:
        m_[0:3, :] = m_obs
    else:
        m_ = m_obs
    M = m.shape[1]
    Nt = m_obs.shape[1]
    T_inv = inversePose(T) # world to imu
    cam_T_imu = inversePose(imu_T_cam) # imu to cam regular
    opt_R_reg = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    opt_T_reg = np.zeros((4, 4))
    opt_T_reg[0:3, 0:3] = opt_R_reg
    opt_T_reg[3, 3] = 1
    opt_T_imu = opt_T_reg @ cam_T_imu # imu to cam optical
    P = np.zeros((3,4))
    P[0:3, 0:3] = np.eye(3)
    P_t = np.transpose(P)

    #  Computing H
    H = np.zeros((4*Nt, 3*M))
    Ks = np.zeros((4, 4))
    Ks[0:3, 0:3] = K
    fsu_b = K[0, 0] * b
    Ks[2, 0:3] = K[0, :]
    Ks[2, 3] = -fsu_b
    Ks[3, 0:3] = K[1, :]
    T_ = opt_T_imu @ T_inv
    T_m_ = T_ @ m_
    T_P = T_ @ P_t
    pJac = projectionJacobian(np.transpose(T_m_))
    pJac = pJac.transpose((1, 2, 0)) # 4 x 4 x Nt 
    pJac_res = np.zeros((4*Nt,4))
    for i in range(Nt):
        pJac_res[i*4:(i+1)*4, :] = pJac[:, :, i] # 4Nt x 4
    pJac_T_P = pJac_res @ T_P
    pJac_T_P_res = np.zeros((4, 3*Nt))
    for i in range(Nt):
        pJac_T_P_res[:, i*3:(i+1)*3] = pJac_T_P[i*4:(i+1)*4, :] # 4Nt x 3
    H_ = Ks @ pJac_T_P_res # 4x3Nt
    for i in range(Nt):
        m_i = obs_id[i] # index in 0:M-1
        curH_ = H_[:, i*3:(i+1)*3]
        H[i*4:(i+1)*4, m_i*3:(m_i+1)*3] = curH_ # 4Nt x 3M
    return H.astype(np.float64)

def ObsModel(K, b, T, imu_T_cam, m, obs_id):
    m_obs = m[:, obs_id]
    m_ = np.ones((4, m_obs.shape[1]))  # Homogeneous coordinates. Append 1
    if m.shape[0] == 3:
        m_[0:3, :] = m_obs
    else:
        m_ = m_obs
    T_inv = inversePose(T) # world to imu
    cam_T_imu = inversePose(imu_T_cam) # imu to cam regular
    opt_R_reg = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    opt_T_reg = np.zeros((4, 4))
    opt_T_reg[0:3, 0:3] = opt_R_reg
    opt_T_reg[3, 3] = 1
    opt_T_imu = opt_T_reg @ cam_T_imu # imu to cam optical
    Ks = np.zeros((4, 4))
    Ks[0:3, 0:3] = K
    fsu_b = K[0, 0] * b
    Ks[2, 0:3] = K[0, :]
    Ks[2, 3] = -fsu_b
    Ks[3, 0:3] = K[1, :]

    t_p = opt_T_imu @ T_inv @ m_
    t_p1 = np.transpose(projection(np.transpose(t_p)))
    z_p = Ks @ t_p1
    return z_p

def DiscardMissing_z(z_all):
    valid = np.where(np.logical_and(np.logical_and(z_all[0,...]!=-1, z_all[1,...]!=-1), \
                                    np.logical_and(z_all[2,...]!=-1, z_all[3,...]!=-1)))
    z = np.squeeze(z_all[...,valid])
    return z, np.squeeze(valid)

def Slam_cross_covaraiance_predict(cur_lil_cov, cur_csr_cov, F, M):
    F_t = np.transpose(F)
    csr_F = csr_matrix(F)
    csr_F_t = csr_matrix(F_t)
    cur_lil_cov[0:3*M, 3*M:] = cur_csr_cov[0:3*M, 3*M:].dot(csr_F_t)
    cur_lil_cov[3*M:, 0:3*M] = csr_F.dot(cur_csr_cov[3*M:, 0:3*M]) 
    cur_lil_cov[3*M:3*M+6, 3*M:3*M+6] = cur_csr_cov[3*M:3*M+6, 3*M:3*M+6]
    next_csr_cov = cur_lil_cov.tocsr()
    return next_csr_cov

def Motion_cov_predict(cur_tau, cur_V, cur_W, cur_cov, noise_cov):
    cur_twist = np.concatenate((cur_V, cur_W))
    cur_twist_curlyhat = axangle2adtwist(cur_twist)
    texpm = scipy.linalg.expm(-cur_tau * cur_twist_curlyhat)
    texpm_T = np.transpose(texpm)
    next_cov = texpm @ cur_cov @ texpm_T + noise_cov # @ -> matrix multiplication
    return next_cov.astype(np.float64), texpm

def Motion_mean_predict(cur_tau, cur_V, cur_W, cur_mu):
    cur_twist = np.concatenate((cur_V, cur_W))
    cur_twist_hat  = axangle2twist(cur_twist) # 6x1 twist vector to 4x4 matrix
    #cur_twist_hat1 = twist2twistmat(cur_twist)
    tp = cur_tau * cur_twist_hat
    exp_tp = scipy.linalg.expm(tp)
    next_mu = cur_mu @ exp_tp
    return next_mu

def readData(datapath):
    files_list_all = sorted(os.listdir(datapath))
    files_list_npz = [x for x in files_list_all if x.endswith('.npz')]
    data_dict = dict()
    for fil in files_list_npz:
        cur_data = load_data(os.path.join(datapath, fil))
        t1 = fil.find('.npz')
        id = int(fil[0:t1])
        if id not in data_dict.keys():
            tpset = dataset()
            tpset.index = id
        else:
            tpset = data_dict[id]
        tpset.tstamp = cur_data[0] 
        tpset.features = cur_data[1] 
        tpset.linear_velocity = cur_data[2] 
        tpset.angular_velocity = cur_data[3] 
        tpset.K_leftcam = cur_data[4] 
        tpset.stereo_baseline = cur_data[5] 
        tpset.imu_T_cam = cur_data[6] 
        if id not in data_dict.keys():
            data_dict[id] = tpset
    return data_dict