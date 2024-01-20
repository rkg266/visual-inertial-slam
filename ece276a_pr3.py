from PR3_functions1 import *
from pr3_utils import *

### USER EDITABLE ###
# Load the measurements
datapath = r'/home/renukrishna/ece276a/Project3/data'
data_dict = readData(datapath)
data_ids = [3, 10]
RUN_SLAM = 1
RUN_PLOTTING = 1
### USER EDITABLE ###

for id in data_ids:
    cur_set = data_dict[id]

    if RUN_SLAM == 1:

        # (a) IMU Localization via EKF Prediction
        cur_set.IMU_EKF_Predict(V_var=1, W_var=1e-4)
        #visualize_trajectory_2d(cur_set.pose_mean, show_ori = True)
        
        # (b) Landmark Mapping via EKF Update
        if id == 3:
            num_skip_ftr = 1
        if id == 10:
            num_skip_ftr = 2 
        print('Data-'+str(id)+': Landmark mapping started...')
        cur_set.Mapping_EKF_Predict_Update(pixel_var=25, num_skip_ftr=num_skip_ftr, num_skip_meas_update=0) ### USER EDITABLE ###
        print('Data-'+str(id)+': Landmark mapping completed...')

        # (c) Visual-Inertial SLAM
        print('Data-'+str(id)+': VI-SLAM started...')
        cur_set.VisualInertial_SLAM(V_var=2, W_var=1e-3, pixel_var=25, num_skip_ftr=num_skip_ftr, num_skip_update=0) ### USER EDITABLE ###
        print('Data-'+str(id)+': VI-SLAM completed...')

        SLAM_landmarks = cur_set.slam_landmarks
        SLAM_pose = cur_set.slam_pose_trajectory
        Dead_reckon_pose = cur_set.pose_mean
        Dead_reckon_landmarks = cur_set.mapping_landmarks
        np.savez('Fullresults_'+str(id)+ '.npz', SLAM_landmarks=SLAM_landmarks, SLAM_pose=SLAM_pose, \
                Dead_reckon_pose=Dead_reckon_pose, Dead_reckon_landmarks=Dead_reckon_landmarks)

    if RUN_PLOTTING == 1:
        # Plotting results
        with np.load('Fullresults_'+str(id)+ '.npz') as data:
            SLAM_landmarks = data['SLAM_landmarks'] # From part c
            SLAM_pose = data['SLAM_pose'] # From part c
            Dead_reckon_pose = data['Dead_reckon_pose'] # From part a
            Mapping_landmarks = data['Dead_reckon_landmarks'] # From part b

        plot_Slam_results(SLAM_landmarks, SLAM_pose, Dead_reckon_pose)
bh=7
