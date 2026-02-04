# EKF-SLAM for Visual Localization and Mapping
Implemented simultaneous localization and mapping (SLAM), leveraging extended Kalman filter for estimating the SLAM pose of a moving car and landmark positions in its surroundings.

## Data utilized:
1. IMU: Linear acceleration and angular velocity.
2. Stereo-camera images: Left-right camera pixel values of labelled landmarks throughout the car's journey.
<br>For data related queries, contact rkgutta266@gmail.com

## Tasks:
1. Estimating the path taken by the car as well as showing the landmarks seen from the car during the journey. <br>
**Results:** Deadreckon and estimated trajectories shown together for two different datasets <br>
![Dataset 3](/plots_images/data3_v1_w0001_pix25.png) <br>
![Dataset 10](/plots_images/data10_v1_w0001_pix25.png)

## Discussion:
* The SLAM estimated trajectories are completely different from the dead-reckoning. In the dead-reckoning, there was
no noise involved and trajectory was computed using kinematics equations. Whereas, in SLAM we introduced noise
to the kinematics and then corrected the estimates based in the observations.
* In real life scenario, motion is not governed absolutely by kinematics. There are many factors like drag, friction, etc.,
that cause deviation from the kinematics equations. Hence, it is always safe to estimate the trajectory in a probabilistic
sense and using the observations to strengthen the hypothesis.

## Running code:
Libraries required: os, numpy, matplotlib, tqdm, scipy.sparse, cv2.

Files present: 
1. ece276_pr3.py: Main code
2. PR3_functions1.py: Functions implementation
3. pr3_utils.py: Instructor provided code

* Open ece276_pr2.py
* Find the "USER EDITABLE" section in the code and update the paths, parameters.
* Run ece276_pr2.py. <br>
 Two modes available: (can be run simultaneously)
	- RUN_SLAM = 1: Executes all the problem statements, saves the results as .npz in working directory and displays plots. <br>
	- RUN_PLOTTING = 1: Runs just the plots from already saved results data.

