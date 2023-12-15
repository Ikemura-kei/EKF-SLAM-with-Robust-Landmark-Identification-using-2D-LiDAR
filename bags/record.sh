#! /bin/bash

rosbag record -o $1 \
    /ground_truth_trajectory \
    /ground_truth_pose \
    /landmark_obs \
    /ground_truth_landmarks \
    /simulated_odometry \
    /ground_truth_pose_marker \
    /ground_truth_landmarks_marker \
    /landmarks_obs_marker \
    /tf \
    /tf_static