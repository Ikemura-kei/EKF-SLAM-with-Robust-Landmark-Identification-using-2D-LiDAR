import numpy as np
import rospy

from landmark_msgs.msg import Landmark, Landmarks
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Pose, PoseStamped, Point
from sensor_msgs.msg import LaserScan
import copy
from visualization_msgs.msg import Marker, MarkerArray
from scipy.odr import RealData, ODR, Model

from ekf_slam.gao_et_al import *
from ekf_slam.utils import pnt_on_line_closest_to, ego2global
import matplotlib.pyplot as plt

import time

from tf.transformations import quaternion_from_euler

def wrap_bearing(bearing):
    return np.mod(bearing + np.pi, 2 * np.pi) - np.pi

def dead_reckoning(x:np.ndarray, dt:float, u:np.ndarray):
    theta = x[2]
    B = np.array([[np.cos(theta)*dt, 0],[np.sin(theta)*dt, 0],[0, dt]])
    
    new_x = np.eye(3) @ x + B @ u
    new_x[2] = wrap_bearing(new_x[2])
    
    return new_x

def get_pose_msg(cur_state, stamp):
    pose = PoseStamped()
    pose.header.stamp = stamp
    pose.header.frame_id = "world"
    pose.pose.position.x = cur_state[0]
    pose.pose.position.y = cur_state[1]
    q = quaternion_from_euler(0, 0, cur_state[2])
    pose.pose.orientation.x = q[0]
    pose.pose.orientation.y = q[1]
    pose.pose.orientation.z = q[2]
    pose.pose.orientation.w = q[3]
    
    return pose

def construct_pose_path_msgs(path, cur_state, stamp):
    pose = get_pose_msg(cur_state, stamp)
    
    path.header.frame_id = "world"
    path.header.stamp = stamp
    pose_copy = PoseStamped()
    pose_copy.pose.position.x = pose.pose.position.x
    pose_copy.pose.position.y = pose.pose.position.y
    pose_copy.pose.orientation.x = pose.pose.orientation.x
    pose_copy.pose.orientation.y = pose.pose.orientation.y
    pose_copy.pose.orientation.z = pose.pose.orientation.z
    pose_copy.pose.orientation.w = pose.pose.orientation.w
    path.poses.append(pose_copy)
    if len(path.poses) > MAX_PATH_LEN:
        path.poses.pop(0)
        
    return pose, path

# -- internal constant variables --
NUM_LANDMARKS = 8
STATE_LEN = NUM_LANDMARKS*2+3
MAX_PATH_LEN = 6100
R = np.zeros((STATE_LEN,STATE_LEN))
Q = np.zeros((2,2))

# -- user defined parameters --
var_x = 0.0055
var_y = 0.0035
var_theta = 0.00575
# -- for gt obs --
# var_r = 0.11
# var_phi = 0.07
# -- for extracted landmarks --
var_r = 0.5
var_phi = 0.425
switch_pose_time = 5

# -- internal global variables --
lock = False
rx_cnt = 0
# -- dead reckoning --
prev_state_dr = cur_state_dr = np.array([0, 0, 0])
path_dr = Path()
path_dr_pub = None
pose_dr_pub = None
# -- ekf slam --
prev_state_ekf = cur_state_ekf = cur_state_ekf_bar = np.zeros(STATE_LEN)
prev_cov = cur_cov = cur_cov_bar = np.zeros((STATE_LEN, STATE_LEN))
path_ekf = Path()
path_ekf_pub = None
pose_ekf_pub = None
prev_odom_time = None
obs_book = [False] * NUM_LANDMARKS

# -- landmark extractors --
gao_et_al = GaoEtAl()
p = s = e = None
scan_rx_cnt = 0
scan = None
landmark_pos = None
vis_pubed = False
dummy_pnt = None
start_time = None
anchor_candidates = np.array([[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0], [1.5, 0], [2.5, 0], [1.5, 1.5], [-1.5, -1.5], [3.5, 3.5], [-3.5, -3.5]])
anchor = anchor_candidates[3]

# -- ekf slam building blocks --
def prediction(mu_t_minus_1:np.ndarray, cov_t_minus_1:np.ndarray, u_t:np.ndarray, dt:float, R:np.ndarray):
    mu_t_bar = mu_t_minus_1.copy()
    linear_displacement = u_t[0] * dt
    angular_displacement = u_t[1] * dt
    
    # -- motion model, mean update --
    c_theta = np.cos(mu_t_minus_1[2])
    s_theta = np.sin(mu_t_minus_1[2])
    mu_t_bar[0] = mu_t_minus_1[0] + c_theta * linear_displacement
    mu_t_bar[1] = mu_t_minus_1[1] + s_theta * linear_displacement
    mu_t_bar[2] = wrap_bearing(mu_t_minus_1[2] + angular_displacement)
    
    # -- compute Jacobian G --
    G_sub = np.zeros((3,3))
    G_sub[0,0] = 1
    G_sub[0,1] = 0
    G_sub[0,2] = -s_theta * linear_displacement
    G_sub[1,0] = 0
    G_sub[1,1] = 1
    G_sub[1,2] = c_theta * linear_displacement
    G_sub[2,0] = 0
    G_sub[2,1] = 0
    G_sub[2,2] = 1
    G_up = np.concatenate([G_sub, np.zeros((3, 2*NUM_LANDMARKS))], axis=1) # (3, 2N+3)
    G_low = np.concatenate([np.zeros((2*NUM_LANDMARKS, 3)), np.eye(2*NUM_LANDMARKS)], axis=1) # (2N, 2N+3)
    G_t = np.concatenate([G_up, G_low], axis=0) # (2N+3, 2N+3)
    
    # -- covariance update --
    cov_t_bar = R + G_t @ cov_t_minus_1 @ G_t.T
    # print()
    # print(G_t)
    # print("\n[PREDICT] mu_t_minus_1")
    # print(mu_t_minus_1)
    # print("\n[PREDICT] cov_t_minus_1")
    # print(cov_t_minus_1)
    # print()
    # print(R)
    # print("\n[PREDICT] cov_t_bar")
    # print(cov_t_bar)
    # print("\n[PREDICT] mu_t_bar")
    # print(mu_t_bar)
    
    return mu_t_bar, cov_t_bar
def correction(mu_t_bar:np.ndarray, cov_t_bar:np.ndarray, z_t:np.ndarray, Q:np.ndarray, obs_book:list):
    mu_t = None
    cov_t = None
    
    for i, obs in enumerate(z_t):
        r = obs[0]
        phi = obs[1]
        j = int(obs[2])
        if j < 0: # outlier
            continue
        
        # -- initialize the landmark with best guess upon first observation --
        if not obs_book[j]:
            obs_book[j] = True
            abs_bearing = wrap_bearing(phi + mu_t_bar[2])
            mu_t_bar[3+2*j] = mu_t_bar[0] + r * np.cos(abs_bearing) # m_j_x
            mu_t_bar[3+2*j+1] = mu_t_bar[1] + r * np.sin(abs_bearing) # m_j_y
            # -- question: should we initialize the covariance as well ? --

            print("\n[CORRECT] abs_bearing {}, x {}, y {}".format(abs_bearing, mu_t_bar[3+2*j], mu_t_bar[3+2*j+1]))
            
        # -- compute predicted measurement --
        delta_x = mu_t_bar[3+2*j] - mu_t_bar[0]
        delta_y = mu_t_bar[3+2*j+1] - mu_t_bar[1]
        q = delta_x**2 + delta_y**2
        sqrt_q = np.sqrt(q)
        z_t_bar = np.array([sqrt_q, wrap_bearing(np.arctan2(delta_y, delta_x) - mu_t_bar[2])])
        # print("\n[CORRECT] z_t_bar")
        # print(z_t_bar)
        
        # print("\n[CORRECT] obs")
        # print(obs)
        
        # -- compute Jacobian H --
        H = np.zeros((2, 2*NUM_LANDMARKS+3))
        one_over_sqrt_q = 1.0 / sqrt_q
        one_over_q = 1.0 / q
        H[0,0] = -delta_x * one_over_sqrt_q
        H[0,1] = -delta_y * one_over_sqrt_q
        H[0,2] = 0
        H[0,3+2*j] = delta_x * one_over_sqrt_q
        H[0,3+2*j+1] = delta_y * one_over_sqrt_q
        H[1,0] = delta_y * one_over_q
        H[1,1] = -delta_x * one_over_q
        H[1,2] = -1
        H[1,3+2*j] = -delta_y * one_over_q
        H[1,3+2*j+1] = delta_x * one_over_q
        # print("\n[CORRECT] H")
        # print(H)
        
        # -- compute Kalman Gain K --
        K = cov_t_bar @ H.T @ np.linalg.inv(H @ cov_t_bar @ H.T + Q)
        # print("\n[CORRECT] K")
        # print(K)
        
        # -- compute updated mean and covariance
        # print(K.shape)
        # print(z_t.shape)
        # print(z_t_bar.shape)
        # print("\n[CORRECT] innovation")
        # print(np.squeeze(obs[:2]) - z_t_bar)
        mu_t_bar = mu_t_bar + K @ (np.squeeze(obs[:2]) - z_t_bar)
        cov_t_bar = (np.eye(STATE_LEN) - K @ H) @ cov_t_bar
        
    mu_t = mu_t_bar
    cov_t = cov_t_bar
    
    return mu_t, cov_t, obs_book
def mahalonobis_matching(z_t:np.ndarray, mu_t_bar:np.ndarray, cov_t_bar:np.ndarray, Q:np.ndarray, obs_book:list, mahalonobis_threshold=0.5):
    assocs = np.zeros((z_t.shape[0]))
    cur_landmark_num = 0
    for is_observed in obs_book:
        if is_observed:
            cur_landmark_num += 1
        else:
            break
    
    for j, z_j in enumerate(z_t): # iterate over observation
        min_m = 1e10
        for i, is_observed in enumerate(obs_book): # iterate over known landmarks
            if not is_observed:
                print(min_m)
                if min_m > mahalonobis_threshold:
                    assocs[j] = cur_landmark_num # new observation
                    print("--> New observation")
                    cur_landmark_num += 1
                # elif min_m > 0.0375: # probably outlier
                elif min_m > 0.06: # probably outlier
                    assocs[j] = -1
                    print("--> Outlier detected")
                else:
                    pass
                    # ok, this is fine
                break
            # -- compute predicted measurement --
            delta_x = mu_t_bar[3+2*i] - mu_t_bar[0]
            delta_y = mu_t_bar[3+2*i+1] - mu_t_bar[1]
            q = delta_x**2 + delta_y**2
            sqrt_q = np.sqrt(q)
            z_t_bar = np.array([sqrt_q, wrap_bearing(np.arctan2(delta_y, delta_x) - mu_t_bar[2])])
            
            # -- compute innovation --
            y_t = (z_j[:2] - z_t_bar)[...,None] # (2, 1)
            
            # -- compute Jacobian H --
            H = np.zeros((2, 2*NUM_LANDMARKS+3))
            one_over_sqrt_q = 1.0 / sqrt_q
            one_over_q = 1.0 / q
            H[0,0] = -delta_x * one_over_sqrt_q
            H[0,1] = -delta_y * one_over_sqrt_q
            H[0,2] = 0
            H[0,3+2*j] = delta_x * one_over_sqrt_q
            H[0,3+2*j+1] = delta_y * one_over_sqrt_q
            H[1,0] = delta_y * one_over_q
            H[1,1] = -delta_x * one_over_q
            H[1,2] = -1
            H[1,3+2*j] = -delta_y * one_over_q
            H[1,3+2*j+1] = delta_x * one_over_q
            
            # -- compute mahalonobis distance --
            S_t = H @ cov_t_bar @ H.T + Q # (2, 2)
            m = np.squeeze(y_t.T @ np.linalg.inv(S_t) @ y_t)
            # print("==> GT {}, candidate {}, mah. distance {}".format(z_j[2], i, m))
            if m < min_m:
                min_m = m
                assocs[j] = i
    return assocs

# -- callbacks --
def odometry_cb(msg:Odometry):
    global lock, prev_state_dr, pose_ekf_pub, pose_dr_pub, prev_odom_time, cur_state_dr, cur_state_ekf, prev_state_ekf, prev_cov, cur_cov, R, cur_state_ekf_bar, cur_cov_bar
    
    if prev_odom_time is None:
        prev_odom_time = msg.header.stamp
        return
    
    # -- construct control command --
    v = msg.twist.twist.linear.x
    w = msg.twist.twist.angular.z
    u = np.array([v, w])
    dt = (msg.header.stamp - prev_odom_time).to_sec()
    prev_odom_time = msg.header.stamp
    
    # -- dead reckoning estimate --
    cur_state_dr = dead_reckoning(prev_state_dr, dt, u)
    prev_state_dr = cur_state_dr
    
    # -- ekf prediction --
    while lock:
        rospy.Rate(1000).sleep()
    lock = True
    cur_state_ekf_bar, cur_cov_bar = prediction(cur_state_ekf_bar, cur_cov_bar, u, dt, R)
    cur_state_ekf = cur_state_ekf_bar
    lock = False
    
    ekf_pose = get_pose_msg(cur_state_ekf, msg.header.stamp)
    pose_ekf_pub.publish(ekf_pose)
    dr_pose = get_pose_msg(cur_state_dr, msg.header.stamp)
    pose_dr_pub.publish(dr_pose)
def landmark_obs_cb(msg:Landmarks):
    global lock, prev_state_dr, prev_odom_time, cur_state_dr, cur_state_ekf, prev_state_ekf, prev_cov, cur_cov, R, cur_state_ekf_bar, cur_cov_bar, Q, obs_book
    
    # -- prepare observation data --
    num_obs = len(msg.landmarks)
    z_t = np.zeros((num_obs, 3)) # (range, bearing, associated landmark index)
    for i, landmark in enumerate(msg.landmarks):
        j = int(landmark.id)
        z_t[i][0] = landmark.range
        z_t[i][1] = landmark.bearing
        # -- groundtruth association --
        z_t[i][2] = j
    
    # -- our data association --
    assocs = mahalonobis_matching(z_t, cur_state_ekf_bar, cur_cov_bar, Q, obs_book, mahalonobis_threshold=0.8)
    
    # -- check association performance --
    for i, pred in enumerate(assocs):
        if int(pred) != int(z_t[i][2]):
            print("==> Wrong association! {} misclassified to be {}".format(int(z_t[i][2]), int(pred)))
        z_t[i][2] = int(pred)
    
    # -- ekf correction --
    while lock:
        rospy.Rate(1000).sleep()
    lock = True
    cur_state_ekf, cur_cov, obs_book = correction(cur_state_ekf_bar, cur_cov_bar, z_t, Q, obs_book)
    cur_state_ekf_bar = cur_state_ekf
    cur_cov_bar = cur_cov
    lock = False
    
def scan_cb(msg:LaserScan):
    # -- we will be using the current state estimate (specifically, the pose), to transform point clouds from ego frame to global frame --
    global cur_state_ekf, rx_cnt
    global p, s, e, landmark_marker_pub
    global scan_republish_pub, landmark_pos_marker_pub
    global scan_rx_cnt, scan, landmark_pos, vis_pubed, dummy_pnt
    global lock, prev_state_dr, prev_odom_time, cur_state_dr, cur_state_ekf, prev_state_ekf, prev_cov, cur_cov, R, cur_state_ekf_bar, cur_cov_bar, Q, obs_book
    global start_time, switch_pose_time, anchor
    scan = msg
    scan.header.frame_id = 'ground_truth_pose'
    scan_republish_pub.publish(scan)
    
    rx_cnt += 1
    if rx_cnt % 3 == 0: # skip 1 sample per 3
        return

    start_t = time.time()
    ranges = np.array(scan.ranges)
    bearings = np.arange(scan.angle_min, scan.angle_max+scan.angle_increment, scan.angle_increment)
    mask = np.isfinite(ranges)
    
    # -- get rid of the NaNs and Infs --
    ranges = ranges[mask==True]
    bearings = bearings[mask==True]
    
    # -- range-bearing data to point cloud --
    x = ranges * np.cos(bearings)
    y = ranges * np.sin(bearings)
    N = len(ranges)
    if N == 0:
        print("--> Empty samples")
        return
    point_cloud = np.stack([x, y], axis=-1) # (N, 2)
    
    # -- do landmark extraction --
    s_, e_ = gao_et_al.compute(point_cloud, ranges, bearings)
    if not (s_ is not None and e_ is not None):
        # print("--> No landmark detection")
        return

    # -- transform the end points from ego-frame to global frame, using the currently estimated position of the robot --
    pose = cur_state_ekf if (time.time() - start_time) > switch_pose_time else cur_state_dr
    s_global = ego2global(pose, s_)
    e_global = ego2global(pose, e_)
    dum = np.array([[1, 0]])
    dummy_pnt = ego2global(pose, dum)
    
    s = s_global
    e = e_global
    # s_global = s
    # e_global = e
    
    ms = (e_global[:,1] - s_global[:,1]) / (e_global[:,0] - s_global[:,0])
    cs = e_global[:,1] - e_global[:,0] * ms
    
    N = len(ms)
    # anchors = np.tile(np.array([[0, 0]]), (N, 1))
    print("--> Anchor is {}".format(anchor))
    anchors = np.tile(anchor[None,...], (N, 1))
    landmark_pos = pnt_on_line_closest_to(anchors, ms, cs)
    
    exe_time = time.time() - start_t
    print("--> Execution time: {}".format(exe_time))
        
    # -- publish visualization marker --
    vis_pubed = False
    
    # -- to common format --
    interface_format = Landmarks()
    for i in range(len(landmark_pos)):
        # if not (i == 0 or i == 3):
        #     continue
        # if not (i == 0 or i == 2):
        #     continue
        if i % 1 != 0:
            continue
        l_x = landmark_pos[i,0]
        l_y = landmark_pos[i,1]
        r_x = pose[0]
        r_y = pose[1]
        r_theta = pose[2]
        landmark = Landmark()
        landmark.id = i
        dx = (l_x - r_x)
        dy = (l_y - r_y)
        landmark.bearing = wrap_bearing(np.arctan2(dy, dx) - r_theta)
        landmark.range = np.sqrt(dx ** 2 + dy ** 2)
        interface_format.landmarks.append(landmark)
    
    # -- prepare observation data --
    num_obs = len(interface_format.landmarks)
    z_t = np.zeros((num_obs, 3)) # (range, bearing, associated landmark index)
    for i, landmark in enumerate(interface_format.landmarks):
        # j = int(landmark.id)
        z_t[i][0] = landmark.range
        z_t[i][1] = landmark.bearing
        # # -- groundtruth association --
        # z_t[i][2] = j
        
    # print(obs_book)
    
    # -- our data association --
    assocs = mahalonobis_matching(z_t, cur_state_ekf_bar, cur_cov_bar, Q, obs_book, mahalonobis_threshold=0.8)
    
    # -- check association performance --
    for i, pred in enumerate(assocs):
        # if int(pred) != int(z_t[i][2]):
        #     print("==> Wrong association! {} misclassified to be {}".format(int(z_t[i][2]), int(pred)))
        z_t[i][2] = int(pred)
    
    # -- ekf correction --
    while lock:
        rospy.Rate(1000).sleep()
    lock = True
    cur_state_ekf, cur_cov, obs_book = correction(cur_state_ekf_bar, cur_cov_bar, z_t, Q, obs_book)
    cur_state_ekf_bar = cur_state_ekf
    cur_cov_bar = cur_cov
    lock = False
    

# -- landmark extraction modules --


def main():
    global pose_dr_pub, path_dr_pub, prev_state_dr, path_dr, path_ekf, cur_state_dr, cur_state_ekf, pose_ekf_pub, path_ekf_pub, prev_cov, cur_cov, R, Q, cur_state_ekf_bar
    global p, s, e, landmark_marker_pub
    global scan_republish_pub, landmark_pos_marker_pub, scan, landmark_pos, vis_pubed
    global start_time, anchor
    start_time = time.time()
    print("==> EKF-SLAM node started")
        
    fig = plt.figure()
    
    np.set_printoptions(precision=3)
    
    rospy.init_node("ekf_slam_node")
    
    # -- initialize ekf slam --
    R[0,0] = var_x
    R[1,1] = var_y
    R[2,2] = var_theta
    Q[0,0] = var_r
    Q[1,1] = var_phi
    cur_cov_bar[:3,:3] = R[:3,:3]
    cur_cov_bar[3,3] = 100
    cur_cov_bar[4,4] = 100
    
    # -- create publishers --
    pose_dr_pub = rospy.Publisher("/dead_reckoning_pose", PoseStamped, queue_size=100)
    path_dr_pub = rospy.Publisher("/dead_reckoning_trajectory", Path, queue_size=1)
    pose_ekf_pub = rospy.Publisher("/ekf_slam_pose", PoseStamped, queue_size=100)
    path_ekf_pub = rospy.Publisher("/ekf_slam_trajectory", Path, queue_size=1)
    landmark_marker_pub = rospy.Publisher("/fitted_landmark_markers", MarkerArray, queue_size=5)
    landmark_pos_marker_pub = rospy.Publisher("/fitted_landmark_position_marker", MarkerArray, queue_size=5)
    scan_republish_pub = rospy.Publisher("/scan_repub", LaserScan, queue_size=10)
    
    # -- create subscribers --
    odom_sub = rospy.Subscriber("/simulated_odometry", Odometry, odometry_cb)
    # landmark_obs_sub = rospy.Subscriber("/landmark_obs", Landmarks, landmark_obs_cb)
    scan_sub = rospy.Subscriber("/scan", LaserScan, scan_cb)
    
    rate = rospy.Rate(500)
    last_vis_pub_time = time.time()
    while not rospy.is_shutdown():
        # -- publish pose and trajectory --
        pose_dr, path_dr = construct_pose_path_msgs(path_dr, cur_state_dr, rospy.Time.now())
        pose_ekf, path_ekf = construct_pose_path_msgs(path_ekf, cur_state_ekf, rospy.Time.now())
        
        path_dr_pub.publish(path_dr)
        path_ekf_pub.publish(path_ekf)
        
        rate.sleep()
        
        if (time.time() - last_vis_pub_time) > 0.1 and scan is not None and landmark_pos is not None and not vis_pubed:
            last_vis_pub_time = time.time()
            vis_markers = MarkerArray()

            for i in range(len(s)):
                # if i > 0:
                #     break
                m = Marker()
                m.header.frame_id = 'world'
                m.header.stamp = scan.header.stamp
                m.type = m.LINE_STRIP
                m.id = i
                m.pose.orientation.w = 1
                m.color.a = 1
                m.color.b = 1
                m.color.g = 0.15
                m.scale.x = 0.08
                start_p = Point()
                if i >= len(s):
                    break
                start_p.x = s[i,0]
                start_p.y = s[i,1]
                end_p = Point()
                end_p.x = e[i,0]
                end_p.y = e[i,1]
                m.points.append(start_p) 
                m.points.append(end_p) 
                m.lifetime = rospy.Duration(0.05)
                # print(m.points)
                vis_markers.markers.append(m)
            landmark_marker_pub.publish(vis_markers)
            
            pos_markers = MarkerArray()
            for i in range(len(landmark_pos)+1):
                if i > len(landmark_pos):
                    break
                x = landmark_pos[i,0] if i != len(landmark_pos) else anchor[0]
                y = landmark_pos[i,1] if i != len(landmark_pos) else anchor[1]
                # x = dummy_pnt[i, 0]
                # y = dummy_pnt[i, 1]
                
                # print("({}, {}): m={}, c={}".format(x, y, ms[i], cs[i]))
                
                m = Marker()
                m.type = Marker().CYLINDER
                m.pose.position.x = x
                m.pose.position.y = y
                m.scale.x = m.scale.y = 0.1 if i != len(landmark_pos) else 0.15
                m.scale.z = 1
                m.color.a = 1
                m.color.r = 0.9 if i != len(landmark_pos) else 0.3
                m.color.g = 0.9 if i != len(landmark_pos) else 1.0
                m.id = i
                m.header.stamp = scan.header.stamp
                m.lifetime = rospy.Duration(0.15) if i != len(landmark_pos) else rospy.Duration(5)
                m.header.frame_id = 'world'
                m.pose.orientation.w = 1
                pos_markers.markers.append(m)
            landmark_pos_marker_pub.publish(pos_markers)
            
            vis_pubed = True
    
if __name__ == "__main__":
    main()