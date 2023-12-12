import numpy as np
import rospy

from landmark_msgs.msg import Landmark, Landmarks
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Pose, PoseStamped

from tf.transformations import quaternion_from_euler

def wrap_bearing(bearing):
    return np.mod(bearing + np.pi, 2 * np.pi) - np.pi

def dead_reckoning(x:np.ndarray, dt:float, u:np.ndarray):
    theta = x[2]
    B = np.array([[np.cos(theta)*dt, 0],[np.sin(theta)*dt, 0],[0, dt]])
    
    new_x = np.eye(3) @ x + B @ u
    new_x[2] = wrap_bearing(new_x[2])
    
    return new_x

def construct_pose_path_msgs(path, cur_state):
    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = "world"
    pose.pose.position.x = cur_state[0]
    pose.pose.position.y = cur_state[1]
    q = quaternion_from_euler(0, 0, cur_state[2])
    pose.pose.orientation.x = q[0]
    pose.pose.orientation.y = q[1]
    pose.pose.orientation.z = q[2]
    pose.pose.orientation.w = q[3]
    
    path.header.frame_id = "world"
    path.header.stamp = rospy.Time.now()
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
NUM_LANDMARKS = 5
STATE_LEN = NUM_LANDMARKS*2+3
MAX_PATH_LEN = 6100
R = np.zeros((STATE_LEN,STATE_LEN))
Q = np.zeros((2,2))

# -- user defined parameters --
var_x = 0.003
var_y = 0.003
var_theta = 0.005
var_r = 0.1
var_phi = 0.05

# -- internal global variables --
lock = False
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

# -- callbacks --
def odometry_cb(msg:Odometry):
    global lock, prev_state_dr, prev_odom_time, cur_state_dr, cur_state_ekf, prev_state_ekf, prev_cov, cur_cov, R, cur_state_ekf_bar, cur_cov_bar
    
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
    # cur_cov_bar = cur_cov
    
def landmark_obs_cb(msg:Landmarks):
    global lock, prev_state_dr, prev_odom_time, cur_state_dr, cur_state_ekf, prev_state_ekf, prev_cov, cur_cov, R, cur_state_ekf_bar, cur_cov_bar, Q, obs_book
    
    # -- prepare observation data --
    num_obs = len(msg.landmarks)
    z_t = np.zeros((num_obs, 3))
    for i, landmark in enumerate(msg.landmarks):
        j = int(landmark.id)
        z_t[i][0] = landmark.range
        z_t[i][1] = landmark.bearing
        z_t[i][2] = j
    
    # -- ekf correction --
    while lock:
        rospy.Rate(1000).sleep()
    lock = True
    cur_state_ekf, cur_cov, obs_book = correction(cur_state_ekf_bar, cur_cov_bar, z_t, Q, obs_book)
    cur_state_ekf_bar = cur_state_ekf
    cur_cov_bar = cur_cov
    lock = False
    
def main():
    global pose_dr_pub, path_dr_pub, prev_state_dr, path_dr, path_ekf, cur_state_dr, cur_state_ekf, pose_ekf_pub, path_ekf_pub, prev_cov, cur_cov, R, Q, cur_state_ekf_bar
    
    print("==> EKF-SLAM node started")
    
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
    pose_dr_pub = rospy.Publisher("/dead_reckoning_pose", PoseStamped, queue_size=1)
    path_dr_pub = rospy.Publisher("/dead_reckoning_trajectory", Path, queue_size=1)
    pose_ekf_pub = rospy.Publisher("/ekf_slam_pose", PoseStamped, queue_size=1)
    path_ekf_pub = rospy.Publisher("/ekf_slam_trajectory", Path, queue_size=1)
    
    # -- create subscribers --
    odom_sub = rospy.Subscriber("/simulated_odometry", Odometry, odometry_cb)
    landmark_obs_sub = rospy.Subscriber("/landmark_obs", Landmarks, landmark_obs_cb)
    
    rate = rospy.Rate(500)
    while not rospy.is_shutdown():
        # -- publish pose and trajectory --
        pose_dr, path_dr = construct_pose_path_msgs(path_dr, cur_state_dr)
        pose_ekf, path_ekf = construct_pose_path_msgs(path_ekf, cur_state_ekf)
        
        path_dr_pub.publish(path_dr)
        pose_dr_pub.publish(pose_dr)
        path_ekf_pub.publish(path_ekf)
        pose_ekf_pub.publish(pose_ekf)
        
        rate.sleep()
    
if __name__ == "__main__":
    main()