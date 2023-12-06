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
MAX_PATH_LEN = 6100
NUM_LANDMARKS = 10

# -- internal global variables --
prev_state_dr = cur_state_dr = np.array([0, 0, 0])
path_dr = Path()
path_dr_pub = None
pose_dr_pub = None
prev_state_ekf = cur_state_ekf = np.zeros(NUM_LANDMARKS*2+3)
path_ekf = Path()
path_ekf_pub = None
pose_ekf_pub = None
prev_odom_time = None

# -- callbacks --
def odometry_cb(msg:Odometry):
    global prev_state_dr, prev_odom_time, cur_state_dr, cur_state_ekf, prev_state_ekf
    
    if prev_odom_time is None:
        prev_odom_time = msg.header.stamp
        return
    
    # -- construct control command --
    v = msg.twist.twist.linear.x
    w = msg.twist.twist.angular.z
    u = np.array([v, w])
    dt = (msg.header.stamp - prev_odom_time).to_sec()
    
    # -- dead reckoning estimate --
    cur_state_dr = dead_reckoning(prev_state_dr, dt, u)
    
    # -- ekf update --
    # -- 1. prediction --
    cur_state_ekf[0] = prev_state_ekf[0] + 0.004 * np.random.rand(1)
    cur_state_ekf[1] = prev_state_ekf[0] + 0.0045 * np.random.rand(1)
    # -- 2. compute innovation --
    # -- 3. compute kalman gain --
    # -- 4. update --
    
    prev_odom_time = msg.header.stamp
    prev_state_dr = cur_state_dr
    prev_state_ekf = cur_state_ekf

def main():
    global pose_dr_pub, path_dr_pub, prev_state_dr, path_dr, path_ekf, cur_state_dr, cur_state_ekf, pose_ekf_pub, path_ekf_pub
    
    print("==> EKF-SLAM node started")
    
    rospy.init_node("ekf_slam_node")
    
    # -- create publishers --
    pose_dr_pub = rospy.Publisher("/dead_reckoning_pose", PoseStamped, queue_size=1)
    path_dr_pub = rospy.Publisher("/dead_reckoning_trajectory", Path, queue_size=1)
    pose_ekf_pub = rospy.Publisher("/ekf_slam_pose", PoseStamped, queue_size=1)
    path_ekf_pub = rospy.Publisher("/ekf_slam_trajectory", Path, queue_size=1)
    
    # -- create subscribers --
    odom_sub = rospy.Subscriber("/simulated_odometry", Odometry, odometry_cb)
    
    rate = rospy.Rate(200)
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