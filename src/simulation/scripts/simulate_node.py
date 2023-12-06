import rospy

import numpy as np
import time

from gazebo_msgs.srv import GetModelState, SpawnModel
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from nav_msgs.msg import Path, Odometry
from landmark_msgs.msg import Landmark, Landmarks

# -- user defined parameters, later will be moved to parameter server --
landmark_positions = [[0.15, 2], [3, 7], [2, 1], [8, 2]]
robot_name = "/"
scan_range = 7.0
scan_coverage = [-np.pi/2, np.pi/2]
obs_noise_var = [0, 0] # range, bearing
odom_noise_var = [0.00001, 0.00001] # linear_vel, angular_vel
obs_pub_rate = 10
pose_pub_rate = 100
odom_pub_rate = 100
cylindrical_landmark_template_urdf = "/home/ikemura/KTH/Research/EKF-SLAM-with-Robust-Landmark-Identification-using-2D-LiDAR/ws/src/simulation/urdf/cylindrical_landmark_template.urdf"

# -- internal constant parameters --
OPERATION_RATE = 125
LANDMARK_RADIUS = 0.125 # m
LANDMARK_HEIGHT = 1.8 # m
MAX_TRAJ_LENGTH = 6100

# -- internal global variables --
robot_pose = PoseStamped()
robot_trajectory = Path()
simulated_odometry = Odometry()
last_pose_pub_time = time.time()
last_obs_pub_time = time.time()
last_gt_landmarks_pub_time = time.time()
last_odom_pub_time = time.time()

# -- helper functions --
def generate_landmarks(urdf_path, landmark_positions):
    urdf_template = open(urdf_path, "r").read()
    urdf_template = urdf_template.replace("RADIUS", str(LANDMARK_RADIUS))
    urdf_template = urdf_template.replace("HEIGHT", str(LANDMARK_HEIGHT))
    
    gt_landmarks = Landmarks()
    gt_landmarks.header.frame_id = "world"
    
    for i, l_pos in enumerate(landmark_positions):
        x, y = l_pos
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = 0
        
        landmark = Landmark()
        landmark.header.frame_id = "world"
        landmark.x = x
        landmark.y = y
        landmark.id = i
        gt_landmarks.landmarks.append(landmark)
        
        rospy.wait_for_service("/gazebo/spawn_urdf_model")
        
        try:
            spawn_landmark = rospy.ServiceProxy("/gazebo/spawn_urdf_model", SpawnModel)
            response = spawn_landmark("landmark_{}".format(i), urdf_template, "landmarks", pose, "")
            
            if not response.success:
                print("==> Landmark spawning at {} failed".format(i))
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
    return gt_landmarks
def get_euler_from_quaternion(quaternion):
    w = quaternion.w
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = np.sqrt(1 + 2 * (w * y - x * z))
    cosp = np.sqrt(1 - 2 * (w * y - x * z))
    pitch = 2 * np.arctan2(sinp, cosp) - np.pi / 2

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw
def wrap_bearing(bearing):
    return np.mod(bearing + np.pi, 2 * np.pi) - np.pi

def main():
    global last_pose_pub_time, last_obs_pub_time, robot_trajectory, last_gt_landmarks_pub_time, last_odom_pub_time
    rospy.init_node("simulate_node")
    
    print("==> Simulate node started")
    
    # -- define publishers --
    robot_trajectory_pub = rospy.Publisher("/ground_truth_trajectory", Path, queue_size=1)
    robot_pose_pub = rospy.Publisher("/ground_truth_pose", PoseStamped, queue_size=1)
    observation_pub = rospy.Publisher("/landmark_obs", Landmarks, queue_size=1)
    true_landmark_pub = rospy.Publisher("/ground_truth_landmarks", Landmarks, queue_size=1)
    simulated_odom_pub = rospy.Publisher("/simulated_odometry", Odometry, queue_size=1)
    
    # -- create landmarks at specified positions --
    gt_landmarks = generate_landmarks(cylindrical_landmark_template_urdf, landmark_positions)
        
    rate = rospy.Rate(OPERATION_RATE)
    while not rospy.is_shutdown():
        rate.sleep()
        
        # -- get robot pose from Gazebo --
        rospy.wait_for_service("/gazebo/get_model_state")
        try:
            get_robot_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
            response = get_robot_state(robot_name, "")
            if not response.success:
                print("==> Get robot state failed")
            else:
                # -- compute odometry --
                simulated_odometry.header.stamp = response.header.stamp
                simulated_odometry.header.frame_id = "world"
                simulated_odometry.child_frame_id = "base_footprint"
                dt = (simulated_odometry.header.stamp - robot_pose.header.stamp).to_sec()
                simulated_odometry.twist.twist.angular.x = simulated_odometry.twist.twist.angular.y = 0
                simulated_odometry.twist.twist.angular.z = (get_euler_from_quaternion(response.pose.orientation)[2] - get_euler_from_quaternion(robot_pose.pose.orientation)[2]) / dt
                simulated_odometry.twist.twist.linear.z = simulated_odometry.twist.twist.linear.y = 0
                simulated_odometry.twist.twist.linear.x = np.sqrt((response.pose.position.x - robot_pose.pose.position.x)**2 + (response.pose.position.y - robot_pose.pose.position.y)**2) / dt
                # -- add noises --
                simulated_odometry.twist.twist.angular.z += np.random.normal(0, odom_noise_var[1], 1)
                simulated_odometry.twist.twist.linear.x += np.random.normal(0, odom_noise_var[0], 1)
                
                # -- store pose --
                robot_pose.pose = response.pose
                robot_trajectory.header.frame_id = robot_pose.header.frame_id = "world"
                robot_trajectory.header.stamp = robot_pose.header.stamp = response.header.stamp
                
                # -- store trajectory --
                local_pose = PoseStamped()
                local_pose.pose = robot_pose.pose
                local_pose.header.frame_id = "base_footprint"
                robot_trajectory.poses.append(local_pose)

                if len(robot_trajectory.poses) > MAX_TRAJ_LENGTH:
                    robot_trajectory.poses.pop(0)
        except rospy.ServiceException as e:
            print("==> Get robot state failed")
            
        # -- calculate observations --
        obs = Landmarks()
        obs.header.stamp = rospy.Time.now()
        robot_yaw = get_euler_from_quaternion(robot_pose.pose.orientation)[2]
        for i, l_pos in enumerate(landmark_positions):
            x, y = l_pos
            x_diff = x - robot_pose.pose.position.x
            y_diff = y - robot_pose.pose.position.y
            
            range = np.sqrt(x_diff**2+y_diff**2)
            bearing = wrap_bearing(np.arctan2(y_diff, x_diff) - robot_yaw)
            
            # -- check if the landmark is observable by our virtual LiDAR --
            if range > scan_range or bearing < scan_coverage[0] or bearing > scan_coverage[1]:
                continue
            
            this_obs = Landmark()
            this_obs.range = range
            this_obs.bearing = bearing
            this_obs.id = i
            obs.landmarks.append(this_obs)
            
        # -- publish robot pose, trajectory as well as observations --
        if (time.time() - last_obs_pub_time) >= (1.0/obs_pub_rate):
            last_obs_pub_time = time.time()
            observation_pub.publish(obs)
            
        if (time.time() - last_pose_pub_time) >= (1.0/pose_pub_rate):
            last_pose_pub_time = time.time()
            robot_pose_pub.publish(robot_pose)
            robot_trajectory_pub.publish(robot_trajectory)
            
        if (time.time() - last_gt_landmarks_pub_time) >= 1:
            last_gt_landmarks_pub_time = time.time()
            true_landmark_pub.publish(gt_landmarks)
            
        if (time.time() - last_odom_pub_time) >= (1.0/odom_pub_rate):
            simulated_odom_pub.publish(simulated_odometry)

if __name__ == "__main__":
    main()