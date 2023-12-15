import rospy
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
import numpy as np
import matplotlib.pyplot as plt

# -- internal global variables --
gt_poses = []
ekf_poses = []
odom_poses = []

# -- helper functions --
def get_euler_from_quaternion(quaternion:Quaternion):
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

def gt_pose_cb(msg:PoseStamped):
    global gt_poses, ekf_poses
    
    x = msg.pose.position.x
    y = msg.pose.position.y
    theta = get_euler_from_quaternion(msg.pose.orientation)[2]
    stamp = msg.header.stamp.to_sec()
    
    pose = np.array([x, y, theta, stamp])
    gt_poses.append(pose)

def ekf_pose_cb(msg:PoseStamped):
    global ekf_poses
    
    x = msg.pose.position.x
    y = msg.pose.position.y
    theta = get_euler_from_quaternion(msg.pose.orientation)[2]
    stamp = msg.header.stamp.to_sec()
    
    pose = np.array([x, y, theta, stamp])
    ekf_poses.append(pose)

def odom_pose_cb(msg:PoseStamped):
    global odom_poses
    
    x = msg.pose.position.x
    y = msg.pose.position.y
    theta = get_euler_from_quaternion(msg.pose.orientation)[2]
    stamp = msg.header.stamp.to_sec()
    
    pose = np.array([x, y, theta, stamp])
    odom_poses.append(pose)

def main():
    global gt_poses, ekf_poses, odom_poses
    rospy.init_node("evaluate_node")
    
    # -- create subscribers --
    gt_pose_sub = rospy.Subscriber("/ground_truth_pose", PoseStamped, gt_pose_cb)
    ekf_pose_sub = rospy.Subscriber("/ekf_slam_pose", PoseStamped, ekf_pose_cb)
    odom_pose_sub = rospy.Subscriber("dead_reckoning_pose", PoseStamped, odom_pose_cb)
    
    HZ = 250
    rate = rospy.Rate(HZ)
    cnt = 0
    last_len = 0
    STOP_TREHSHOLD = 2.5 * HZ # specify seconds but show as number of iterations
    while not rospy.is_shutdown():
        rate.sleep()
        
        if len(ekf_poses) == last_len:
            cnt += 1
        else:
            cnt = 0
            
        last_len = len(ekf_poses)
        
        if cnt >= STOP_TREHSHOLD:
            break
    
    gt = np.array(gt_poses)
    ekf = np.array(ekf_poses)
    dr = np.array(odom_poses)
    
    diff = gt[:,-1] - ekf[0,-1]
    start_idx_1 = np.argmax(diff>=0)
    
    diff = gt[:,-1] - dr[0,-1]
    start_idx_2 = np.argmax(diff>=0)
    
    if start_idx_1 > start_idx_2:
        gt = gt[start_idx_1:]
        dr = dr[start_idx_1-start_idx_2:]
    else:
        gt = gt[start_idx_2:]
        ekf = ekf[start_idx_2-start_idx_1:]
        
    len_array = np.array([len(gt), len(ekf), len(dr)])
    L = len_array[np.argmin(len_array)]
    gt = gt[:L,:3]
    ekf = ekf[:L,:3]
    dr = dr[:L,:3]
    
    ekf_err = gt - ekf
    dr_err = gt - dr
    
    ekf_err[:,-1] = wrap_bearing(ekf_err[:,-1])
    dr_err[:,-1] = wrap_bearing(dr_err[:,-1])
    
    ekf_err = np.abs(ekf_err)
    dr_err = np.abs(dr_err)
    
    t = np.arange(0,L,1)
    fig = plt.figure()
    
    ax = fig.add_subplot(3,1,1)
    ax.set_title("Error x")
    ax.plot(t, ekf_err[:,0], "r", label='EKF-SLAM')
    ax.plot(t, dr_err[:,0], "g", label='Dead Reckoning')
    
    ax = fig.add_subplot(3,1,2)
    ax.set_title("Error y")
    ax.plot(t, ekf_err[:,1], "r", label='EKF-SLAM')
    ax.plot(t, dr_err[:,1], "g", label='Dead Reckoning')
    
    ax = fig.add_subplot(3,1,3)
    ax.set_title("Error theta")
    ax.plot(t, ekf_err[:,2], "r", label='EKF-SLAM')
    ax.plot(t, dr_err[:,2], "g", label='Dead Reckoning')
    
    plt.show()
    
    tmp = np.sum((ekf_err[:,:2] ** 2), axis=-1)
    ATE_pos_ekf = np.mean(tmp)
    
    tmp = np.sum((dr_err[:,:2] ** 2), axis=-1)
    ATE_pos_dr = np.mean(tmp)
    
    ATE_rot_ekf = np.mean(ekf_err[:,2])
    ATE_rot_dr = np.mean(dr_err[:,2])
    
    print("ATE_pos_ekf {}, ATE_pos_dr {}, ATE_rot_ekf {}, ATE_rot_dr {}".format(ATE_pos_ekf, ATE_pos_dr, ATE_rot_ekf, ATE_rot_dr))
    
if __name__ == "__main__":
    main()