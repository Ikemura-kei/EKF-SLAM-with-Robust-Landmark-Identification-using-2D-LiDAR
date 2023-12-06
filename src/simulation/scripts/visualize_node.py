import rospy

import matplotlib.pyplot as plt
import os

from landmark_msgs.msg import Landmark, Landmarks
from geometry_msgs.msg import PoseStamped, Pose
from visualization_msgs.msg import Marker, MarkerArray

# -- user defined parameters --
ground_truth_pose_topic = "/ground_truth_pose"
landmark_obs_topic = "/landmark_obs"
ground_truth_landmarks_topic = "/ground_truth_landmarks"

# -- internal variables --
landmarks = Landmarks()
gt_landmarks = Landmarks()
gt_pose = PoseStamped()
gt_pose_marker_pub = None
gt_landmarks_pub = None

# -- other helper functions --
def pose2marker(pose:PoseStamped, scales=[0.65, 0.15, 0.15], shape=Marker.ARROW, color=[22,228,250]):
    marker = Marker()
    marker.header = pose.header
    marker.ns = ""
    marker.id = 0
    marker.type = shape
    marker.action = Marker.MODIFY
    marker.pose = pose.pose
    marker.scale.x = scales[0]
    marker.scale.y = scales[1]
    marker.scale.z = scales[2]
    marker.color.r = color[0]/255.0
    marker.color.g = color[1]/255.0
    marker.color.b = color[2]/255.0
    marker.color.a = 1.0
    return marker

# -- callbacks --
def landmarks_cb(msg):
    global landmarks
    landmarks = msg
    
def gt_pose_cb(msg):
    global gt_pose, gt_pose_marker_pub
    gt_pose = msg
    gt_pose_marker = pose2marker(gt_pose)
    gt_pose_marker_pub.publish(gt_pose_marker)
    
def gt_landmarks_cb(msg):
    global gt_landmarks, gt_landmarks_pub
    gt_landmarks = msg
    
    now = rospy.Time.now()
    marker_array = MarkerArray()
    for landmark in gt_landmarks.landmarks:
        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.header.stamp = now
        pose.pose.position.x = landmark.x
        pose.pose.position.y = landmark.y
        pose.pose.orientation.w = 1
        pose.pose.orientation.x = pose.pose.orientation.y = pose.pose.orientation.z = 0
        
        marker = pose2marker(pose, shape=Marker.CYLINDER, scales=[0.21,0.21,0.8], color=[230,3,20])
        marker.id = landmark.id
        marker_array.markers.append(marker)
        
    gt_landmarks_pub.publish(marker_array)

def main():
    global gt_pose_marker_pub, gt_landmarks_pub
    print("==> Visualize node started")
    
    rospy.init_node("visualize_node")
    
    # -- create publishers --
    gt_pose_marker_pub = rospy.Publisher("/ground_truth_pose_marker", Marker, queue_size=1)
    gt_landmarks_pub = rospy.Publisher("/ground_truth_landmarks_marker", MarkerArray, queue_size=1)
    
    # -- create subscribers --
    landmarks_sub = rospy.Subscriber(landmark_obs_topic, Landmarks, landmarks_cb)
    landmarks_sub = rospy.Subscriber(ground_truth_landmarks_topic, Landmarks, gt_landmarks_cb)
    gt_pose_sub = rospy.Subscriber(ground_truth_pose_topic, PoseStamped, gt_pose_cb)
    
    rate = rospy.Rate(1000)
    while not rospy.is_shutdown():
        rate.sleep()
    
if __name__ == "__main__":
    main()