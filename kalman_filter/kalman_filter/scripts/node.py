#!/usr/bin/python

"""
This is the main entry point for the kalman filter exercise node. It
subscribes to laser, map, and odometry and creates an instance of
kf.KFLocaliser() to do the localisation.
"""

import rospy
import kalman_filter.kf
from kalman_filter.util import *

from geometry_msgs.msg import ( PoseStamped, PoseWithCovarianceStamped,
                                PoseArray, Quaternion )
from tf.msg import tfMessage
import tf 
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
import kalman_filter
from threading import Lock
import numpy as np

import sys
from copy import deepcopy

class KalmanFilterLocalisationNode(object):
    def __init__(self):
        # ----- Minimum change (m/radians) before publishing new  pose
        self._PUBLISH_DELTA = rospy.get_param("publish_delta", 0.1)  
        
        self._kalman_filter = kalman_filter.kf.KFLocaliser()

        self._latest_scan = None
	self._prev_scan = None
        self._last_published_pose = None
        self._initial_pose_received = False

        self._pose_publisher = rospy.Publisher("/estimatedpose", PoseStamped)
        self._amcl_pose_publisher = rospy.Publisher("/amcl_pose",
                                                    PoseWithCovarianceStamped)

        self._tf_publisher = rospy.Publisher("/tf", tfMessage)

        rospy.loginfo("Waiting for a map...")
        try:
            occupancy_map = rospy.wait_for_message("/map", OccupancyGrid, 20)
        except:
            rospy.logerr("Problem getting a map. Check that you have a map_server"
                     " running: rosrun map_server map_server <mapname> " )
            sys.exit(1)
        rospy.loginfo("Map received. %d X %d, %f px/m." %
                      (occupancy_map.info.width, occupancy_map.info.height,
                       occupancy_map.info.resolution))
        self._kalman_filter.set_map(occupancy_map)
        
        self._laser_subscriber = rospy.Subscriber("/base_scan", LaserScan,
                                                  self._laser_callback,
                                                  queue_size=1)
        self._initial_pose_subscriber = rospy.Subscriber("/initialpose",
                                                         PoseWithCovarianceStamped,
                                                         self._initial_pose_callback)
        self._odometry_subscriber = rospy.Subscriber("/odom", Odometry,
                                                     self._odometry_callback,
                                                     queue_size=1)

    def _initial_pose_callback(self, pose):
        """ called when RViz sends a user supplied initial pose estimate """
        self._kalman_filter.set_initial_pose(pose)
        self._last_published_pose = deepcopy(self._kalman_filter.estimatedpose)
        self._initial_pose_received = True
	self._amcl_pose_publisher.publish(self._kalman_filter.estimatedpose)
        estimatedpose =  PoseStamped()
        estimatedpose.pose = self._kalman_filter.estimatedpose.pose.pose
        estimatedpose.header.frame_id = "map"
        self._pose_publisher.publish(estimatedpose)

    def _odometry_callback(self, odometry):
        """
        Odometry received. If the filter is initialised then execute
        a filter predict step with odeometry followed by an update step using
        the latest laser.
        """
        if self._initial_pose_received:
	    if self._prev_scan == None:
		self._prev_scan = deepcopy(self._latest_scan)
            t_filter = self._kalman_filter.update_filter(self._latest_scan,odometry,self._prev_scan,self._last_published_pose)
	    self.estimatedpose = self._kalman_filter.estimatedpose.pose.pose
 	    self._last_published_pose.pose.pose = deepcopy(self.estimatedpose)
            rospy.loginfo("Kalman update: %fs"%t_filter)
    
    def _laser_callback(self, scan):
        """
        Laser received. Store a ref to the latest scan. If robot has moved
        much, republish the latest pose to update RViz
        """
	if self._latest_scan == None:
	    self._prev_scan = scan
	else:
	    self._prev_scan = deepcopy(self._latest_scan)
        self._latest_scan = scan
        if self._initial_pose_received:
            if  self._sufficientMovementDetected(self._kalman_filter.estimatedpose):
                # ----- Publish the new pose
		print('publishing estimatedpose:')
                self._amcl_pose_publisher.publish(self._kalman_filter.estimatedpose)
                estimatedpose =  PoseStamped()
                estimatedpose.pose = self._kalman_filter.estimatedpose.pose.pose
                estimatedpose.header.frame_id = "map"
                self._pose_publisher.publish(estimatedpose)
                
                # ----- Update record of previously-published pose
                self._last_published_pose = deepcopy(self._kalman_filter.estimatedpose)
        
                # ----- Get updated transform and publish it
                self._tf_publisher.publish(self._kalman_filter.tf_message)
    
    def _sufficientMovementDetected(self, latest_pose):
        """
        Compares the last published pose to the current pose. Returns true
        if movement is more the self._PUBLISH_DELTA
        """
        # ----- Check that minimum required amount of movement has occurred before re-publishing
        latest_x = latest_pose.pose.pose.position.x
        latest_y = latest_pose.pose.pose.position.y
        prev_x = self._last_published_pose.pose.pose.position.x
        prev_y = self._last_published_pose.pose.pose.position.y
        location_delta = abs(latest_x - prev_x) + abs(latest_y - prev_y)

        # ----- Also check for difference in orientation: Take a zero-quaternion,
        # ----- rotate forward by latest_rot, and rotate back by prev_rot, to get difference)
        latest_rot = latest_pose.pose.pose.orientation
        prev_rot = self._last_published_pose.pose.pose.orientation

        q = rotateQuaternion(Quaternion(w=1.0),
                             getHeading(latest_rot))   # Rotate forward
        q = rotateQuaternion(q, -getHeading(prev_rot)) # Rotate backward
        heading_delta = abs(getHeading(q))
        #rospy.loginfo("Moved by %f"%location_delta)
        return (location_delta > self._PUBLISH_DELTA or
                heading_delta > self._PUBLISH_DELTA)

if __name__ == '__main__':
    # --- Main Program  ---
    rospy.init_node("kalman_filter")
    node = KalmanFilterLocalisationNode()
    rospy.spin()
