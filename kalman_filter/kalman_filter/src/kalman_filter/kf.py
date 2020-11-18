#!/usr/bin/python
from geometry_msgs.msg import Pose, PoseArray, Quaternion
from sensor_msgs.msg import LaserScan
from kf_base import KFLocaliserBase
import math
import rospy
import numpy as np

from nav_msgs.msg import Odometry

from util import rotateQuaternion, getHeading, multiply_quaternions
from random import random, gauss
import sensor_model

from time import time, sleep


class KFLocaliser(KFLocaliserBase):
       
    def __init__(self):
        # ----- Call the superclass constructor
        super(KFLocaliser, self).__init__()
        
        # ----- Set motion model parameters
	self.ODOM_ROTATION_NOISE = 0.05 # Odometry model rotation noise
	self.ODOM_TRANSLATION_NOISE = 0.1 # Odometry model x axis (forward) noise
	self.ODOM_DRIFT_NOISE = 0.1 # Odometry model y axis (side-to-side) noise

        # ----- Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 20     # Number of readings to predict
  



