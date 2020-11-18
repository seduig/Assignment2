"""
@author rowanms
A Kalman Localiser
@author burbrcjc
Converted to Python
"""

import rospy

from geometry_msgs.msg import (PoseStamped, PoseWithCovarianceStamped, PoseArray,
                               Quaternion,  Transform,  TransformStamped )
from tf.msg import tfMessage
from tf import transformations
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan

import math
import random
import numpy as np
import scipy.linalg
from util import rotateQuaternion, getHeading
import numpy as np
from threading import Lock
import time
from copy import deepcopy
import sensor_model
PI_OVER_TWO = math.pi/2 # For faster calculations
TWO_PI = 2*math.pi # For faster calculations
from nav_msgs.msg import Odometry

class KFLocaliserBase(object):

    INIT_X = 10 		# Initial x location of robot (metres)
    INIT_Y = 5			# Initial y location of robot (metres)
    INIT_Z = 0 			# Initial z location of robot (metres)
    INIT_HEADING = 0 	# Initial orientation of robot (radians)
    
    def __init__(self):
        # ----- Initialise fields
        self.estimatedpose =  PoseWithCovarianceStamped()
	self.estimatedvel = [0,0]
        self.occupancy_map = OccupancyGrid()
        self.tf_message = tfMessage()
        
        self._update_lock =  Lock()
        
        # ----- Parameters
        self.ODOM_ROTATION_NOISE = 0 		# Odometry model rotation noise
        self.ODOM_TRANSLATION_NOISE = 0 	# Odometry x axis (forward) noise
        self.ODOM_DRIFT_NOISE = 0 			# Odometry y axis (side-side) noise
        self.NUMBER_PREDICTED_READINGS = 20 # Number of readings to predict
    
        # ----- Set 'previous' translation to origin
        # ----- All Transforms are given relative to 0,0,0, not in absolute coords.
        self.prev_odom_x  = 0.0 # Previous odometry translation from origin
        self.prev_odom_y = 0.0  # Previous odometry translation from origin
        self.prev_odom_heading = 0.0 # Previous heading from odometry data
        self.last_odom_pose = None
        
        # ----- Request robot's initial odometry values to be recorded in prev_odom
        self.odom_initialised = False
        self.sensor_model_initialised = False

        # ----- Set default initial pose to initial position and orientation.
        self.estimatedpose.pose.pose.position.x = self.INIT_X
        self.estimatedpose.pose.pose.position.y = self.INIT_Y
        self.estimatedpose.pose.pose.position.z = self.INIT_Z
        self.estimatedpose.pose.pose.orientation = rotateQuaternion(Quaternion(w=1.0),
                                                                    self.INIT_HEADING)
        # ----- NOTE: Currently not making use of covariance matrix
        
        self.estimatedpose.header.frame_id = "/map"

        
        # ----- Sensor model
        self.sensor_model =  sensor_model.SensorModel()

    def initialise_kalman(self):
        """
        Called whenever an initialpose message is received (to change the
        starting location of the robot), or a new occupancy_map is received.
        
        :Args:
            | initialpose: the initial pose estimate
        :Return:
            | (list) covariance matrix of odometry
        """
	print('kf base init kalman running')
	sigma = [[2,0.0,0.0],[0.0,1.3,0.0],[0.0,0.0,1]]
	return sigma


    def update_filter(self, scan,odom,prev_scan,prev_pose):
        """
        Called whenever there is a new LaserScan message.
        This calls update methods to do
        kalman filtering, given the map and the LaserScan, and then updates
        Transform tf appropriately.
        
        :Args:
            |  scan (sensor_msgs.msg.LaserScan) latest laser scan to resample
               the kalman filter based on
        """
	print('update filter running')
        if not self.sensor_model_initialised:
            self.sensor_model.set_laser_scan_parameters(self.NUMBER_PREDICTED_READINGS,
                                                        scan.range_max,
                                                        len(scan.ranges),
                                                        scan.angle_min,
                                                        scan.angle_max)
            self.sensor_model_initialised = True
        with self._update_lock:
            t = time.time()

	    self.estimatedpose.pose.pose, self.sigma = self.kalman_predict(self.estimatedpose.pose.pose,self.sigma,odom)
	    self.estimatedpose.pose.pose, self.sigma = self.kalman_update(self.estimatedpose.pose.pose,self.sigma,scan,prev_scan,prev_pose.pose.pose)
            currentTime = rospy.Time.now()
            
            # ----- Given new estimated pose, now work out the new transform
            self.recalculate_transform(currentTime)
            # ----- Insert correct timestamp in estimatedpose,
            # ----- so extending subclasses don't need to worry about this, but can
            # ----- just concentrate on updating actual pose locations

            self.estimatedpose.header.stamp = currentTime

        return time.time()-t

    def kalman_predict(self,mu_pose,sigma,odom):
	print('kalman predict running')
	mu_pose, R = self.predict_from_odom(odom,mu_pose)
	sigma = sigma + R
	return mu_pose, sigma

    def kalman_update(self,mu_pose,sigma,n_scan,p_scan,prev_pose):
	print('kalman update running')
	Q = np.zeros((100,100)) #100x100
	np.fill_diagonal(Q,100)
	
	prev_scan = LaserScan() #copy of lists to avoid accidentally changing their values
	prev_scan.ranges = [i for i in p_scan.ranges]
	scan = LaserScan()
	scan.ranges = [i for i in n_scan.ranges]
	
	ranges_list = list(scan.ranges) #need to iterate over scan.ranges to edit invalid values, this cannot be done on tuples so convert
	p_ranges_list = list(prev_scan.ranges)
	print(len(p_ranges_list))
	for i in range(0,len(ranges_list)):
		if math.isnan(ranges_list[i]): #check if each value is 'nan', which turns up occasionally on real data scans
			ranges_list[i] = -1.0 #if it is, replace it with -1.0 so the program can continue without error
		if math.isnan(p_ranges_list[i]):
			p_ranges_list[i] = -1.0

	scan.ranges = np.array(ranges_list)[0:500:5] #we just use 100 laser readings instead of all 500
	prev_scan.ranges = np.array(p_ranges_list)[0:500:5]

	#put mu and prev_mu in array form for manipulation
	mu = np.array([mu_pose.position.x,mu_pose.position.y,getHeading(mu_pose.orientation)])
	prev_mu = np.array([prev_pose.position.x,prev_pose.position.y,getHeading(prev_pose.orientation)])
	print('prev mu', prev_mu)
	print('start mu', mu)

	ang_inc = math.pi/500 #angle between scan readings
	pred_scan = [0]*100 #predicted scan reading based on current estimated pose
	for j in range(0,500,5):
		obs_bearing = -PI_OVER_TWO + ang_inc*j #angle of inclination
		pred_scan[j/5] = self.sensor_model.calc_map_range(mu[0], mu[1], mu[2] + obs_bearing)

	diff = scan.ranges - pred_scan #difference between observed scan and real, will be used as z - C*mu,  100x1
	scanchange = scan.ranges - prev_scan.ranges #difference between previous scan and current
#	sc_mean =sum(scanchange)/len(scanchange) #average change in scan (unused)
	posechange = prev_mu - mu #change in position
	print('scanchange',scanchange)
	print('posechange',posechange)

	J = np.zeros((100,3)) #Jacobian, will be used as C, 100x3
	for i in range(0,100):
            for j in range(0,3):
		J[i][j] = scanchange[i]/(posechange[j]+0.001) #change of scan relative to change of position
		#+0.001 is to avoid division by 0.
#		Q[i][j] = (scanchange[i]-sc_mean)*(scanchange[j]-sc_mean) #covariance of scan (unused)
#	    for j in range(3,100):
#		Q[i][j] = (scanchange[i]-sc_mean)*(scanchange[j]-sc_mean)
	print('J',J)
	print('Q',Q)
	sigmaJT = np.matmul(sigma,J.T) #so subsequent calculation is faster, 3x100
	K = np.matmul(sigmaJT,np.linalg.inv(np.matmul(J,sigmaJT)+Q)) #Kalman gain, 3x100
	print('K',K)
	print('diff',diff)
	mu = np.add(mu, np.matmul(K,diff)) #update position
	sigma = np.matmul(np.subtract(np.identity(3),np.matmul(K,J)),sigma) #and sigma
	print('end mu', mu)
	print('end sigma',sigma)

	mu_pose.position.x = mu[0] #change new mu to pose form so we can return it
	mu_pose.position.y = mu[1]
	mu_pose.orientation.w = 1.0
	mu_pose.orientation.z = 0.0
	mu_pose.orientation = rotateQuaternion(mu_pose.orientation, mu[2])

	return mu_pose, sigma

    def recalculate_transform(self, currentTime):
        """
        Creates updated transform from /odom to /map given recent odometry and
        laser data.
        
        :Args:
            | currentTime (rospy.Time()): Time stamp for this update
         """
        
        transform = Transform()

        T_est = transformations.quaternion_matrix([self.estimatedpose.pose.pose.orientation.x,
                                                   self.estimatedpose.pose.pose.orientation.y,
                                                   self.estimatedpose.pose.pose.orientation.z,
                                                   self.estimatedpose.pose.pose.orientation.w])
        T_est[0, 3] = self.estimatedpose.pose.pose.position.x
        T_est[1, 3] = self.estimatedpose.pose.pose.position.y
        T_est[2, 3] = self.estimatedpose.pose.pose.position.z
        
        T_odom = transformations.quaternion_matrix([self.last_odom_pose.pose.pose.orientation.x,
                                                   self.last_odom_pose.pose.pose.orientation.y,
                                                   self.last_odom_pose.pose.pose.orientation.z,
                                                   self.last_odom_pose.pose.pose.orientation.w])
        T_odom[0, 3] = self.last_odom_pose.pose.pose.position.x
        T_odom[1, 3] = self.last_odom_pose.pose.pose.position.y
        T_odom[2, 3] = self.last_odom_pose.pose.pose.position.z
        T = np.dot(T_est, np.linalg.inv(T_odom))
        q = transformations.quaternion_from_matrix(T) #[:3, :3])

        transform.translation.x = T[0, 3] 
        transform.translation.y = T[1, 3] 
        transform.translation.z = T[2, 3] 
        transform.rotation.x = q[0]
        transform.rotation.y = q[1]
        transform.rotation.z = q[2]
        transform.rotation.w = q[3]
        

        # ----- Insert new Transform into a TransformStamped object and add to the
        # ----- tf tree
        new_tfstamped = TransformStamped()
        new_tfstamped.child_frame_id = "/odom"
        new_tfstamped.header.frame_id = "/map"
        new_tfstamped.header.stamp = currentTime
        new_tfstamped.transform = transform

        # ----- Add the transform to the list of all transforms
        self.tf_message = tfMessage(transforms=[new_tfstamped])
        

    def predict_from_odom(self, odom, mu):
        """
        Adds the estimated motion from odometry readings to mu.
        
        :Args:
            | odom (nav_msgs.msg.Odometry): Recent Odometry data
        """
	print('kf base predict from odom running')
#        with self._update_lock:
	if 0 == 0:
            t = time.time()
            x = odom.pose.pose.position.x
            y = odom.pose.pose.position.y
            new_heading = getHeading(odom.pose.pose.orientation)
            # ----- On our first run, the incoming translations may not be equal to 
            # ----- zero, so set them appropriately
            if not self.odom_initialised:

                self.prev_odom_x = x
                self.prev_odom_y = y
                self.prev_odom_heading = new_heading
                self.odom_initialised = True

            # ----- Find difference between current and previous translations
            dif_x = x - self.prev_odom_x
            dif_y = y - self.prev_odom_y
            dif_heading = new_heading - self.prev_odom_heading
            if dif_heading >  math.pi:
                dif_heading = (math.pi * 2) - dif_heading
            if dif_heading <  -math.pi:
                dif_heading = (math.pi * 2) + dif_heading
            
            # ----- Update previous pure odometry location (i.e. excluding noise) 
            # ----- with the new translation
            self.prev_odom_x = x
            self.prev_odom_y = y
            self.prev_odom_heading = new_heading
            self.last_odom_pose = odom

            # ----- Find robot's linear forward/backward motion, given the dif_x and 
            # ----- dif_y changes and its orientation
            distance_travelled = math.sqrt(dif_x*dif_x + dif_y*dif_y)
            direction_travelled = math.atan2(dif_y, dif_x)
            temp = abs(new_heading - direction_travelled)
    
            if temp < -PI_OVER_TWO or temp > PI_OVER_TWO:
                # ----- We are going backwards
                distance_travelled = distance_travelled * -1

        rnd = random.normalvariate(0, 1)
	mu.orientation = rotateQuaternion(mu.orientation, dif_heading)
	theta = getHeading(mu.orientation)
	travel_x = distance_travelled * math.cos(theta)
        travel_y = distance_travelled * math.sin(theta)
        mu.position.x = (mu.position.x + travel_x)
	mu.position.y = (mu.position.y + travel_y)

	a = travel_x/2 #dummy variables for finding covariance matrix
	b = travel_y/2
	c = getHeading(rotateQuaternion(mu.orientation, -dif_heading/2))
	ab = a*b
	ac = a*c
	bc = b*c
	R = np.array([[a**2,ab,ac],[ab,b**2,bc],[ac,bc,c**2]]) #covariance matrix

        return mu, R
    
	

    def set_initial_pose(self, pose):
        """ Initialise filter with start pose """
        self.estimatedpose.pose = pose.pose
        # ----- Estimated pose has been set, so we should now reinitialise the 
        # ----- kalman filter
        rospy.loginfo("Got pose. Calling initialise_kalman().")
	self.sigma = self.initialise_kalman()
    
    def set_map(self, occupancy_map):
        """ Set the map for localisation """
        self.occupancy_map = occupancy_map
        self.sensor_model.set_map(occupancy_map)
        # ----- Map has changed, so we should reinitialise the kalman filter
        rospy.loginfo("Kalman filter got map. (Re)initialising.")
	self.sigma = self.initialise_kalman()

