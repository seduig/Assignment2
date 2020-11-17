"""
@author rowanms
An abstract Localiser which needs to be extended as PFLocaliser
before PFLocalisationNode will work.
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
import sensor_model
PI_OVER_TWO = math.pi/2 # For faster calculations
TWO_PI = 2*math.pi # For faster calculations
from nav_msgs.msg import Odometry

class PFLocaliserBase(object):

    INIT_X = 10 		# Initial x location of robot (metres)
    INIT_Y = 5			# Initial y location of robot (metres)
    INIT_Z = 0 			# Initial z location of robot (metres)
    INIT_HEADING = 0 	# Initial orientation of robot (radians)
    
    def __init__(self):
        # ----- Initialise fields
        self.estimatedpose =  PoseWithCovarianceStamped()
	self.estimatedvel = [0,0]
        self.occupancy_map = OccupancyGrid()
#        self.particlecloud =  PoseArray()
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
#        self.particlecloud.header.frame_id = "/map"
        
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
	print('pf base init kalman running')
#	sigma = [[0.25,0.0,0.0],[0.0,0.25,0.0],[0.0,0.0,0.06853892326654787]]
	sigma = [[2,0.0,0.0],[0.0,1.3,0.0],[0.0,0.0,1]]
	return sigma


    def update_filter(self, scan,odom):
        """
        Called whenever there is a new LaserScan message.
        This calls update methods (implemented by subclass) to do actual
        particle filtering, given the map and the LaserScan, and then updates
        Transform tf appropriately.
        
        :Args:
            |  scan (sensor_msgs.msg.LaserScan) latest laser scan to resample
               the particle filter based on
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
            # ----- Call user-implemented particle filter update method
#            self.update_particle_cloud(scan)
	    self.estimatedpose.pose.pose, self.sigma = self.kalman_predict(self.estimatedpose.pose.pose,self.sigma,odom)
	    self.estimatedpose.pose.pose, self.sigma = self.ku2(self.estimatedpose.pose.pose,self.sigma,scan)
#	    self.estimatedpose.pose = self.kalman_predict(self.estimatedpose.pose.pose,self.estimatedpose.pose.covariance,odom)
#            self.particlecloud.header.frame_id = "/map"
#            self.estimatedpose.pose.pose = self.estimate_pose(scan)
            currentTime = rospy.Time.now()
            
            # ----- Given new estimated pose, now work out the new transform
            self.recalculate_transform(currentTime)
            # ----- Insert correct timestamp in particlecloud and estimatedpose,
            # ----- so extending subclasses don't need to worry about this, but can
            # ----- just concentrate on updating actual particle and pose locations
#            self.particlecloud.header.stamp = currentTime
            self.estimatedpose.header.stamp = currentTime

        return time.time()-t

    def kalman_predict(self,mu_pose,sigma,odom):
	g = random.gauss(0,4)
	h = random.gauss(0,4)
	j = random.gauss(0,4)
#	R = np.array([[random.gauss(0,6),g,h],[g,random.gauss(0,3),j],[h,j,random.gauss(0,0.4)]])
	R = np.array([[0.2,0.0,0.0],[0.0,0.2,0.0],[0,0,0.2]])
	mu_pose = self.predict_from_odom(odom,mu_pose)
	sigma = sigma + R
	return mu_pose, sigma

    def ku2(self,mu_pose,sigma,scan):
	mu = np.array([mu_pose.position.x, mu_pose.position.y, getHeading(mu_pose.orientation)])
	ang_inc = math.pi/500.0
	pred_minj = 0
	obs_minj = 0
	pred_scan = [0]*500
	Q = np.zeros((2,2))
	np.fill_diagonal(Q,100)
	for j in range(0,500):
		obs_bearing = -PI_OVER_TWO + ang_inc*j
		pred_scan[j] = self.sensor_model.calc_map_range(mu[0], mu[1], mu[2] + obs_bearing)
		if pred_scan[j] < pred_scan[pred_minj] and pred_scan[j] > 0:
			pred_minj = j
		if scan.ranges[j] < scan.ranges[obs_minj] and scan.ranges[j] > 0:
			obs_minj = j
#	print('predscan',pred_scan)
#	print('scan.ranges',scan.ranges)
	pred_ang = -PI_OVER_TWO + ang_inc*pred_minj
	obs_ang = -PI_OVER_TWO + ang_inc*obs_minj
	print('pred_ang',pred_ang)
	print('obs_ang',obs_ang)
	pred_point = np.array([0.0,0.0]) #2x1
	obs_point = np.array([0.0,0.0]) #2x1
	print('pred_minj',pred_minj)
	print('obs_minj',obs_minj)
	print('minpredrange',pred_scan[pred_minj])
	print('minobsrange',scan.ranges[obs_minj])
    	pred_point[0] = (-2*(pred_minj>249)+1)*pred_scan[pred_minj]*math.cos(pred_ang)
        pred_point[1] = pred_scan[pred_minj]*math.sin(pred_ang)
    	obs_point[0] = (-2*(obs_minj>249)+1)*scan.ranges[obs_minj]*math.cos(obs_ang)
        obs_point[1] = scan.ranges[obs_minj]*math.sin(obs_ang)
	print('obspoint',obs_point)
	print('predpoint',pred_point)
	C = np.array([[pred_point[0]/mu[0],0,0],[0,0,pred_point[1]/mu[1]]]) #2x3
	sigmaCT = np.matmul(sigma,C.T) #3x2
	K = np.matmul(sigmaCT,np.linalg.inv(np.add(np.matmul(C,sigmaCT),Q))) #3x2
	print('C',C)
	print('K',K)
	print('start mu',mu)
	print('start sigma',sigma)
	mu = np.add(mu, np.matmul(K,(obs_point - pred_point)))
	sigma = np.matmul(np.subtract(np.identity(3),np.matmul(K,C)),sigma)
	print('end mu',mu)
	print('end sigma',sigma)
	mu_pose.position.x = mu[0]
	mu_pose.position.y = mu[1]
	mu_pose.orientation.w = 1.0
	mu_pose.orientation.z = 0.0
	mu_pose.orientation = rotateQuaternion(mu_pose.orientation, mu[2])
	return mu_pose,sigma

    def ku1(self,mu_pose,sigma,scan):
	mu = np.array([mu_pose.position.x, mu_pose.position.y, getHeading(mu_pose.orientation)])
	print('mu start',mu)
	pred_scan = np.array([0.0]*500)
	pred_rel_points = np.array([[0.0,0.0,0.0]]*500) #500x3
	obs_rel_points = np.array([[0.0,0.0,0.0]]*500) #500x3
	pred_vectors = np.array([[0.0,0.0]]*499) #499x2
	obs_vectors = np.array([[0.0,0.0]]*499) #499x2
	ang_inc = math.pi/500.0
	for i in range(0,500):
	    theta = ang_inc*i
	    pred_scan[i] = self.sensor_model.calc_map_range(mu[0],mu[1],mu[2]-PI_OVER_TWO+theta)
    	    pred_rel_points[i][0] = (-2*(i>249)+1)*pred_scan[i]*math.cos(theta)
	    pred_rel_points[i][1] = pred_scan[i]*math.sin(theta)
	    obs_rel_points[i][0] = (-2*(i>249)+1)*scan.ranges[i]*math.cos(theta)
	    obs_rel_points[i][1] = scan.ranges[i]*math.sin(theta)
	    if i > 0:
		pred_vectors[i-1] = pred_rel_points[i][0:2] - pred_rel_points[i-1][0:2]
		obs_vectors[i-1] = obs_rel_points[i][0:2] - obs_rel_points[i-1][0:2]



	cost = np.array([0]*250)
	minj = 0
	for j in range(250): #starting point of regions of pred_vectors
	    cost[j] = sum(sum(abs(pred_vectors[125:375] - obs_vectors[j:j+250])))
	    if cost[j] < cost[minj]:
		minj = j


	pred_mes = pred_rel_points[125:376] #251x3
	obs_mes = obs_rel_points[minj:minj+251] #251x3
#	obs_mes[:,2] = np.array([-PI_OVER_TWO + ang_inc*minj]).T

	mu_mat = np.array([[mu[0],1,0],[mu[1],0,1],[mu[2],1,0]]) #3x3

	Q = np.zeros((251,251))
	np.fill_diagonal(Q,10000000)
	C = np.array(np.matmul(pred_mes,np.linalg.inv(mu_mat))) #251x3
#	C=np.eye(251,3)
#	C = np.array(np.matmul(dummy,np.linalg.inv(mu_mat))) #753x3
	sigCT = np.matmul(sigma,np.transpose(C)) #3x251
	K = np.matmul(sigCT,np.linalg.inv(np.matmul(C,sigCT)+Q)) #3x251
#	K = np.matmul(sigma,np.matmul(C.T,np.linalg.inv(np.add(np.matmul(C,np.matmul(sigma,C.T)),Q)))) #3x753
	print('K',K)
	add_to_mu = np.array(np.subtract(obs_mes, pred_mes)) #251x3
	mu = np.add(mu_mat, np.matmul(K,add_to_mu))[:,0]
	sigma = np.matmul(np.subtract(np.identity(3),np.matmul(K,C)),sigma)
#	diff = (np.sum(np.subtract(obs_mes,pred_mes)))/251
#	mu = np.add(mu, diff)
	print('mu_end',mu)
	mu_pose.position.x = mu[0]
	mu_pose.position.y = mu[1]
	mu_pose.orientation.w = 1.0
	mu_pose.orientation.z = 0.0
	mu_pose.orientation = rotateQuaternion(mu_pose.orientation, mu[2])
	return mu_pose,sigma




    def kalman_update(self,mu_pose,sigma,scan):
	print('pf base kalman update running')
	Q = np.zeros((20,20))
	np.fill_diagonal(Q,10000)

	x = mu_pose.position.x
	y = mu_pose.position.y
	z = getHeading(mu_pose.orientation)

	mu_mat = np.array([[x,x+1,x-1],[y,y-1,y+2],[z,z-1,z+1]])
	pred_mat = np.zeros((20,3))
	act_mat = np.zeros((20,3))
	for i in range(0,3):
            for k, obs_bearing in self.sensor_model.reading_points:
                act_mat[k/26,i] = scan.ranges[k]
                if (act_mat[k/26,i] <= 0.0):
                    act_mat[k/26,i] = sensor_model.scan_range_max
                pred_mat[k/26,i] = self.sensor_model.calc_map_range(mu_mat[0,i], mu_mat[1,i], mu_mat[2,i] + obs_bearing)

#	C = np.matmul(pred_mat,np.linalg.inv(mu_mat))   #20x3
	C = np.eye(20,3)

	print('mu_mat:',mu_mat) #3x3
	print('pred_mat[:,0]',pred_mat[:,0])
	print('act_mat',act_mat) #20x1
	print('act-pred',np.subtract(act_mat[:,0],pred_mat[:,0]))
	print('C',C)



	K = np.matmul(sigma,np.matmul(C.T,np.linalg.inv(np.matmul(np.matmul(C,sigma),C.T)+Q)))
	print('K',K)
	mu = np.add(mu_mat[:,0], np.matmul(K,np.subtract(act_mat[:,0],pred_mat[:,0])))
	print('adding:',np.matmul(K,np.subtract(act_mat[:,0],pred_mat[:,0])))
	print('mu',mu)
	mu_pose.position.x = mu[0]
	mu_pose.position.y = mu[1]
	mu_pose.orientation.w = 1.0
	mu_pose.orientation.z = 0.0
	mu_pose.orientation = rotateQuaternion(mu_pose.orientation, mu[2])
	sigma = np.matmul(np.subtract(np.identity(3),np.matmul(K,C)),sigma)
	print('mu_pose',mu_pose)
	print('sigma',sigma)
	return mu_pose, sigma

    def alt_kal(self,mu_pose,sigma,scan):
	mu = np.array([mu_pose.position.x, mu_pose.position.y, getHeading(mu_pose.orientation)])
	print('mu start',mu)
	C_sigma = np.zeros((20,3)) #20x3
	z = np.zeros((20,1)) #20x1
	C_mu = np.zeros((20,1)) #20x1
	Q = np.zeros((20,20)) #20x20
	np.fill_diagonal(Q,1)
	CCsigmaT = np.zeros((20,20)) #20x20

	for i in range(0,3): #3
            for k, obs_bearing in self.sensor_model.reading_points: #20
		C_sigma[k/26][i] = self.sensor_model.calc_map_range(sigma[0][i],sigma[1][i],(sigma[2][i]+obs_bearing)%(TWO_PI))
#		C_sigma[k/26][i] = self.sensor_model.calc_map_range(24,11,(10.05+obs_bearing)%(TWO_PI))
#		C_sigma[k/26][i] = self.sensor_model.calc_map_range(mu[0],mu[1],mu[2] + obs_bearing)
            	z[k/26] = scan.ranges[k]
            	if (z[k/26] <= 0.0):
            	    z[k/26] = sensor_model.scan_range_max
            	C_mu[k/26] = self.sensor_model.calc_map_range(mu[0], mu[1], mu[2] + obs_bearing)
	print('C_mu',C_mu)
	print('z',z)
	for k ,obs_bearing in self.sensor_model.reading_points: #20
	    for i in range(0,20): #20
		CCsigmaT[k/26][i] = self.sensor_model.calc_map_range(C_sigma.T[0][i],C_sigma.T[1][i],C_sigma.T[2][i])
	print('sigma',sigma)
	print('C_sigma',C_sigma)
	K = np.matmul(C_sigma.T,np.linalg.inv(CCsigmaT+Q)) #3x20
	print('K',K)
	print('adding:',np.matmul(K,np.subtract(z,C_mu)))
	print('mu mid',mu)
	mu = np.add(mu, np.matmul(K,np.subtract(z,C_mu)).T)[0]
	sigma = np.subtract(sigma,np.matmul(K,C_sigma))
	print('mu',mu)
	mu_pose.position.x = mu[0]
	mu_pose.position.y = mu[1]
	mu_pose.orientation.w = 1.0
	mu_pose.orientation.z = 0.0
	mu_pose.orientation = rotateQuaternion(mu_pose.orientation, mu[2])
	return mu_pose, sigma


    def new_kal(self,mu_pose,sigma,scan):
	Q = np.zeros((3,3)) #3x3
	np.fill_diagonal(Q,1)

	mu = [mu_pose.position.x,mu_pose.position.y,getHeading(mu_pose.orientation)]
	pred_scan = [0]*500
	idp1 = 0
	id1 = 0
	for j in range(0,500):
		obs_bearing = -PI_OVER_TWO + 0.00629577692598*j
		pred_scan[j] = self.sensor_model.calc_map_range(mu[0], mu[1], mu[2] + obs_bearing)
		if pred_scan[j] > pred_scan[idp1] and pred_scan[j] < 5.58:
			idp1 = j
		if scan.ranges[j] > scan.ranges[id1] and scan.ranges[j] < 5.58:
			id1 = j

#	idp1 = pred_scan.index(max([x for x in pred_scan with x < 5.56])) #index of max predicted range
	idp2 = np.argmin(pred_scan) #index of min predicted range
	dp1 = pred_scan[idp1] #value of furthest predicted range
	dp2 = pred_scan[idp2] #value of closest predicted range
	theta1 = -PI_OVER_TWO + idp1*0.00629577692598 #angle of predicted furthest point
	theta2 = -PI_OVER_TWO + idp2*0.00629577692598 #angle of predicted closest point
	b1 = theta1 - PI_OVER_TWO + mu[2]
	b2 = theta2 - PI_OVER_TWO + mu[2]
	p1 = [mu[0] + dp1*math.cos(b1), mu[1] + dp1*math.sin(b1)]
	p2 = [mu[0] + dp2*math.cos(b2), mu[1] + dp2*math.sin(b2)]
	pFurtherToLeft = (theta1 > theta2) #bool: is predicted furthest further to left than predicted closest

#	id1 = scan.ranges.index(max([x for x in scan.ranges with x < 5.56])) #index of maximum range
	id2 = np.argmin(scan.ranges) #index of minimum range
	da1 = scan.ranges[id1] #value of maximum range (distance to furthest point)
	da2 = scan.ranges[id2] #value of minimum range (distance to closest point)
	atheta1 = id1*0.00629577692598 #angle of furthest measurement
	atheta2 = id2*0.00629577692598 #angle of closest measurement
	FurthertoLeft = (atheta1 > atheta2) #bool: is furthest further to left than closest
	
	d = math.sqrt((p1[0]-p2[0])**2 + (p1[0]-p2[0])**2)
	l = (da1**2 - da2**2 + d**2)/(2*d)
	h = math.sqrt(abs(da1**2 - l**2))
	
	obs_x1 = (l/d)*(p2[0]-p1[0]) + (h/d)*(p2[1]-p1[1]) + p1[0]
	obs_y1 = (l/d)*(p2[1]-p1[1]) - (h/d)*(p2[0]-p1[0]) + p1[1]
	obs_x2 = (l/d)*(p2[0]-p1[0]) - (h/d)*(p2[1]-p1[1]) + p1[0]
	obs_y2 = (l/d)*(p2[1]-p1[1]) + (h/d)*(p2[0]-p1[0]) + p1[1]

	if (obs_x1 - mu[0])**2 + (obs_y1 - mu[1])**2 < (obs_x2 - mu[0])**2 + (obs_y2 - mu[1])**2:
		obs_mu = [obs_x1, obs_y1, 0]
	else:
		obs_mu = [obs_x2, obs_y1, 0]

	if obs_mu[0] > p1[0]:
		if obs_mu[1] <= p1[1]:
			phi = math.pi - math.asin((p1[1]-obs_mu[1])/da1)
		else:
			phi = math.asin((obs_mu[1]-p1[1])/da1) - math.pi
	else:
		phi = math.asin((p1[1] - obs_mu[1])/da1)
	
	obs_mu[2] = PI_OVER_TWO - (atheta1 - phi)

	K = np.matmul(sigma,np.linalg.inv(np.add(sigma,Q)))
	mu = np.add(mu, np.matmul(K,(np.subtract(obs_mu,mu))))
	sigma = np.matmul(np.subtract(np.identity(3),K),sigma)

	mu_pose.position.x = mu[0]
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
	print('pf base predict from odom running')
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
	mu.orientation = rotateQuaternion(mu.orientation, dif_heading)# + rnd * dif_heading * self.ODOM_ROTATION_NOISE)
	theta = getHeading(mu.orientation)
	travel_x = distance_travelled * math.cos(theta)
        travel_y = distance_travelled * math.sin(theta)
        mu.position.x = (mu.position.x + travel_x)# + (rnd * travel_x * self.ODOM_TRANSLATION_NOISE))
	mu.position.y = (mu.position.y + travel_y)# + (rnd * travel_y * self.ODOM_DRIFT_NOISE))

        return mu
    
	

    def set_initial_pose(self, pose):
        """ Initialise filter with start pose """
        self.estimatedpose.pose = pose.pose
        # ----- Estimated pose has been set, so we should now reinitialise the 
        # ----- particle cloud around it
        rospy.loginfo("Got pose. Calling initialise_particle_cloud().")
#        self.particlecloud = self.initialise_particle_cloud(self.estimatedpose)
	self.sigma = self.initialise_kalman()
#        self.particlecloud.header.frame_id = "/map"
    
    def set_map(self, occupancy_map):
        """ Set the map for localisation """
        self.occupancy_map = occupancy_map
        self.sensor_model.set_map(occupancy_map)
        # ----- Map has changed, so we should reinitialise the particle cloud
        rospy.loginfo("Particle filter got map. (Re)initialising.")
#        self.particlecloud = self.initialise_particle_cloud(self.estimatedpose)
	self.sigma = self.initialise_kalman()
#        self.particlecloud.header.frame_id = "/map"
