#!/usr/bin/env python3

"""
Start ROS node to publish linear and angular velocities to mymobibot in order to perform wall following.
"""

# Ros handlers services and messages
import rospy, roslib
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Range
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
#Math imports
from math import sin, cos, atan2, pi, sqrt
from numpy.linalg import inv, det, norm, pinv
import numpy as np
import time as t
import matplotlib.pyplot as plt

# from tf.transformations import euler_from_quaternion
# from tf.transformations import quaternion_matrix
# matrix = quaternion_matrix([1, 0, 0, 0])

def quaternion_to_euler(w, x, y, z):
    """Converts quaternions with components w, x, y, z into a tuple (roll, pitch, yaw)"""
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * np.pi / 2, np.arcsin(sinp))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

class mymobibot_follower():
    """Class to compute and publish joints positions"""
    def __init__(self,rate):

        # linear and angular velocity
        self.velocity = Twist()
        # joints' states
        self.joint_states = JointState()
        # Sensors
        self.imu = Imu()
        self.imu_yaw = 0.0 # (-pi, pi]
        self.sonar_F = Range()
        self.sonar_FL = Range()
        self.sonar_FR = Range()
        self.sonar_L = Range()
        self.sonar_R = Range()

        # ROS SETUP
        # initialize subscribers for reading encoders and publishers for performing position control in the joint-space
        # Robot
        self.velocity_pub = rospy.Publisher('/mymobibot/cmd_vel', Twist, queue_size=1)
        self.joint_states_sub = rospy.Subscriber('/mymobibot/joint_states', JointState, self.joint_states_callback, queue_size=1)
        # Sensors
        self.imu_sub = rospy.Subscriber('/imu', Imu, self.imu_callback, queue_size=1)
        self.sonar_front_sub = rospy.Subscriber('/sensor/sonar_F', Range, self.sonar_front_callback, queue_size=1)
        self.sonar_frontleft_sub = rospy.Subscriber('/sensor/sonar_FL', Range, self.sonar_frontleft_callback, queue_size=1)
        self.sonar_frontright_sub = rospy.Subscriber('/sensor/sonar_FR', Range, self.sonar_frontright_callback, queue_size=1)
        self.sonar_left_sub = rospy.Subscriber('/sensor/sonar_L', Range, self.sonar_left_callback, queue_size=1)
        self.sonar_right_sub = rospy.Subscriber('/sensor/sonar_R', Range, self.sonar_right_callback, queue_size=1)

        #Publishing rate
        self.period = 1.0/rate
        self.pub_rate = rospy.Rate(rate)

        self.publish()

    #SENSING CALLBACKS
    def joint_states_callback(self, msg):
        # ROS callback to get the joint_states

        self.joint_states = msg
        # (e.g. the angular position of the left wheel is stored in :: self.joint_states.position[0])
        # (e.g. the angular velocity of the right wheel is stored in :: self.joint_states.velocity[1])

    def imu_callback(self, msg):
        # ROS callback to get the /imu

        self.imu = msg
        # (e.g. the orientation of the robot wrt the global frome is stored in :: self.imu.orientation)
        # (e.g. the angular velocity of the robot wrt its frome is stored in :: self.imu.angular_velocity)
        # (e.g. the linear acceleration of the robot wrt its frome is stored in :: self.imu.linear_acceleration)

        #quaternion = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
        #(roll, pitch, self.imu_yaw) = euler_from_quaternion(quaternion)
        (roll, pitch, self.imu_yaw) = quaternion_to_euler(msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z)

    def sonar_front_callback(self, msg):
        # ROS callback to get the /sensor/sonar_F

        self.sonar_F = msg
        # (e.g. the distance from sonar_front to an obstacle is stored in :: self.sonar_F.range)

    def sonar_frontleft_callback(self, msg):
        # ROS callback to get the /sensor/sonar_FL

        self.sonar_FL = msg
        # (e.g. the distance from sonar_frontleft to an obstacle is stored in :: self.sonar_FL.range)

    def sonar_frontright_callback(self, msg):
        # ROS callback to get the /sensor/sonar_FR

        self.sonar_FR = msg
        # (e.g. the distance from sonar_frontright to an obstacle is stored in :: self.sonar_FR.range)

    def sonar_left_callback(self, msg):
        # ROS callback to get the /sensor/sonar_L

        self.sonar_L = msg
        # (e.g. the distance from sonar_left to an obstacle is stored in :: self.sonar_L.range)

    def sonar_right_callback(self, msg):
        # ROS callback to get the /sensor/sonar_R

        self.sonar_R = msg
        # (e.g. the distance from sonar_right to an obstacle is stored in :: self.sonar_R.range)

    def publish(self):

        # set configuration
        self.velocity.linear.x = 0.0
        self.velocity.angular.z = 0.0
        tmp_rate = rospy.Rate(1)
        tmp_rate.sleep()
        print("The system is ready to execute your algorithm...")
        sonar_Left_prev = 0
        sonar_frontLeft_prev = 0
        rostime_now = rospy.get_rostime()
        time_now = rostime_now.to_nsec()
        trying_to_find_wall = True
        start = True 
        plot = True 
        zero_time = 0
        time_for_plot=[]
        possition_error = []
        linear_vel = []
        angular_vel = []
        minimum_distance_from_walls = []

        while not rospy.is_shutdown():
            
            sonar_front = self.sonar_F.range
            sonar_frontRight = self.sonar_FR.range
            sonar_frontLeft = self.sonar_FL.range
            sonar_Left = self.sonar_L.range

            yd = 0.25
            Kp = 15
            Kd = 8
            dt = 0.01

            # ROBOT IN STATE 0 (TRYING TO FIND SIDE OF WALL)
            
            if min(sonar_front,sonar_frontLeft,sonar_frontRight) >= yd  and trying_to_find_wall:
                self.velocity.linear.x = 0.2 
                self.velocity.angular.z = 0.1 
            
            # ROBOT FOUND WALL 

            else :
                # TIME REFERNCE FOR THE PLOTS 
                if start :
                    rostime_now = rospy.get_rostime()
                    time_ref = rostime_now.to_nsec()
                    zero_time = time_ref/1e9
                    start = False

                #SET FLAG TO "FALSE"
                trying_to_find_wall  = False

                # ROBOT IN STATE 2 (ROBOT IS NEARBY ONE OF FOUR CORNERS AND THEREFORE MUST TURN)
                # USE OF CONSTANT -0.2 TO ACHIEVE "EXTRA" SMOOTHNESS IN z-ANGULAR VELOCITY
                if sonar_frontLeft>= sonar_front-0.2 or sonar_Left>=sonar_front-0.2:
                    self.velocity.angular.z = 0.9 
                    self.velocity.linear.x = 0.2
                    print("Turning...")
                
                # ROBOT IN STATE 1 (ROBOT IS ALLIGNED TO A SIDE OF WALL AND MUST FOLLOW IT)
                # USE OF PD CONTROLLER FOR ANGULAR z-VELOCITY HERE
                else :
                    vel_error = max((sonar_Left_prev - sonar_Left), (sonar_frontLeft_prev-sonar_frontLeft))/dt
                    pos_error = max(yd - sonar_Left, yd-sonar_frontLeft*cos(pi/4)) 
                    self.velocity.angular.z =  Kp*pos_error +Kd*vel_error
                    self.velocity.linear.x = 0.2
                    print("Following...")
            
            # AGAIN TIME FOR THE PLOTS
            rostime_now = rospy.get_rostime()
            time_now2 = rostime_now.to_nsec()
            time = time_now2/1e9-zero_time
            
            # Plots Part

            # if time <= 60 and  trying_to_find_wall==False :
            #     time_for_plot.append(time)
            #     minimum_distance_from_walls.append(min(sonar_front,sonar_frontLeft,sonar_frontRight,sonar_Left))
            #     linear_vel.append(self.velocity.linear.x)
            #     angular_vel.append(self.velocity.angular.z)
            #     possition_error.append(abs(pos_error))
                

            
            # if time > 60 and plot :
            #     print(minimum_distance_from_walls)
            #     f_errorx,ax = plt.subplots()
            #     name_for_erros_plots = "pos_error"
            #     ax.grid(True)
            #     ax.plot(time_for_plot ,possition_error)
            #     ax.set_title("Position Error", fontsize= 18)
            #     ax.set_xlabel("Time (sec)")
            #     ax.set_ylabel("Error (m)")
            #     # ax.set_ylim(0.18,0.3)
            #     f_errorx.savefig('/home/savas/Desktop/robo_plots/{}(Kp={},Kd={})_new.png'.format(name_for_erros_plots,Kp,Kd), bbox_inches ="tight") 

            #     f_min_dist,ax = plt.subplots()
            #     name_for_erros_plots = "min_dist"
            #     ax.grid(True)
            #     ax.plot(time_for_plot ,minimum_distance_from_walls)
            #     ax.set_title("Minimum Distance from walls", fontsize= 18)
            #     # ax.set_ylim(0.18,0.3)
            #     ax.set_xlabel("Time (sec)")
            #     ax.set_ylabel("Distance (m)")
            #     f_min_dist.savefig('/home/savas/Desktop/robo_plots/{}(Kp={},Kd={})_new.png'.format(name_for_erros_plots,Kp,Kd), bbox_inches ="tight")

            #     f_vel , ax = plt.subplots()
            #     name_for_ef_pos = "x-Velocities"
            #     ax.grid(True)
            #     ax.set_xlabel('Time (sec)')
            #     ax.set_ylabel('Velocity (m/s)')
            #     # ax.set_ylim(-0.25, 1.0)
            #     ax.set_title("x-Velocity over time", fontsize = 18)
            #     ax.plot (time_for_plot,linear_vel, label = 'Linear x-velocity', color = 'red')
            #     f_vel.savefig('/home/savas/Desktop/robo_plots/{}(Kp={},Kd={}).png'.format(name_for_ef_pos,Kp,Kd), bbox_inches ="tight")

            #     f_velz , ax = plt.subplots()
            #     name_for_ef_pos = "z-Velocities"
            #     ax.grid(True)
            #     ax.set_xlabel('Time (sec)')
            #     ax.set_ylabel('Velocity (m/s)')
            #     # ax.set_ylim(-0.25, 1.0)
            #     ax.set_title("z-Velocity over time", fontsize = 18)
            #     ax.plot (time_for_plot,angular_vel, label = 'Linear z-velocity', color = 'blue')
            #     f_velz.savefig('/home/savas/Desktop/robo_plots/{}(Kp={},Kd={}).png'.format(name_for_ef_pos,Kp,Kd), bbox_inches ="tight")

            #     plot = False

            # Calculate time interval (in case it is needed)
            time_prev = time_now
            rostime_now = rospy.get_rostime()
            time_now = rostime_now.to_nsec()
            dt = (time_now - time_prev)/1e9
            sonar_Left_prev = sonar_Left
            sonar_frontLeft_prev = sonar_frontLeft


            # Publish the new joint's angular positions
            self.velocity_pub.publish(self.velocity)

            self.pub_rate.sleep()
        
        

    def turn_off(self):
        pass

def follower_py():
    # Starts a new node
    rospy.init_node('follower_node', anonymous=True)
    # Reading parameters set in launch file
    rate = rospy.get_param("/rate")

    follower = mymobibot_follower(rate)
    rospy.on_shutdown(follower.turn_off)
    rospy.spin()

if __name__ == '__main__':
    try:
        follower_py()
    except rospy.ROSInterruptException:
        pass
