#!/usr/bin/env python3

"""
Start ROS node to publish angles for the position control of the xArm7.
"""

# Ros handlers services and messages
import rospy, roslib
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ModelStates
#Math imports
from math import sin, cos, atan2, pi, sqrt
from numpy.linalg import inv, det, norm, pinv
import numpy as np
import time as t
from sympy import * 
import matplotlib.pyplot as plt 

# Arm parameters
# xArm7 kinematics class
from kinematics import xArm7_kinematics

# from tf.transformations import quaternion_matrix
# matrix = quaternion_matrix([1, 0, 0, 0])

class xArm7_controller():
    """Class to compute and publish joints positions"""
    def __init__(self,rate):

        #plot
        self.pub1 = rospy.Publisher("/py_endeffector", Float64, queue_size=100)
        self.pub2 = rospy.Publisher("/pxdot", Float64, queue_size=100)
        self.pub3 = rospy.Publisher("/pzdot", Float64, queue_size=100)


        # Init xArm7 kinematics handler
        self.kinematics = xArm7_kinematics()

        # joints' angular positions
        self.joint_angpos = np.array([0, 0, 0, 0, 0, 0, 0])
        # joints' angular velocities
        self.joint_angvel = np.array([0, 0, 0, 0, 0, 0, 0])
        # joints' states
        self.joint_states = JointState()
        # joints' transformation matrix wrt the robot's base frame
        self.A01 = self.kinematics.tf_A01(self.joint_angpos)
        self.A02 = self.kinematics.tf_A02(self.joint_angpos)
        self.A03 = self.kinematics.tf_A03(self.joint_angpos)
        self.A04 = self.kinematics.tf_A04(self.joint_angpos)
        self.A05 = self.kinematics.tf_A05(self.joint_angpos)
        self.A06 = self.kinematics.tf_A06(self.joint_angpos)
        self.A07 = self.kinematics.tf_A07(self.joint_angpos)

        # gazebo model's states
        self.model_states = ModelStates()
        #self.pub3 = rospy.Publisher("/pydot", Float64, queue_size=100)
        # ROS SETUP
        # initialize subscribers for reading encoders and publishers for performing position control in the joint-space
        # Robot
        self.joint_states_sub = rospy.Subscriber('/xarm/joint_states', JointState, self.joint_states_callback, queue_size=1)
        self.joint1_pos_pub = rospy.Publisher('/xarm/joint1_position_controller/command', Float64, queue_size=1)
        self.joint2_pos_pub = rospy.Publisher('/xarm/joint2_position_controller/command', Float64, queue_size=1)
        self.joint3_pos_pub = rospy.Publisher('/xarm/joint3_position_controller/command', Float64, queue_size=1)
        self.joint4_pos_pub = rospy.Publisher('/xarm/joint4_position_controller/command', Float64, queue_size=1)
        self.joint5_pos_pub = rospy.Publisher('/xarm/joint5_position_controller/command', Float64, queue_size=1)
        self.joint6_pos_pub = rospy.Publisher('/xarm/joint6_position_controller/command', Float64, queue_size=1)
        self.joint7_pos_pub = rospy.Publisher('/xarm/joint7_position_controller/command', Float64, queue_size=1)
        # Obstacles
        self.model_states_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback, queue_size=1)

        #Publishing rate
        self.period = 1.0/rate
        self.pub_rate = rospy.Rate(rate)

        self.publish()

    #SENSING CALLBACKS
    def joint_states_callback(self, msg):
        # ROS callback to get the joint_states

        self.joint_states = msg
        # (e.g. the angular position of joint 1 is stored in :: self.joint_states.position[0])

    def model_states_callback(self, msg):
        # ROS callback to get the gazebo's model_states

        self.model_states = msg
        # (e.g. #1 the position in y-axis of GREEN obstacle's center is stored in :: self.model_states.pose[1].position.y)
        # (e.g. #2 the position in y-axis of RED obstacle's center is stored in :: self.model_states.pose[2].position.y)
    
    def coeffs(self,S,E,t):

        # x(t)=a0+a1t+a2t^2+a3t^3+a4t^4+a5t^5

        a0=symbols('a0')
        a1=symbols('a1')
        a2=symbols('a2')
        a3=symbols('a3')
        a4=symbols('a4')
        a5=symbols('a5')

        # 5-th order
        eq0=Eq(a0-S,0)
        eq1=Eq(a1,0)
        eq2=Eq(a2,0)
        eq3=Eq(a0+a1*t+a2*(t**2)+a3*(t**3)+a4*(t**4)+a5*(t**5)-E,0)
        eq4=Eq(a1+2*a2*t+3*a3*(t**2)+4*a4*(t**3)+5*a5*(t**4),0)
        eq5=Eq(2*a2*t+6*a3*(t)+12*a4*(t**2)+20*a5*(t**3),0)

        sol=solve((eq0,eq1,eq2,eq3,eq4,eq5),(a0,a1,a2,a3,a4,a5))
        res = []
        for item in sol.values():
            res.insert(0,float(item)) 
    
        return res


    def dangerous_point(self, r_joints_array):

        #Check which frame is more possible to collide with obstacles (i.e has minimum distance)
        pointA = self.kinematics.tf_A04A(self.joint_angpos)
        pointF = self.kinematics.tf_A04F(self.joint_angpos)
        pointB = self.kinematics.tf_A04B(self.joint_angpos)
        pointC = self.kinematics.tf_A04C(self.joint_angpos)
        pointD = self.kinematics.tf_A04D(self.joint_angpos)
        pointE = self.kinematics.tf_A04E(self.joint_angpos)

        P = np.array([[pointA[0,3], pointA[1,3]],[pointF[0,3], pointF[1,3]],[pointB[0,3], pointB[1,3]],[pointC[0,3], pointC[1,3]],[pointD[0,3], pointD[1,3]],[pointE[0,3], pointE[1,3]]])

        min_d = 1
        obj = 0  
        index = 0  
        for i in range(len(P)):

            # Find Minimum Distance between xarm7 and green obstacle's perimeter
            temp=np.sqrt((P[i,0]-self.model_states.pose[1].position.x)**2 + (P[i,1]-self.model_states.pose[1].position.y)**2)-0.05
            if (temp < min_d):
                min_d = temp
                index = i 
                obj = 1

            # Find Minimum Distance between xarm7 and red obstacle's perimeter
            temp=np.sqrt((P[i,0]-self.model_states.pose[2].position.x)**2 + (P[i,1]-self.model_states.pose[2].position.y)**2)-0.05
            if (temp < min_d):
                min_d = temp
                index = i 
                obj = 2 

        return min_d, index, obj

    def select_func(self,r_joints_array, d0) :

        # Given the frame closest to green or red obstacle , calculate the qrdot by minimizing the criterion ||frame_pos - obstacle_pos||
        d , index , objct = self.dangerous_point(r_joints_array)
        if d>d0 :
            return np.zeros((7,))
        else :
            if index == 0 :
                return self.kinematics.pointA_func(self.joint_angpos,self.model_states.pose[objct].position.y)
            elif index == 1 :
                return self.kinematics.pointB_func(self.joint_angpos,self.model_states.pose[objct].position.y)
            elif index == 2:
                return self.kinematics.pointC_func(self.joint_angpos,self.model_states.pose[objct].position.y)
            elif index == 3 :
                return self.kinematics.pointD_func(self.joint_angpos,self.model_states.pose[objct].position.y)
            elif index == 4 :
                return self.kinematics.pointE_func(self.joint_angpos,self.model_states.pose[objct].position.y)
            elif index == 5 :
                return self.kinematics.pointF_func(self.joint_angpos,self.model_states.pose[objct].position.y)

    def publish(self):

        # set configuration
        self.joint_angpos = np.array([0, 0.75, 0, 1.5, 0, 0.75, 0],dtype='float')
        tmp_rate = rospy.Rate(1)
        tmp_rate.sleep()
        self.joint4_pos_pub.publish(self.joint_angpos[3])
        tmp_rate.sleep()
        self.joint2_pos_pub.publish(self.joint_angpos[1])
        self.joint6_pos_pub.publish(self.joint_angpos[5])
        tmp_rate.sleep()

        #Oscilation Period
        T = 5
        
        # Polynom for first quarter of movement
        cs = (self.coeffs(0.000,0.200,T/4))

        # Polynom for movement from pA to pB 
        csAB = self.coeffs(0.200,-0.200,T/2) 

        # Polynom for movement from pB to pA 
        csBA = self.coeffs(-0.200,0.200,T/2)

        polynom = np.poly1d(cs)
        der = np.polyder(polynom)
        print("The system is ready to execute your algorithm...")

        rostime_now = rospy.get_rostime()
        time_now = rostime_now.to_nsec()
        
        K1 = 0.5
        kc = 4
        d_safe = 0.056
        cnt=0
        start = True
        first_time = True
        AB =True
        again = True
        ef_pos_x = []
        ef_pos_y = []
        ef_pos_z = []
        time_for_plot=[]
        min_dis = []
        y_error =[]
        ef_error_x = []
        ef_error_z = []
        vel_x = [] 
        vel_y = []
        vel_z = []

        while not rospy.is_shutdown():
            # Compute each transformation matrix wrt the base frame from joints' angular positions
            self.A07 = self.kinematics.tf_A07(self.joint_angpos)
            
            if start  :
                rostime_now = rospy.get_rostime()
                time_ref = rostime_now.to_nsec()
                zero_time = time_ref/1e9
                start = False
                time = np.abs(time_now-time_ref)/1e9
            
            #Reset time when end effector reaches maximum positive displacement 
            if (self.A07[1,3]>0.20001 and AB == True) :
                rostime_now = rospy.get_rostime()
                time_ref = rostime_now.to_nsec()
                polynom = np.poly1d(csAB) 
                AB = False
                time = np.abs(time_now-time_ref)/1e9
                y_error.append(np.abs(0.2-self.kinematics.tf_A07(self.joint_states.position)[1,3])) # Calculate the error when at maximum positive displacement
                if not first_time :
                    cnt+=1
                first_time = False

            #Reset time when end effector reaches maximum negative displacement
            if (self.A07[1,3]<-0.20001 and AB == False ) :
                rostime_now = rospy.get_rostime()
                time_ref = rostime_now.to_nsec()
                polynom = np.poly1d(csBA)
                AB = True
                time = np.abs(time_now-time_ref)/1e9
                y_error.append(np.abs(-0.2-self.kinematics.tf_A07(self.joint_states.position)[1,3])) # Calculate the error when at maximum negative displacement
                cnt+=1

            time = np.abs(time_now-time_ref)/1e9
            
            # Compute jacobian matrix
            J = self.kinematics.compute_jacobian(self.joint_angpos)
            # Pseudoinverse jacobian
            pinvJ = pinv(J)

            #1st subtask
            pxdot = 0
            pzdot = 0
            if ((np.abs(self.A07[0,3]-0.6043)>0.00001)):
                pxdot = -5*K1*(self.A07[0,3]-0.6043)
            if(np.abs(self.A07[2,3]-0.1508)>0.00001):
                pzdot = -2*K1*(self.A07[2,3]-0.1508)

            # Calculate the derivative of trajectory polynom 
            der = np.polyder(polynom)

            # Calculation of x,y,z Velocities of end effector 
            pydot = np.polyval(der,time)
            pdots = np.array([pxdot , pydot , pzdot])
            q1dot = np.asarray(np.dot(np.asarray(pinvJ),pdots))

            #2nd subtask
            pinvJJ = np.asarray(np.dot(pinvJ,J))
            med = np.identity(7)-pinvJJ
            qrdot = self.select_func(self.joint_angpos,d_safe)
            q2dot = np.asarray(np.dot(med,qrdot))

            #Final qdot 
            self.joint_angvel= q1dot + kc*q2dot
            
            #Gathering information for plots
            if cnt <= 6 :
                rostime_now = rospy.get_rostime()
                time_now2 = rostime_now.to_nsec()
                time_for_plot.append(time_now2/1e9-zero_time)
                ef_pos_x.append(self.kinematics.tf_A07(self.joint_states.position)[0,3])
                ef_pos_y.append(self.kinematics.tf_A07(self.joint_states.position)[1,3])
                ef_pos_z.append(self.kinematics.tf_A07(self.joint_states.position)[2,3])
                ef_error_x.append(np.abs(self.kinematics.tf_A07(self.joint_states.position)[0,3]-0.6043))
                ef_error_z.append(np.abs(self.kinematics.tf_A07(self.joint_states.position)[2,3]-0.1508))
                tmp,_,_ = self.dangerous_point(self.joint_states.position)
                min_dis.append(tmp)
                vel_x.append(pxdot)
                vel_y.append(pydot)
                vel_z.append(pzdot)
                

            # Plots for the first 3 Periods
            if cnt > 6  and again == True :

                # x,y,z position of end effector
                f_pos , ax = plt.subplots()
                name_for_ef_pos = "ef_pos"
                ax.grid(True)
                ax.set_xlabel('Time (sec)')
                ax.set_ylabel('Position')
                ax.set_ylim(-0.25, 1.0)
                ax.set_title("Position of End Effector over time", fontsize = 18)
                line1 = ax.plot (time_for_plot,ef_pos_y, label = 'Movement on y-axis', color = 'red')
                line2 = ax.plot (time_for_plot, ef_pos_x, label = 'Movement on x - axis', color = 'blue')
                line3 = ax.plot (time_for_plot,ef_pos_z, label = 'Movement on z- axis', color = 'green')
                lns = line1 + line2 +line3
                labs = [l.get_label() for l in lns]
                ax.legend(lns, labs, loc = 'upper right')
                # f_pos.savefig('/home/savas/Desktop/{}.png'.format(name_for_ef_pos), bbox_inches ="tight")

                # Plot x deviation from 0.6043
                f_errorx,ax = plt.subplots()
                name_for_erros_plots = "erros_x"
                ax.grid(True)
                ax.plot(time_for_plot ,ef_error_x)
                ax.set_title("Error in X axis", fontsize= 18)
                ax.set_xlabel("Time (sec)")
                ax.set_ylabel("Error (m)")
                # f_errorx.savefig('/home/savas/Desktop/{}.png'.format(name_for_erros_plots), bbox_inches ="tight")

                # Plot z deviation from 0.1508
                f_errorz,ax = plt.subplots()
                name_for_errors_plots = "erros_z"
                ax.grid(True)
                ax.plot(time_for_plot ,ef_error_z)
                ax.set_title("Error in Z axis", fontsize= 18)
                ax.set_xlabel("Time (sec)")
                ax.set_ylabel("Error (m)")
                # f_errorz.savefig('/home/savas/Desktop/{}.png'.format(name_for_errors_plots), bbox_inches ="tight")

                # y deviation at maximum displacement 
                f_errory,ax = plt.subplots()
                name_for_errors_plots = "erros_y"
                ax.grid(True)
                x = np.arange(len(y_error))
                ax.scatter(x,y_error)
                ax.set_ylim(min(y_error)-0.0007, max(y_error)+0.0007)
                ax.set_title("Error in Y axis at maximum displacement", fontsize= 18)
                ax.set_xlabel("Times")
                ax.set_ylabel("Error (m)")
                # f_errory.savefig('/home/savas/Desktop/{}.png'.format(name_for_errors_plots), bbox_inches ="tight")

                # Plot of minimum distance between xarm7 and obstacles over time 
                f_dis , ax = plt.subplots()
                ax.grid(True)
                name = "min_dist"
                ax.plot(time_for_plot,min_dis)
                ax.set_title("Minimum Distance from obstacles", fontsize= 18)
                ax.set_xlabel("Time (sec)")
                ax.set_ylabel("Distance (m)")
                # f_dis.savefig('/home/savas/Desktop/{}.png'.format(name), bbox_inches ="tight")

                # x component of end effector's velocity 
                f_vel_x, ax = plt.subplots()
                ax.grid(True)
                ax.set_ylabel("Velocity (m/s)")
                ax.set_xlabel("Time (sec)")
                ax.set_title("Velocity of End Effector on X axis")
                ax.plot(time_for_plot,vel_x)
                # f_vel_x.savefig('/home/savas/Desktop/velocity_x.png')

                # y component of end effector's velocity
                f_vel_y, ax = plt.subplots()
                ax.grid(True)
                ax.set_ylabel("Velocity (m/s)")
                ax.set_xlabel("Time (sec)")
                ax.set_title("Velocity of End Effector on Y axis")
                ax.plot(time_for_plot,vel_y)
                # f_vel_y.savefig('/home/savas/Desktop/velocity_y.png')

                # z component of end effector's velocity 
                f_vel_z, ax = plt.subplots()
                ax.grid(True)
                ax.set_ylabel("Velocity (m/s)")
                ax.set_xlabel("Time (sec)")
                ax.set_title("Velocity of End Effector on Z axis")
                ax.plot(time_for_plot,vel_z)
                # f_vel_z.savefig('/home/savas/Desktop/velocity_z.png')
            

                again = False 
            # Convertion to angular position after integrating the angular speed in time
            # Calculate time interval
            time_prev = time_now
            rostime_now = rospy.get_rostime()
            time_now = rostime_now.to_nsec()
            dt = (time_now - time_prev)/1e9
            # print(self.joint_states.position)
            # Integration
            self.joint_angpos = np.add(self.joint_angpos , np.array([index * dt for index in self.joint_angvel]))

            # Publish the new joint's angular positions
            self.joint1_pos_pub.publish(self.joint_angpos[0])
            self.joint2_pos_pub.publish(self.joint_angpos[1])
            self.joint3_pos_pub.publish(self.joint_angpos[2])
            self.joint4_pos_pub.publish(self.joint_angpos[3])
            self.joint5_pos_pub.publish(self.joint_angpos[4])
            self.joint6_pos_pub.publish(self.joint_angpos[5])
            self.joint7_pos_pub.publish(self.joint_angpos[6])

            #plot
            self.pub1.publish(self.A07[0,3])
            self.pub2.publish(pxdot)
            self.pub3.publish(pzdot)


            self.pub_rate.sleep()

    def turn_off(self):
        pass

def controller_py():
    # Starts a new node
    rospy.init_node('controller_node', anonymous=True)
    # Reading parameters set in launch file
    rate = rospy.get_param("/rate")

    controller = xArm7_controller(rate)
    rospy.on_shutdown(controller.turn_off)
    rospy.spin()

if __name__ == '__main__':
    try:
        controller_py()
    except rospy.ROSInterruptException:
        pass

