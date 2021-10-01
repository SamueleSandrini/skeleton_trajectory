#! /usr/bin/env python3

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo, Image
from RealSense import RealSense
import cv2
import mediapipe as mp

#import matplotlib.pyplot as plt
def my_hook():
    print("Programma terminato")

def main():
    # Subscriber node
        rospy.init_node("nodoSubscriberRealsense",anonymous=True)

        #Realsense object to menage acquisition
        realsense=RealSense()
        realsense.getCameraParam()  #For subscribe camera info usefull for Deprojection
        realsense.waitCameraInfo()
        while not rospy.is_shutdown():
            realsense.acquire()


if __name__=="__main__":
    main()
