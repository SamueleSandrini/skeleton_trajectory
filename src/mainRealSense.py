#! /usr/bin/env python3

import rospy
from RealSense import RealSense

def my_hook():
    rospy.loginfo("Shutting down node...")

def main():
    # Subscriber node
    rospy.init_node("acquisition_node",anonymous=True)

    # Realsense object to menage acquisition
    realsense=RealSense()
    realsense.getCameraParam()  # For subscribe camera info usefull for Deprojection
    realsense.waitCameraInfo()
    while not rospy.is_shutdown():
        realsense.acquire()


if __name__=="__main__":
    main()
