#! /usr/bin/env python3

import rospy
from KeypointsFilter import KeypointsFilter

def main():
    rospy.init_node("indipendent_kalman",anonymous=True)
    rospy.loginfo("Indipendent Kalman")

    keypointsFilter=KeypointsFilter()
    keypointsFilter.subAndFilter()


if __name__=="__main__":
    main()
