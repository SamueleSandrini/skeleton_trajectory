#! /usr/bin/env python3

import rospy
from KeypointsFilter import KeypointsFilter

def main():
    rospy.init_node("Indipendent_Kalman",anonymous=True)
    print("Indipendent_Kalman")
    keypointsFilter=KeypointsFilter()
    keypointsFilter.subAndFilter()


if __name__=="__main__":
    main()
