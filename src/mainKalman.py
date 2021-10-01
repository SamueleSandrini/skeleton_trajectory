#! /usr/bin/env python3

import rospy
from KeypointsFilter import KeypointsFilter

def main():
    rospy.init_node("Kalman",anonymous=True)
    print("Base")
    keypointsFilter=KeypointsFilter()
    keypointsFilter.subAndFilter()


if __name__=="__main__":
    main()
