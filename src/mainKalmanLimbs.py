#! /usr/bin/env python3

import rospy
from KeypointsAdvancedFiltering import KeypointsAdvancedFiltering

def main():
    rospy.init_node("Kinematic_Kalman",anonymous=True)
    print("Kinematic_Kalman")
    keypointsFilter=KeypointsAdvancedFiltering()
    keypointsFilter.subAndFilter()


if __name__=="__main__":
    main()
