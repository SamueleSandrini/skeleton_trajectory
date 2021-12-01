#! /usr/bin/env python3

import rospy
from KeypointsAdvancedFiltering import KeypointsAdvancedFiltering

def main():
    rospy.init_node("kinematic_kalman",anonymous=True)
    rospy.loginfo("Kinematic Kalman")

    keypointsFilter=KeypointsAdvancedFiltering()
    keypointsFilter.subAndFilter()


if __name__=="__main__":
    main()
