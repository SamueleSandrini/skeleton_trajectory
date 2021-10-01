#! /usr/bin/env python3

import rospy
from KeypointsAdvancedFiltering import KeypointsAdvancedFiltering

def main():
    rospy.init_node("KalmanLimbs",anonymous=True)
    print("Limbs")
    keypointsFilter=KeypointsAdvancedFiltering()
    keypointsFilter.subAndFilter()


if __name__=="__main__":
    main()
