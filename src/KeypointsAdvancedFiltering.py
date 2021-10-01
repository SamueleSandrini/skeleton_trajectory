#! /usr/bin/env python

import rospy
from geometry_msgs.msg import Pose, Point, Quaternion, PoseArray, Transform, Vector3, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
from KalmanFilter import KalmanFilter
from KalmanFilterLimbs import KalmanFilterLimbs
from scipy.spatial.transform import Rotation as R
from math import sin, cos

import time
import mediapipe as mp
import tf2_ros

SKELETON_MARKER_SUB = '/skeleton_marker'
KEYPOINTS_FILTERED = '/keypoints_filtered'
SKELETON_FILTERED = '/skeleton_filtered'
SKELETON_FILTERED_ARRAY = '/poses'    #/skeleton_filtered_PoseArray
LIMBS_FILTERED = '/limbs_filtered'
JOINT = '/jointLimb'

CAMERA_FRAME = "camera_color_optical_frame"

N_KEYPOINTS = 33
N_LIMBS = 4
BUST_KEYPOINTS = [11,12,23,24]
LEFT_ARM_KEYPOINTS = [11, 13, 15]
RIGHT_ARM_KEYPOINTS = [12, 14, 16]
LEFT_LEG_KEYPOINTS = [23, 25, 27]
RIGHT_LEG_KEYPOINTS = [24, 26, 28]

class KeypointsAdvancedFiltering():
    def __init__(self):
        """
        Class builder
        """
        self.isKeypointTracked = np.zeros((N_KEYPOINTS,),dtype=bool)
        self.keypointsFilters = np.empty((N_KEYPOINTS,),dtype=object)
        #For limbs filtering
        self.limbsFilters = np.empty((N_LIMBS,),dtype=object)
        self.isLimbTracked = np.zeros((N_LIMBS,),dtype=bool)
        self.isKeypointPresent = np.zeros((N_KEYPOINTS,),dtype=bool)

        self.timeKeypoint = np.zeros((N_KEYPOINTS,))
        self.timeLimbs = np.zeros((N_LIMBS,))

        #Create array of KalmanFilter object one for each keypoint
        for k in range(0,N_KEYPOINTS):
            self.keypointsFilters[k]=KalmanFilter()
        for k in range(0,N_LIMBS):
            self.limbsFilters[k]=KalmanFilterLimbs()
            pass

        #Subscriber
        self.subKeypoints = None

        #Message general definition
        self.marker = Marker()
        self.marker.type = Marker.SPHERE
        self.marker.ns = "KeypointsFiltered"
        self.marker.header.frame_id = CAMERA_FRAME
        self.marker.header.stamp = rospy.Time.now()
        self.marker.action = Marker.ADD
        self.marker.scale.x = 0.04
        self.marker.scale.y = 0.04
        self.marker.scale.z = 0.04

        self.marker.color.r = 0.0
        self.marker.color.g = 1.0
        self.marker.color.b = 0.0

        self.marker.color.a =1.0

        #Publisher keypoint filtered
        self.pubMarkerFiltered = rospy.Publisher(KEYPOINTS_FILTERED, Marker, queue_size = 100)

        """
        Da qui in poi
        """
        self.listKeypoints = []
        self.listOfIndexesPres = []

        self.pubSkeletonFiltered = rospy.Publisher(SKELETON_FILTERED, Marker, queue_size = 100)
        self.pubSkeletonFilteredArray = rospy.Publisher(SKELETON_FILTERED_ARRAY,PoseArray, queue_size = 100)

        #Publisher for limbs filtering
        self.pubLimbsFiltered = rospy.Publisher(LIMBS_FILTERED, Marker, queue_size = 100)

        #Publisher limbs joints
        self.pubJoint = rospy.Publisher(JOINT, JointState, queue_size = 100)

        # Mediapipe Utils and pose
        self.mp_pose = mp.solutions.pose




    def callbackKeypoint(self, keypoints):
        """
        Callback method to retrieve keypoints and apply Kalman Filter
        @param: keypoint: ROS Marker message
        """
        # Message costructor for all segments Filtering
        self.skeleton = Marker()
        self.skeleton.type = Marker.LINE_LIST
        self.skeleton.ns = "SegmentsFiltered"
        self.skeleton.header.frame_id = CAMERA_FRAME

        self.skeleton.id=100
        self.skeleton.action = Marker.ADD
        self.skeleton.scale.x=0.03
        self.skeleton.color.r = 0.0
        self.skeleton.color.g = 2.0
        self.skeleton.color.b = 0.0
        self.skeleton.color.a =0.7

        # Message costructor for Limbs Filtering
        self.limbs = Marker()
        self.limbs.type = Marker.LINE_LIST
        self.limbs.ns = "SegmentsFiltered"
        self.limbs.header.frame_id = CAMERA_FRAME

        self.limbs.action = Marker.ADD
        self.limbs.scale.x=0.03
        self.limbs.color.r = 0.0
        self.limbs.color.g = 0.0
        self.limbs.color.b = 2.0
        self.limbs.color.a =0.7
        self.limbs.header.stamp = rospy.Time.now()

        rawKeypoint={}
        for keypoint in keypoints.markers:
            idKeypoint=keypoint.id

            self.isKeypointPresent[idKeypoint]=True

            #Measurements
            y=np.ones((3,))*[keypoint.pose.position.x,keypoint.pose.position.y,keypoint.pose.position.z]
            rawKeypoint[idKeypoint]=y
            #Time of this keypoints
            self.timeKeypoint[idKeypoint]=time.time()


            #Check if time is too old
            for id,singleTime in enumerate(self.timeKeypoint):
                #print(singleTime)
                if singleTime!=0.0:
                    deltaT=time.time()-singleTime
                else:
                    deltaT=0
                #print("Id Keypoint: {}, tempo:{}".format(idKeypoint,deltaT))
                if deltaT>1.0/10.0:
                    if self.isKeypointTracked[id]:
                        self.isKeypointTracked[id]=False
                        print("Keypoint number: {} too old, time:{}".format(id,deltaT))
                        print("DeltaT: ",deltaT)
                    else:
                        pass
                        #print("Keypoint number: {} già troppo vecchio:".format(id))

            #Apply Kalaman Filter if tracked or initialize it if first time
            if not self.isKeypointTracked[idKeypoint]:
                self.isKeypointTracked[idKeypoint]=self.keypointsFilters[idKeypoint].initialize(y)
                yFiltered = self.keypointsFilters[idKeypoint].getYAfterInitialize()                 #Actually it is not filtered but in this way only one Pose msg.
                print("Keypoint number: {} is re-initialized".format(idKeypoint))
            else:
                yFiltered = self.keypointsFilters[idKeypoint].update(y,idKeypoint)



            #Populate message (this is message of keypoints that are seen by mediapipe and not tacked in open loop)
            self.marker.id = idKeypoint
            self.marker.pose = Pose(Point(yFiltered[0],yFiltered[1],yFiltered[2]),Quaternion(0,0,0,1))

            #self.pubMarkerFiltered.publish(self.marker)


            self.listOfIndexesPres.append(idKeypoint)
            self.listKeypoints.append(self.marker.pose)

        self.skeleton.header.stamp = rospy.Time.now()

        #Iterate all non detected keypoints and update it in open loop if the time passed is not too much (approximate 3 samples)
        for idKeypointNotDetected in np.setdiff1d(np.array(range(0,N_KEYPOINTS)),self.listOfIndexesPres):
            if self.isKeypointTracked[idKeypointNotDetected]:               #Not too much time
                yModel = self.keypointsFilters[idKeypointNotDetected].updateOpenLoop()  #It returns
                #Populate message
                self.marker.id = idKeypointNotDetected
                self.marker.pose = Pose(Point(yModel[0],yModel[1],yModel[2]),Quaternion(0,0,0,1))
                #self.pubMarkerFiltered.publish(self.marker)
                self.listOfIndexesPres.append(idKeypointNotDetected)
                self.listKeypoints.append(self.marker.pose)
                print("Keypoint number: {} is in open Loop".format(idKeypointNotDetected))
        #B[np.where(np.isin(self.listKeypoints,BUST_KEYPOINTS))[0]].shape[0]==len(BUST_KEYPOINTS)


        #Check at which limbs i can apply the Kalman filter
        if self.isKeypointTracked[BUST_KEYPOINTS].all():        #If i can define first reference frame
            PbustDxU=self.fromKeypointToPoint(self.listKeypoints[self.listOfIndexesPres.index(12)])
            PbustSxU=self.fromKeypointToPoint(self.listKeypoints[self.listOfIndexesPres.index(11)])
            PbustDxL=self.fromKeypointToPoint(self.listKeypoints[self.listOfIndexesPres.index(24)])
            PbustSxL=self.fromKeypointToPoint(self.listKeypoints[self.listOfIndexesPres.index(23)])
            if self.isKeypointPresent[LEFT_ARM_KEYPOINTS].all():
                #C7 Reference frame definition
                zAxis = ( PbustDxU - PbustSxU ) / np.linalg.norm( PbustDxU - PbustSxU )
                c7 = ( PbustDxU + PbustSxU ) / 2
                l5 = ( PbustDxL + PbustSxL ) / 2
                yAxis = ( c7 - l5 ) / np.linalg.norm( c7 - l5 )
                yAxis = yAxis - np.dot(yAxis,zAxis)/(np.dot(zAxis,zAxis))*zAxis
                yAxis = yAxis/np.linalg.norm(yAxis)
                xAxis = np.cross(yAxis,zAxis)
                matR_camera_c7=np.array([xAxis,yAxis,zAxis]).T

                self.showReferenceFrame(matR_camera_c7,c7,CAMERA_FRAME,"Frame_C7")
                self.showReferenceFrame(matR_camera_c7,PbustSxU,CAMERA_FRAME,"Frame_SpallaSx")

                #Compongo le matrici di roto-traslazione

                M_camera_c7 = np.concatenate((np.concatenate((matR_camera_c7,np.reshape(c7, (3,1))),axis=1),np.array([[0,0,0,1]])),axis=0)
                M_camera_spallaSx = np.concatenate((np.concatenate((matR_camera_c7,np.reshape(PbustSxU, (3,1))),axis=1),np.array([[0,0,0,1]])),axis=0)
                matR_camera_c7_inv = np.linalg.inv(matR_camera_c7)

                #Check che M_camera_spallaSx^-1 * PbustSxU = [0,0,0,1] (circa)
                #print("Check M_camera_spallaSx^-1 * PbustSxU = [0,0,0,1]: ", np.dot(np.linalg.inv(M_camera_spallaSx),np.concatenate([PbustSxU,[1]])))

                #These are filtered one P and G: if you want to use filtered one and not raw one you have to remove comments and add comments at following line
                #G_camera=self.fromKeypointToPoint(self.listKeypoints[self.listOfIndexesPres.index(LEFT_ARM_KEYPOINTS[1])])
                #P_camera=self.fromKeypointToPoint(self.listKeypoints[self.listOfIndexesPres.index(LEFT_ARM_KEYPOINTS[2])])
                G_camera=rawKeypoint[LEFT_ARM_KEYPOINTS[1]]
                P_camera=rawKeypoint[LEFT_ARM_KEYPOINTS[2]]

                #Computation of G respect to 11: G(11)=M11_camera * G(camera)
                G_11=np.dot(np.linalg.inv(M_camera_spallaSx),np.concatenate([G_camera,[1]]))                                   # [xG yG zG 1]
                P_11=np.dot(np.linalg.inv(M_camera_spallaSx),np.concatenate([P_camera,[1]]))                                   # [xP yP zP 1]

                #print("G rispetto 11: ", G_11)
                #print("P rispetto 11: ", P_11)

                yMeas=np.concatenate((G_11[:-1],P_11[:-1]))             # Concateno eccetto ultimo elemento (1)
                #print("Misura :",yMeas)

                if not self.isLimbTracked[0]:
                    # Initialize Kalman Filter of that limb
                    self.isLimbTracked[0]=self.limbsFilters[0].initialize(yMeas,matR_camera_c7_inv[:,2])
                    y_observed = self.limbsFilters[0].getYAfterInitialize()
                else:
                    # Update Kalman Filter of that limb
                    y_observed=self.limbsFilters[0].update(yMeas,matR_camera_c7_inv[:,2])
                G_cam_obs=np.dot(M_camera_spallaSx,np.concatenate([y_observed[0:3],[1]]))         # [xG yG zG 1]
                P_cam_obs=np.dot(M_camera_spallaSx,np.concatenate([y_observed[3:],[1]]))          # [xG yG zG 1]

                #Message building
                self.limbs.points.append(Point(PbustSxU[0],PbustSxU[1],PbustSxU[2]))
                self.limbs.points.append(Point(G_cam_obs[0],G_cam_obs[1],G_cam_obs[2]))
                self.limbs.points.append(Point(G_cam_obs[0],G_cam_obs[1],G_cam_obs[2]))
                self.limbs.points.append(Point(P_cam_obs[0],P_cam_obs[1],P_cam_obs[2]))
                self.pubLimbsFiltered.publish(self.limbs)

                #Joint Message
                jointMessage=self.buildJointMessage(["LeftArm_joint_1","LeftArm_joint_2","LeftArm_joint_3","LeftArm_joint_4"],self.limbsFilters[0].getJointsPosition(),self.limbsFilters[0].getJointsVelocity(),self.limbsFilters[0].getJointsAcceleration())
                self.pubJoint.publish(jointMessage)
                #Limbs length
                #lengthMessage=Float64MultiArray()
                self.replaceKeypoint(LEFT_ARM_KEYPOINTS[1:],G_cam_obs,P_cam_obs)


            else:
                self.isLimbTracked[0]=False
            if self.isKeypointPresent[RIGHT_ARM_KEYPOINTS].all():
                #C7 Reference frame definition
                zAxis = ( PbustDxU - PbustSxU ) / np.linalg.norm( PbustDxU - PbustSxU )
                c7 = ( PbustDxU + PbustSxU ) / 2
                l5 = ( PbustDxL + PbustSxL ) / 2
                yAxis = ( c7 - l5 ) / np.linalg.norm( c7 - l5 )
                yAxis = yAxis - np.dot(yAxis,zAxis)/(np.dot(zAxis,zAxis))*zAxis
                yAxis = yAxis/np.linalg.norm(yAxis)
                xAxis = np.cross(yAxis,zAxis)
                matR_camera_c7=np.array([xAxis,yAxis,zAxis]).T
                matR_camera_12=np.dot(np.array([xAxis,yAxis,zAxis]).T,np.array([[cos(np.pi), 0, sin(np.pi)],[0, 1, 0],[-sin(np.pi), 0, cos(np.pi)]]))
                #self.showReferenceFrame(matR_camera_c7,c7,CAMERA_FRAME,"Frame_C7")
                self.showReferenceFrame(matR_camera_12,PbustDxU,CAMERA_FRAME,"Frame_SpallaDx")

                #Compongo le matrici di roto-traslazione
                M_camera_c7 = np.concatenate((np.concatenate((matR_camera_c7,np.reshape(c7, (3,1))),axis=1),np.array([[0,0,0,1]])),axis=0)
                M_camera_spallaDx = np.concatenate((np.concatenate((matR_camera_12,np.reshape(PbustDxU, (3,1))),axis=1),np.array([[0,0,0,1]])),axis=0)
                matR_camera_l2_inv = np.linalg.inv(matR_camera_12)

                #Check che M_camera_spallaSx^-1 * PbustSxU = [0,0,0,1] (circa)
                #print("Check M_camera_spallaSx^-1 * PbustSxU = [0,0,0,1]: ", np.dot(np.linalg.inv(M_camera_spallaSx),np.concatenate([PbustSxU,[1]])))

                #These are filtered one P and G: if you want to use filtered one and not raw one you have to remove comments and add comments at following line
                #G_camera=self.fromKeypointToPoint(self.listKeypoints[self.listOfIndexesPres.index(RIGHT_ARM_KEYPOINTS[1])])
                #P_camera=self.fromKeypointToPoint(self.listKeypoints[self.listOfIndexesPres.index(RIGHT_ARM_KEYPOINTS[2])])
                G_camera=rawKeypoint[RIGHT_ARM_KEYPOINTS[1]]
                P_camera=rawKeypoint[RIGHT_ARM_KEYPOINTS[2]]

                #Computation of G respect to 12: G(12)=M12_camera * G(camera)
                G_12=np.dot(np.linalg.inv(M_camera_spallaDx),np.concatenate([G_camera,[1]]))                                   # [xG yG zG 1]
                P_12=np.dot(np.linalg.inv(M_camera_spallaDx),np.concatenate([P_camera,[1]]))                                   # [xP yP zP 1]


                yMeas=np.concatenate((G_12[:-1],P_12[:-1]))             # Concateno eccetto ultimo elemento (1)
                #print("Misura :",yMeas)

                if not self.isLimbTracked[1]:
                    # Initialize Kalman Filter of that limb
                    self.isLimbTracked[1]=self.limbsFilters[1].initialize(yMeas,matR_camera_l2_inv[:,2])
                    y_observed = self.limbsFilters[1].getYAfterInitialize()
                else:
                    # Update Kalman Filter of that limb
                    y_observed=self.limbsFilters[1].update(yMeas,matR_camera_l2_inv[:,2])

                G_cam_obs=np.dot(M_camera_spallaDx,np.concatenate([y_observed[0:3],[1]]))         # [xG yG zG 1]
                P_cam_obs=np.dot(M_camera_spallaDx,np.concatenate([y_observed[3:],[1]]))          # [xG yG zG 1]

                #Message building
                self.limbs.points.append(Point(PbustDxU[0],PbustDxU[1],PbustDxU[2]))
                self.limbs.points.append(Point(G_cam_obs[0],G_cam_obs[1],G_cam_obs[2]))
                self.limbs.points.append(Point(G_cam_obs[0],G_cam_obs[1],G_cam_obs[2]))
                self.limbs.points.append(Point(P_cam_obs[0],P_cam_obs[1],P_cam_obs[2]))
                self.pubLimbsFiltered.publish(self.limbs)

                #Joint Message
                jointMessage=self.buildJointMessage(["RightArm_joint_1","RightArm_joint_2","RightArm_joint_3","RightArm_joint_4"],self.limbsFilters[1].getJointsPosition(),self.limbsFilters[1].getJointsVelocity(),self.limbsFilters[1].getJointsAcceleration())
                self.pubJoint.publish(jointMessage)
                self.replaceKeypoint(RIGHT_ARM_KEYPOINTS[1:],G_cam_obs,P_cam_obs)
            else:
                self.isLimbTracked[1]=False
            if self.isKeypointPresent[LEFT_LEG_KEYPOINTS].all():
                #C7 Reference frame definition
                zAxis = ( PbustDxU - PbustSxU ) / np.linalg.norm( PbustDxU - PbustSxU )
                c7 = ( PbustDxU + PbustSxU ) / 2
                l5 = ( PbustDxL + PbustSxL ) / 2
                yAxis = ( c7 - l5 ) / np.linalg.norm( c7 - l5 )
                yAxis = yAxis - np.dot(yAxis,zAxis)/(np.dot(zAxis,zAxis))*zAxis
                yAxis = yAxis/np.linalg.norm(yAxis)
                xAxis = np.cross(yAxis,zAxis)
                matR_camera_c7=np.array([xAxis,yAxis,zAxis]).T

                self.showReferenceFrame(matR_camera_c7,c7,CAMERA_FRAME,"Frame_C7")
                self.showReferenceFrame(matR_camera_c7,PbustSxL,CAMERA_FRAME,"Frame_LegSx")

                #Compongo le matrici di roto-traslazione

                M_camera_c7 = np.concatenate((np.concatenate((matR_camera_c7,np.reshape(c7, (3,1))),axis=1),np.array([[0,0,0,1]])),axis=0)
                M_camera_legSx = np.concatenate((np.concatenate((matR_camera_c7,np.reshape(PbustSxL, (3,1))),axis=1),np.array([[0,0,0,1]])),axis=0)
                matR_camera_c7_inv = np.linalg.inv(matR_camera_c7)

                #Check che M_camera_spallaSx^-1 * PbustSxU = [0,0,0,1] (circa)
                #print("Check M_camera_spallaSx^-1 * PbustSxU = [0,0,0,1]: ", np.dot(np.linalg.inv(M_camera_spallaSx),np.concatenate([PbustSxU,[1]])))

                #These are filtered one P and G: if you want to use filtered one and not raw one you have to remove comments and add comments at following line
                #G_camera=self.fromKeypointToPoint(self.listKeypoints[self.listOfIndexesPres.index(LEFT_ARM_KEYPOINTS[1])])
                #P_camera=self.fromKeypointToPoint(self.listKeypoints[self.listOfIndexesPres.index(LEFT_ARM_KEYPOINTS[2])])
                G_camera=rawKeypoint[LEFT_LEG_KEYPOINTS[1]]
                P_camera=rawKeypoint[LEFT_LEG_KEYPOINTS[2]]

                #Computation of G respect to 11: G(11)=M11_camera * G(camera)
                G_11=np.dot(np.linalg.inv(M_camera_legSx),np.concatenate([G_camera,[1]]))                                   # [xG yG zG 1]
                P_11=np.dot(np.linalg.inv(M_camera_legSx),np.concatenate([P_camera,[1]]))                                   # [xP yP zP 1]

                #print("G rispetto 11: ", G_11)
                #print("P rispetto 11: ", P_11)

                yMeas=np.concatenate((G_11[:-1],P_11[:-1]))             # Concateno eccetto ultimo elemento (1)
                #print("Misura :",yMeas)

                if not self.isLimbTracked[2]:
                    # Initialize Kalman Filter of that limb
                    self.isLimbTracked[2]=self.limbsFilters[2].initialize(yMeas,matR_camera_c7_inv[:,2])
                    y_observed = self.limbsFilters[2].getYAfterInitialize()
                else:
                    # Update Kalman Filter of that limb
                    y_observed=self.limbsFilters[2].update(yMeas,matR_camera_c7_inv[:,2])
                G_cam_obs=np.dot(M_camera_legSx,np.concatenate([y_observed[0:3],[1]]))         # [xG yG zG 1]
                P_cam_obs=np.dot(M_camera_legSx,np.concatenate([y_observed[3:],[1]]))          # [xG yG zG 1]

                #Message building
                self.limbs.points.append(Point(PbustSxL[0],PbustSxL[1],PbustSxL[2]))
                self.limbs.points.append(Point(G_cam_obs[0],G_cam_obs[1],G_cam_obs[2]))
                self.limbs.points.append(Point(G_cam_obs[0],G_cam_obs[1],G_cam_obs[2]))
                self.limbs.points.append(Point(P_cam_obs[0],P_cam_obs[1],P_cam_obs[2]))
                self.pubLimbsFiltered.publish(self.limbs)

                #Joint Message
                jointMessage=self.buildJointMessage(["LeftLeg_joint_1","LeftLeg_joint_2","LeftLeg_joint_3","LeftLeg_joint_4"],self.limbsFilters[2].getJointsPosition(),self.limbsFilters[2].getJointsVelocity(),self.limbsFilters[2].getJointsAcceleration())
                self.pubJoint.publish(jointMessage)
                #Limbs length
                #lengthMessage=Float64MultiArray()
                self.replaceKeypoint(LEFT_LEG_KEYPOINTS[1:],G_cam_obs,P_cam_obs)

            else:
                self.isLimbTracked[2]=False
            if self.isKeypointPresent[RIGHT_LEG_KEYPOINTS].all():
                #C7 Reference frame definition
                zAxis = ( PbustDxU - PbustSxU ) / np.linalg.norm( PbustDxU - PbustSxU )
                c7 = ( PbustDxU + PbustSxU ) / 2
                l5 = ( PbustDxL + PbustSxL ) / 2
                yAxis = ( c7 - l5 ) / np.linalg.norm( c7 - l5 )
                yAxis = yAxis - np.dot(yAxis,zAxis)/(np.dot(zAxis,zAxis))*zAxis
                yAxis = yAxis/np.linalg.norm(yAxis)
                xAxis = np.cross(yAxis,zAxis)
                matR_camera_c7=np.array([xAxis,yAxis,zAxis]).T
                matR_camera_12=np.dot(np.array([xAxis,yAxis,zAxis]).T,np.array([[cos(np.pi), 0, sin(np.pi)],[0, 1, 0],[-sin(np.pi), 0, cos(np.pi)]]))
                #self.showReferenceFrame(matR_camera_c7,c7,CAMERA_FRAME,"Frame_C7")
                self.showReferenceFrame(matR_camera_12,PbustDxL,CAMERA_FRAME,"Frame_SpallaDx")

                #Compongo le matrici di roto-traslazione
                M_camera_c7 = np.concatenate((np.concatenate((matR_camera_c7,np.reshape(c7, (3,1))),axis=1),np.array([[0,0,0,1]])),axis=0)
                M_camera_spallaDx = np.concatenate((np.concatenate((matR_camera_12,np.reshape(PbustDxL, (3,1))),axis=1),np.array([[0,0,0,1]])),axis=0)
                matR_camera_l2_inv = np.linalg.inv(matR_camera_12)

                #Check che M_camera_spallaSx^-1 * PbustSxU = [0,0,0,1] (circa)
                #print("Check M_camera_spallaSx^-1 * PbustSxU = [0,0,0,1]: ", np.dot(np.linalg.inv(M_camera_spallaSx),np.concatenate([PbustSxU,[1]])))

                #These are filtered one P and G: if you want to use filtered one and not raw one you have to remove comments and add comments at following line
                #G_camera=self.fromKeypointToPoint(self.listKeypoints[self.listOfIndexesPres.index(RIGHT_ARM_KEYPOINTS[1])])
                #P_camera=self.fromKeypointToPoint(self.listKeypoints[self.listOfIndexesPres.index(RIGHT_ARM_KEYPOINTS[2])])
                G_camera=rawKeypoint[RIGHT_LEG_KEYPOINTS[1]]
                P_camera=rawKeypoint[RIGHT_LEG_KEYPOINTS[2]]

                #Computation of G respect to 12: G(12)=M12_camera * G(camera)
                G_12=np.dot(np.linalg.inv(M_camera_spallaDx),np.concatenate([G_camera,[1]]))                                   # [xG yG zG 1]
                P_12=np.dot(np.linalg.inv(M_camera_spallaDx),np.concatenate([P_camera,[1]]))                                   # [xP yP zP 1]


                yMeas=np.concatenate((G_12[:-1],P_12[:-1]))             # Concateno eccetto ultimo elemento (1)
                #print("Misura :",yMeas)

                if not self.isLimbTracked[3]:
                    # Initialize Kalman Filter of that limb
                    self.isLimbTracked[3]=self.limbsFilters[3].initialize(yMeas,matR_camera_l2_inv[:,2])
                    y_observed = self.limbsFilters[3].getYAfterInitialize()
                else:
                    # Update Kalman Filter of that limb
                    y_observed=self.limbsFilters[3].update(yMeas,matR_camera_l2_inv[:,2])
                #print("Covariance: ",self.limbsFilters[3].getCartesianCovariance())
                G_cam_obs=np.dot(M_camera_spallaDx,np.concatenate([y_observed[0:3],[1]]))         # [xG yG zG 1]
                P_cam_obs=np.dot(M_camera_spallaDx,np.concatenate([y_observed[3:],[1]]))          # [xG yG zG 1]

                #Message building
                self.limbs.points.append(Point(PbustDxL[0],PbustDxL[1],PbustDxL[2]))
                self.limbs.points.append(Point(G_cam_obs[0],G_cam_obs[1],G_cam_obs[2]))
                self.limbs.points.append(Point(G_cam_obs[0],G_cam_obs[1],G_cam_obs[2]))
                self.limbs.points.append(Point(P_cam_obs[0],P_cam_obs[1],P_cam_obs[2]))
                self.pubLimbsFiltered.publish(self.limbs)

                #Joint Message
                jointMessage=self.buildJointMessage(["RightLeg_joint_1","RightLeg_joint_2","RightLeg_joint_3","RightLeg_joint_4"],self.limbsFilters[3].getJointsPosition(),self.limbsFilters[3].getJointsVelocity(),self.limbsFilters[3].getJointsAcceleration())
                self.pubJoint.publish(jointMessage)
                self.replaceKeypoint(RIGHT_LEG_KEYPOINTS[1:],G_cam_obs,P_cam_obs)
            else:
                self.isLimbTracked[3]=False

        #Create pose array of all filtered KeyPoints
        skeletonArrayFiltered = PoseArray()
        skeletonArrayFiltered.header.stamp = rospy.Time.now()
        skeletonArrayFiltered.header.frame_id = CAMERA_FRAME
        skeletonArrayFiltered.poses = self.listKeypoints
        self.pubSkeletonFilteredArray.publish(skeletonArrayFiltered)


        #Iterate connections between keypoints: skeleton information (plot)
        n=0
        for connection in self.mp_pose.POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]

            if (start_idx in self.listOfIndexesPres) and (end_idx in self.listOfIndexesPres):
                 #self.skeleton.id = n
                 index_start=self.listOfIndexesPres.index(start_idx)              # 1 2 10 21 : i don't have to go in 21th element of landmark List but 4th
                 index_end=self.listOfIndexesPres.index(end_idx)
                 self.skeleton.points.append(self.listKeypoints[index_start].position)
                 self.skeleton.points.append(self.listKeypoints[index_end].position)
                 n+=1
        self.pubSkeletonFiltered.publish(self.skeleton)

        n=0
        for keyp in self.listKeypoints:
            self.marker.header.stamp = rospy.Time.now()
            self.marker.id = self.listOfIndexesPres[n]
            self.marker.pose = keyp
            self.pubMarkerFiltered.publish(self.marker)
            n+=1

        #Reset presences when one skeleton is completed
        self.listOfIndexesPres = []
        self.listKeypoints = []
        self.isKeypointPresent = np.zeros((N_KEYPOINTS,),dtype=bool)

    def subAndFilter(self):
        self.subscriberKeypoint=rospy.Subscriber(SKELETON_MARKER_SUB,MarkerArray,self.callbackKeypoint)
        rospy.spin()
    def fromKeypointToPoint(self,keypoint):
        return np.array([keypoint.position.x, keypoint.position.y, keypoint.position.z])

    def showReferenceFrame(self,matR,tran,originFrame,destinationFrame):
        """
        Method for show (publish) a reference frame
        @param matR: rotation matrix
        @param tran: translation Vector
        @param originFrame: base frame
        @param destinationFrame: child frame
        """
        t0=Transform()
        t0TS=TransformStamped()
        r = R.from_matrix(matR)
        quatR=r.as_quat()
        t0.rotation=Quaternion(quatR[0],quatR[1],quatR[2],quatR[3])
        t0.translation=Vector3(tran[0],tran[1],tran[2])
        t0TS.header.frame_id=originFrame
        t0TS.header.stamp= rospy.Time.now()
        t0TS.child_frame_id=destinationFrame
        t0TS.transform=t0
        br = tf2_ros.TransformBroadcaster()
        br.sendTransform(t0TS)

    def buildJointMessage(self,nameJ,positionJ,velocityJ,effortJ):
        """
        Utility method for build Joint Message
        @param name: joints name
        @param position: joints position
        @param velocity: joints velocity
        """
        jointMessage=JointState()
        jointMessage.header.stamp= rospy.Time.now()
        jointMessage.name=nameJ
        jointMessage.position=positionJ
        jointMessage.velocity=velocityJ
        jointMessage.effort=effortJ
        return jointMessage

    def replaceKeypoint(self, indexesToChange,G,P):
        """
        Utility method for replace keypoints of limb
        @param indexesToChange: list of indexes to change
        @param G: meddle point coordinate
        @oaram P: ent point coordinate
        """
        indexForList=self.listOfIndexesPres.index(indexesToChange[0])
        self.listKeypoints[indexForList]=Pose(Point(G[0],G[1],G[2]),Quaternion(0,0,0,1))

        indexForList=self.listOfIndexesPres.index(indexesToChange[1])
        self.listKeypoints[indexForList]=Pose(Point(P[0],P[1],P[2]),Quaternion(0,0,0,1))
