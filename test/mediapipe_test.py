# import mediapipe as mp
# import cv2 as cv

# image = cv.imread('test.jpg')
# cv.imshow('image', image)
# cv.waitKey(0)
# cv.destroyAllWindows()
# pose_detector = mp.solutions.pose.Pose()

# result = pose_detector.process(image)

# print(result.pose_landmarks)
# print(type(result))
# print(type(result.pose_landmarks))
# print(type(result.pose_landmarks.landmark))

# for landmark in result.pose_landmarks.landmark:
#     print(landmark)
# from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n-pose.pt")

# result = model.predict("test2.jpeg")
#print keypoints
# for single_instance in result:
#     print(single_instance.keypoints)

# import torch
# from ultralytics.engine.results import Keypoints

# Creiamo un tensore con i keypoints: (x, y, confidenza)
# Supponiamo di avere 2 persone e 3 keypoints a testa (per esempio)
# data = torch.tensor([
#     [[100, 200, 0.9], [150, 250, 0.8], [120, 180, 0.95]],  # Persona 1
#     [[200, 300, 0.85], [250, 350, 0.75], [220, 280, 0.9]]   # Persona 2
# ])

# Creiamo l'oggetto Keypoints
# keypoints_obj = Keypoints(data)

# Verifica
# print(keypoints_obj)

# print(type(result))

# class PoseLandmark(enum.IntEnum):
#   """The 33 pose landmarks."""
#   NOSE = 0
#   LEFT_EYE_INNER = 1
#   LEFT_EYE = 2
#   LEFT_EYE_OUTER = 3
#   RIGHT_EYE_INNER = 4
#   RIGHT_EYE = 5
#   RIGHT_EYE_OUTER = 6
#   LEFT_EAR = 7
#   RIGHT_EAR = 8
#   MOUTH_LEFT = 9
#   MOUTH_RIGHT = 10
#   LEFT_SHOULDER = 11
#   RIGHT_SHOULDER = 12
#   LEFT_ELBOW = 13
#   RIGHT_ELBOW = 14
#   LEFT_WRIST = 15
#   RIGHT_WRIST = 16
#   LEFT_PINKY = 17
#   RIGHT_PINKY = 18
#   LEFT_INDEX = 19
#   RIGHT_INDEX = 20
#   LEFT_THUMB = 21
#   RIGHT_THUMB = 22
#   LEFT_HIP = 23
#   RIGHT_HIP = 24
#   LEFT_KNEE = 25
#   RIGHT_KNEE = 26
#   LEFT_ANKLE = 27
#   RIGHT_ANKLE = 28
#   LEFT_HEEL = 29
#   RIGHT_HEEL = 30
#   LEFT_FOOT_INDEX = 31
#   RIGHT_FOOT_INDEX = 32

