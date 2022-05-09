# from google.colab.patches import cv2_imshow
import sys

sys.path.append('/content/drive/MyDrive/face_detector')
from preprocessing import ImageFaceDetector

from cv2 import cv2


detector = ImageFaceDetector('/content/drive/MyDrive/face_detector/preprocessing/shape_predictor_68_face_landmarks.dat')

faces = detector.visualize_facial_landmarks('/content/drive/MyDrive/EfficientDet/test/test.jpg')
cv2.imwrite('/content/drive/MyDrive/face_detector/landmark.jpg', faces[0])