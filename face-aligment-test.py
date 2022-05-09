# from google.colab.patches import cv2_imshow
import sys

sys.path.append('/content/drive/MyDrive/face_detector')
from preprocessing import ImageFaceDetector

from cv2 import cv2


detector = ImageFaceDetector('/content/drive/MyDrive/face_detector/preprocessing/shape_predictor_68_face_landmarks.dat')

(orig, aligned) = detector.preprocess_image_from_file('/content/drive/MyDrive/test_photos/LHUX-L0Z9Oc.jpg')[0]
cv2.imwrite('/content/drive/MyDrive/face_detector/face-aligment.jpg', aligned)