from cv2 import cv2
from face_aligment import FaceAligner
from landmarks_utils import rect_to_bb
from image_utils import resize
import numpy as np
import dlib


class ImageFaceDetector:

  def __init__(self, faceSize=256, imageSize=800):
    self.detector = dlib.get_frontal_face_detector()
    self.faceSize = faceSize
    self.imageSize = imageSize
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    self.faceAligner = FaceAligner(predictor, desiredLeftEye=(0.38, 0.38), desiredFaceWidth=self.faceSize)


  def _detect_faces(self, grayImage):
    rects = self.detector(grayImage, 2)
    return rects


  def preprocess_image_from_file(self, image: str):
    readedImage = cv2.imread(image)
    return self.preprocess_image(readedImage)


  def preprocess_image(self, image: np.ndarray):
    image = resize(image, width=self.imageSize)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detectedFacesRect = self._detect_faces(grayImage)
    facesTuples = []
    
    for rect in detectedFacesRect:
      (x, y, w, h) = rect_to_bb(rect)
      originalFace = resize(image[y:y + h, x:x + w], width=self.faceSize)
      alignedFace = self.faceAligner.align(image, grayImage, rect)
      facesTuples.append((originalFace, alignedFace))

    return facesTuples

