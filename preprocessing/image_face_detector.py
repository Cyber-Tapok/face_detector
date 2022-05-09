from cv2 import cv2
from .face_aligment import FaceAligner
from .landmarks_utils import rect_to_bb, visualize_facial_landmarks, shape_to_np
from .image_utils import resize
import numpy as np
import dlib


class ImageFaceDetector:

  def __init__(self, predictor_path, faceSize=512, imageSize=800):
    self.detector = dlib.get_frontal_face_detector()
    self.faceSize = faceSize
    self.imageSize = imageSize
    self.predictor = dlib.shape_predictor(predictor_path)
    self.faceAligner = FaceAligner(self.predictor, desiredLeftEye=(0.38, 0.38), desiredFaceWidth=self.faceSize)


  def _detect_faces(self, grayImage):
    rects = self.detector(grayImage, 2)
    return rects

  def visualize_facial_landmarks(self, image: str):
    image = cv2.imread(image)
    image = resize(image, width=self.imageSize)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detectedFacesRect = self._detect_faces(grayImage)
    faces = []

    for rect in detectedFacesRect:
      (x, y, w, h) = rect_to_bb(rect)
      imageRect = image[y:y + h, x:x + w]
      (h_r, w_r) = imageRect.shape[:2]
      if h_r != 0 and w_r != 0:
        shape = self.predictor(grayImage, rect)
        shape = shape_to_np(shape)
        alignedFace = visualize_facial_landmarks(image, shape)
        faces.append(alignedFace)

    return faces

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
      imageRect = image[y:y + h, x:x + w]
      (h_r, w_r) = imageRect.shape[:2]
      if h_r != 0 and w_r != 0:
        originalFace = resize(imageRect, width=self.faceSize)
        alignedFace = self.faceAligner.align(image, grayImage, rect)
        facesTuples.append((originalFace, alignedFace))

    return facesTuples

