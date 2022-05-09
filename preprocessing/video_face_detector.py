from .image_face_detector import ImageFaceDetector
from .frame_extractor import FrameExtractor
from cv2 import cv2
from os import path

class VideoFaceDetector:

  def __init__(self, predictor_path):
    self.imageDetector = ImageFaceDetector(predictor_path=predictor_path)
    self.videoExtractor = FrameExtractor()

  def preprocess_video(self, video: str, targetFramerate=30.0, targetFrames = None, callback = None, savePath=None):
    frames = self.videoExtractor.extract_from(video, targetFramerate, targetFrames)

    result = []
    
    for (frameIndex, frame) in enumerate(frames):
      faces = self.imageDetector.preprocess_image(frame)
      for (faceIndex, face) in enumerate(faces):

        if callback is not None:
          callback(face)

        if savePath is not None:
          basename = path.basename(video)
          videoname = path.splitext(basename)[0]
          (faceOrig, faceAligned) = face
          self._save_face(savePath, videoname, frameIndex, faceIndex, faceAligned)
          
        result.append(face)
    return result

  def _save_face(self, path, name, frameIndex, faceIndex, face):
    cv2.imwrite(f"{path}/{name}-{frameIndex}-{faceIndex}.jpg", face)


