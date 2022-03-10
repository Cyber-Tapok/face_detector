from .image_face_detector import ImageFaceDetector
from .frame_extractor import FrameExtractor

class VideoFaceDetector:

  def __init__(self, predictor_path):
    self.imageDetector = ImageFaceDetector(predictor_path=predictor_path)
    self.videoExtractor = FrameExtractor()

  def preprocess_video(self, video: str, targetFramerate=30.0, targetFrames = None, callback = None):
    frames = self.videoExtractor.extract_from(video, targetFramerate, targetFrames)

    result = []
    
    for frame in frames:
      faces = self.imageDetector.preprocess_image(frame)
      for face in faces:
        if(callback is not None):
          callback(face)
        result.append(face)
    return result

