from cv2 import cv2
from os import path


class FrameExtractor:

  def _write_image(self, savePath, image, id, videoname):
    cv2.imwrite(f"{savePath}/{videoname}-{id}.jpg", image)


  def extract_from(self, video: str, targetFramerate: float = 30.0, targetFrames: int = None, savePath: str = None):
    vidcap = cv2.VideoCapture(video)
    
    basename = path.basename(video)
    videoname = path.splitext(basename)[0]

    frames = self._extract(vidcap, targetFramerate, targetFrames)
    if savePath is not None:
      for index, frame in enumerate(frames):
         self._write_image(savePath, frame, index, videoname)

    vidcap.release()
    return frames


  def _extract(self, vidcap, targetFramerate: float, targetFrames: int = None):
    extractedFrames = []
    oldPosition = 0
    count = 0
    
    (success, image) = vidcap.read()
    interval = targetFramerate / vidcap.get(cv2.CAP_PROP_FPS)

    while success:
      if int(count) > oldPosition :
        oldPosition = int(count)
        extractedFrames.append(image)

      (success, image) = vidcap.read()
      
      count += interval

      if targetFrames == oldPosition and targetFrames is not None:
        break
      
    print("Total extracted frames:", len(extractedFrames))
    return extractedFrames

