import cv2


def resize(image, width: int = None, height: int = None, inter=cv2.INTER_AREA):
  dim = None
  (h, w) = image.shape[:2]
  if width is None and height is None:
    return image
    
  print(f'h: {h}, w: {w}')
  if width is None:
    r = height / float(h)
    dim = (int(w * r), height)
  else:
    r = width / float(w)
    dim = (width, int(h * r))

  resized = cv2.resize(image, dim, interpolation=inter)
  return resized