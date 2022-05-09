from collections import OrderedDict
from typing import NamedTuple
import numpy as np
from cv2 import cv2


FACIAL_LANDMARKS_68_IDXS = OrderedDict([
  ("mouth", (48, 68)),
  ("inner_mouth", (60, 68)),
  ("right_eyebrow", (17, 22)),
  ("left_eyebrow", (22, 27)),
  ("right_eye", (36, 42)),
  ("left_eye", (42, 48)),
  ("nose", (27, 36)),
  ("jaw", (0, 17))
])

FACIAL_LANDMARKS_5_IDXS = OrderedDict([
  ("right_eye", (2, 3)),
  ("left_eye", (0, 1)),
  ("nose", (4))
])


def rect_to_bb(rect):
  x = rect.left()
  y = rect.top()
  w = rect.right() - x
  h = rect.bottom() - y

  return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
  coords = np.zeros((shape.num_parts, 2), dtype=dtype)

  for i in range(0, shape.num_parts):
    coords[i] = (shape.part(i).x, shape.part(i).y)

  return coords

def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
	output = image.copy()
	for i, point in enumerate(shape):
		x = point[0]
		y = point[1]
		cv2.circle(output, (x,y), radius=4, color=(0, 0, 255), thickness=-1)
		cv2.putText(output, str(i + 1), (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
		cv2.putText(output, str(i + 1), (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
	return output