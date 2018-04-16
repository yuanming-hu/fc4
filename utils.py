import math
import cv2
import numpy as np
import os
from math import *
import numpy as np
import sys

UV_SCALE = 0.75


def set_target_gpu(gpus):
  os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))


def angular_error(estimation, ground_truth):
  return acos(
      np.clip(
          np.dot(estimation, ground_truth) / np.linalg.norm(estimation) /
          np.linalg.norm(ground_truth), -1, 1))


def summary_angular_errors(errors):
  errors = sorted(errors)

  def g(f):
    return np.percentile(errors, f * 100)

  median = g(0.5)
  mean = np.mean(errors)
  trimean = 0.25 * (g(0.25) + 2 * g(0.5) + g(0.75))
  results = {
      '25': np.mean(errors[:int(0.25 * len(errors))]),
      '75': np.mean(errors[int(0.75 * len(errors)):]),
      '95': g(0.95),
      'tri': trimean,
      'med': median,
      'mean': mean
  }
  return results


def just_print_angular_errors(results):
  print "25: %5.3f," % results['25'],
  print "med: %5.3f" % results['med'],
  print "tri: %5.3f" % results['tri'],
  print "avg: %5.3f" % results['mean'],
  print "75: %5.3f" % results['75'],
  print "95: %5.3f" % results['95']


def print_angular_errors(errors):
  print "%d images tested. Results:" % len(errors)
  results = summary_angular_errors(errors)
  just_print_angular_errors(results)
  return results


class LowestTrigger:

  def __init__(self):
    self.minimum = None

  def push(self, value):
    if self.minimum is None or value < self.minimum:
      self.minimum = value
      return True
    return False


def rotate_image(image, angle):
  """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

  # Get the image size
  # No that's not an error - NumPy stores image matricies backwards
  image_size = (image.shape[1], image.shape[0])
  image_center = tuple(np.array(image_size) / 2)

  # Convert the OpenCV 3x2 rotation matrix to 3x3
  rot_mat = np.vstack(
      [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])

  rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

  # Shorthand for below calcs
  image_w2 = image_size[0] * 0.5
  image_h2 = image_size[1] * 0.5

  # Obtain the rotated coordinates of the image corners
  rotated_coords = [
      (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
      (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
      (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
      (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
  ]

  # Find the size of the new image
  x_coords = [pt[0] for pt in rotated_coords]
  x_pos = [x for x in x_coords if x > 0]
  x_neg = [x for x in x_coords if x < 0]

  y_coords = [pt[1] for pt in rotated_coords]
  y_pos = [y for y in y_coords if y > 0]
  y_neg = [y for y in y_coords if y < 0]

  right_bound = max(x_pos)
  left_bound = min(x_neg)
  top_bound = max(y_pos)
  bot_bound = min(y_neg)

  new_w = int(abs(right_bound - left_bound))
  new_h = int(abs(top_bound - bot_bound))

  # We require a translation matrix to keep the image centred
  trans_mat = np.matrix([[1, 0, int(new_w * 0.5 - image_w2)],
                         [0, 1, int(new_h * 0.5 - image_h2)], [0, 0, 1]])

  # Compute the tranform for the combined rotation and translation
  affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

  # Apply the transform
  result = cv2.warpAffine(
      image, affine_mat, (new_w, new_h), flags=cv2.INTER_LINEAR)

  return result


def largest_rotated_rect(w, h, angle):
  """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

  quadrant = int(math.floor(angle / (math.pi / 2))) & 3
  sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
  alpha = (sign_alpha % math.pi + math.pi) % math.pi

  bb_w = w * math.cos(alpha) + h * math.sin(alpha)
  bb_h = w * math.sin(alpha) + h * math.cos(alpha)

  gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

  delta = math.pi - alpha - gamma

  length = h if (w < h) else w

  d = length * math.cos(alpha)
  a = d * math.sin(alpha) / math.sin(delta)

  y = a * math.cos(gamma)
  x = y * math.tan(gamma)

  return (bb_w - 2 * x, bb_h - 2 * y)


def crop_around_center(image, width, height):
  """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

  image_size = (image.shape[1], image.shape[0])
  image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

  if (width > image_size[0]):
    width = image_size[0]

  if (height > image_size[1]):
    height = image_size[1]

  x1 = int(image_center[0] - width * 0.5)
  x2 = int(image_center[0] + width * 0.5)
  y1 = int(image_center[1] - height * 0.5)
  y2 = int(image_center[1] + height * 0.5)

  return image[y1:y2, x1:x2]


def rotate_and_crop(image, angle):
  image_width, image_height = image.shape[:2]
  image_rotated = rotate_image(image, angle)
  image_rotated_cropped = crop_around_center(image_rotated,
                                             *largest_rotated_rect(
                                                 image_width, image_height,
                                                 math.radians(angle)))
  return image_rotated_cropped


class Tee(object):

  def __init__(self, name):
    self.file = open(name, 'w')
    self.stdout = sys.stdout
    self.stderr = sys.stderr
    sys.stdout = self
    sys.stderr = self

  def __del__(self):
    self.file.close()

  def write(self, data):
    self.file.write(data)
    self.stdout.write(data)
    self.file.flush()
    self.stdout.flush()

  def write_to_file(self, data):
    self.file.write(data)


def hdr2ldr(raw):
  return (np.clip(np.power(raw / (
      raw.max() * 0.5), 1 / 2.2), 0, 1) * 255).astype(np.uint8)


def bgr2uvl(raw):
  u = np.log(raw[:, :, 2] / raw[:, :, 1])
  v = np.log(raw[:, :, 0] / raw[:, :, 1])
  l = np.log(0.2126 * raw[:, :, 2] + 0.7152 * raw[:, :, 1] +
             0.0722 * raw[:, :, 0])
  l = (l - l.mean()) * 0.3 + 0.5
  u = u * UV_SCALE + 0.5
  v = v * UV_SCALE + 0.5
  uvl = np.stack([u, v, l], axis=2)
  uvl = (np.clip(uvl, 0, 1) * 255).astype(np.uint8)
  return uvl


def bgr2nrgb(raw):
  rgb = raw / np.maximum(1e-4, np.linalg.norm(raw, axis=2, keepdims=True))
  return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def get_WB_image(img, illum):
  return img / illum[::-1]


def slice_list(l, fractions):
  sliced = []
  for i in range(len(fractions)):
    total_fraction = sum(fractions)
    start = int(round(1.0 * len(l) * sum(fractions[:i]) / total_fraction))
    end = int(round(1.0 * len(l) * sum(fractions[:i + 1]) / total_fraction))
    sliced.append(l[start:end])
  return sliced


def get_session():
  import tensorflow as tf
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  return tf.Session(config=config)
