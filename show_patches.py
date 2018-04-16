import cPickle as pickle
import sys
import cv2
import os
from data_provider import ImageRecord

import numpy as np

def show_patches():
  from data_provider import DataProvider
  dp = DataProvider(True, ['g0'])
  dp.set_batch_size(10)
  while True:
    batch = dp.get_batch()
    for img in batch[0]:
      #img = img / np.mean(img, axis=(0, 1))[None, None, :]
      img = img / img.max()
      cv2.imshow("Input", np.power(img, 1 / 2.2))
      cv2.waitKey(0)


if __name__ == '__main__':
  show_patches()
