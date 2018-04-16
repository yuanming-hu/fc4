import cv2
import cPickle as pickle
import scipy.io
import numpy as np
import os
import sys
import random
from utils import slice_list

SHOW_IMAGES = False
FOLDS = 3

DATA_FRAGMENT = -1
BOARD_FILL_COLOR = 1e-5


def get_image_pack_fn(key):
  ds = key[0]
  if ds == 'g':
    fold = int(key[1])
    return GehlerDataSet().get_image_pack_fn(fold)
  elif ds == 'c':
    camera = int(key[1])
    fold = int(key[2])
    return ChengDataSet(camera).get_image_pack_fn(fold)
  elif ds == 'm':
    assert False


class ImageRecord:

  def __init__(self, dataset, fn, illum, mcc_coord, img, extras=None):
    self.dataset = dataset
    self.fn = fn
    self.illum = illum
    self.mcc_coord = mcc_coord
    # BRG images
    self.img = img
    self.extras = extras

  def __repr__(self):
    return '[%s, %s, (%f, %f, %f)]' % (self.dataset, self.fn, self.illum[0],
                                       self.illum[1], self.illum[2])


class DataSet:

  def get_subset_name(self):
    return ''

  def get_directory(self):
    return 'data/' + self.get_name() + '/'

  def get_img_directory(self):
    return 'data/' + self.get_name() + '/'

  def get_meta_data_fn(self):
    return self.get_directory() + self.get_subset_name() + 'meta.pkl'

  def dump_meta_data(self, meta_data):
    print 'Dumping data =>', self.get_meta_data_fn()
    print '  Total records:', sum(map(len, meta_data))
    print '  Slices:', map(len, meta_data)
    with open(self.get_meta_data_fn(), 'wb') as f:
      pickle.dump(meta_data, f, protocol=-1)
    print 'Dumped.'

  def load_meta_data(self):
    with open(self.get_meta_data_fn()) as f:
      return pickle.load(f)

  def get_image_pack_fn(self, fold):
    return self.get_directory() + self.get_subset_name(
    ) + 'image_pack.%d.pkl' % fold

  def dump_image_pack(self, image_pack, fold):
    with open(self.get_image_pack_fn(fold), 'wb') as f:
      pickle.dump(image_pack, f, protocol=-1)

  def load_image_pack(self, fold):
    with open(self.get_meta_data_fn()) as f:
      return pickle.load(f)

  def regenerate_image_pack(self, meta_data, fold):
    image_pack = []
    for i, r in enumerate(meta_data):
      print 'Processing %d/%d\r' % (i + 1, len(meta_data)),
      sys.stdout.flush()
      r.img = self.load_image_without_mcc(r)

      if SHOW_IMAGES:
        cv2.imshow('img',
                   cv2.resize(
                       np.power(r.img / 65535., 1.0 / 3.2), (0, 0),
                       fx=0.25,
                       fy=0.25))
        il = r.illum
        if len(il.shape) >= 3:
          cv2.imshow('Illum', il)
        cv2.waitKey(0)

      image_pack.append(r)
    print
    self.dump_image_pack(image_pack, fold)

  def regenerate_image_packs(self):
    meta_data = self.load_meta_data()
    print 'Dumping image packs...'
    print '%s folds found' % len(meta_data)
    for f, m in enumerate(meta_data):
      self.regenerate_image_pack(m, f)

  def get_folds(self):
    return FOLDS


class GehlerDataSet(DataSet):

  def get_name(self):
    return 'gehler'

  def regenerate_meta_data(self):
    meta_data = []
    print "Loading and shuffle fn_and_illum[]"
    ground_truth = scipy.io.loadmat(self.get_directory() + 'ground_truth.mat')[
        'real_rgb']
    ground_truth /= np.linalg.norm(ground_truth, axis=1)[..., np.newaxis]
    filenames = sorted(os.listdir(self.get_directory() + 'images'))
    folds = scipy.io.loadmat(self.get_directory() + 'folds.mat')
    filenames2 = map(lambda x: str(x[0][0][0]), folds['Xfiles'])
    #print filenames
    #print filenames2
    for i in range(len(filenames)):
      assert filenames[i][:-4] == filenames2[i][:-4]
    for i in range(len(filenames)):
      fn = filenames[i]
      mcc_coord = self.get_mcc_coord(fn)
      meta_data.append(
          ImageRecord(
              dataset=self.get_name(),
              fn=fn,
              illum=ground_truth[i],
              mcc_coord=mcc_coord,
              img=None))

    if DATA_FRAGMENT != -1:
      meta_data = meta_data[:DATA_FRAGMENT]
      print 'Warning: using only first %d images...' % len(meta_data)

    meta_data_folds = [[], [], []]
    for i in range(FOLDS):
      fold = list(folds['te_split'][0][i][0])
      print len(fold)
      for j in fold:
        meta_data_folds[i].append(meta_data[j - 1])
    for i in range(3):
      print 'Fold', i
      print map(lambda m: m.fn, meta_data_folds[i])
    print sum(map(len, meta_data_folds))
    assert sum(map(len, meta_data_folds)) == len(filenames)
    for i in range(3):
      assert set(meta_data_folds[i]) & set(meta_data_folds[(i + 1) % 3]) == set(
      )
    self.dump_meta_data(meta_data_folds)

  def get_mcc_coord(self, fn):
    # Note: relative coord
    with open(self.get_directory() + 'coordinates/' + fn.split('.')[0] +
              '_macbeth.txt', 'r') as f:
      lines = f.readlines()
      width, height = map(float, lines[0].split())
      scale_x = 1 / width
      scale_y = 1 / height
      lines = [lines[1], lines[2], lines[4], lines[3]]
      polygon = []
      for line in lines:
        line = line.strip().split()
        x, y = (scale_x * float(line[0])), (scale_y * float(line[1]))
        polygon.append((x, y))
      return np.array(polygon, dtype='float32')

  def load_image(self, fn):
    file_path = self.get_img_directory() + '/images/' + fn
    raw = np.array(cv2.imread(file_path, -1), dtype='float32')
    if fn.startswith('IMG'):
      # 5D3 images
      black_point = 129
    else:
      black_point = 1
    raw = np.maximum(raw - black_point, [0, 0, 0])
    return raw

  def load_image_without_mcc(self, r):
    raw = self.load_image(r.fn)
    img = (np.clip(raw / raw.max(), 0, 1) * 65535.0).astype(np.uint16)
    polygon = r.mcc_coord * np.array([img.shape[1], img.shape[0]])
    polygon = polygon.astype(np.int32)
    cv2.fillPoly(img, [polygon], (BOARD_FILL_COLOR,) * 3)
    return img


class ChengDataSet(DataSet):

  def __init__(self, camera_id):
    camera_names = [
        'Canon1DsMkIII', 'Canon600D', 'FujifilmXM1', 'NikonD5200',
        'OlympusEPL6', 'PanasonicGX1', 'SamsungNX2000', 'SonyA57'
    ]
    self.camera_name = camera_names[camera_id]

  def get_subset_name(self):
    return self.camera_name + '-'

  def get_name(self):
    return 'cheng'

  def regenerate_meta_data(self):
    meta_data = []
    ground_truth = scipy.io.loadmat(self.get_directory() + 'ground_truth/' +
                                    self.camera_name + '_gt.mat')
    illums = ground_truth['groundtruth_illuminants']
    darkness_level = ground_truth['darkness_level']
    saturation_level = ground_truth['saturation_level']
    cc_coords = ground_truth['CC_coords']
    illums /= np.linalg.norm(illums, axis=1)[..., np.newaxis]
    filenames = sorted(os.listdir(self.get_directory() + 'images'))
    filenames = filter(lambda f: f.startswith(self.camera_name), filenames)
    extras = {
        'darkness_level': darkness_level,
        'saturation_level': saturation_level
    }
    for i in range(len(filenames)):
      fn = filenames[i]
      y1, y2, x1, x2 = cc_coords[i]
      mcc_coord = np.array([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
      meta_data.append(
          ImageRecord(
              dataset=self.get_name(),
              fn=fn,
              illum=illums[i],
              mcc_coord=mcc_coord,
              img=None,
              extras=extras))

    random.shuffle(meta_data)

    if DATA_FRAGMENT != -1:
      meta_data = meta_data[:DATA_FRAGMENT]
      print 'Warning: using only first %d images...' % len(meta_data)

    meta_data = slice_list(meta_data, [1] * self.get_folds())
    self.dump_meta_data(meta_data)

  def load_image(self, fn, darkness_level, saturation_level):
    file_path = self.get_directory() + '/images/' + fn
    raw = np.array(cv2.imread(file_path, -1), dtype='float32')
    raw = np.maximum(raw - darkness_level, [0, 0, 0])
    raw *= 1.0 / saturation_level
    return raw

  def load_image_without_mcc(self, r):
    img = (np.clip(
        self.load_image(r.fn, r.extras['darkness_level'], r.extras[
            'saturation_level']), 0, 1) * 65535.0).astype(np.uint16)
    #polygon = r.mcc_coord * np.array([img.shape[1], img.shape[0]])
    polygon = r.mcc_coord
    polygon = polygon.astype(np.int32)
    cv2.fillPoly(img, [polygon], (BOARD_FILL_COLOR,) * 3)
    return img


if __name__ == '__main__':
  ds = GehlerDataSet()
  ds.regenerate_meta_data()
  ds.regenerate_image_packs()
