import cPickle as pickle
import sys
import cv2
import os

import numpy as np
import tensorflow as tf

from fcn import FCN
from config import *
from utils import get_session
from data_provider import load_data
import utils
from datasets import get_image_pack_fn
from data_provider import ImageRecord


def get_average(image_packs):
  data = load_data(image_packs.split(','))
  avg = np.zeros(shape=(3,), dtype=np.float32)
  for record in data:
    cv2.imshow('img',
               cv2.resize((record.img / 2.0**16)**0.5, (0, 0), fx=0.2, fy=0.2))
    cv2.waitKey(0)
    avg += np.mean(record.img.astype(np.float32), axis=(0, 1))
  avg = avg / np.linalg.norm(avg)
  print '(%.3f, %.3f, %.3f)' % (avg[0], avg[1], avg[2])


def test(name, ckpt, image_pack_name=None, output_filename=None):
  if image_pack_name is None:
    data = None
  else:
    data = load_data(image_pack_name.split(','))
  with get_session() as sess:
    fcn = FCN(sess=sess, name=name)
    fcn.load(ckpt)
    errors, _, _, _, ret, conf = fcn.test(
        scales=[0.5],
        summary=True,
        summary_key=123,
        data=data,
        eval_speed=False,
        visualize=True)
    if output_filename is not None:
      with open('outputs/%s.pkl' % output_filename, 'wb') as f:
        pickle.dump(ret, f)
      with open('outputs/%s_err.pkl' % output_filename, 'wb') as f:
        pickle.dump(errors, f)
      with open('outputs/%s_conf.pkl' % output_filename, 'wb') as f:
        pickle.dump(conf, f)
      print ret
      print 'results dumped'


def test_input_gamma(name,
                     ckpt,
                     input_gamma,
                     image_pack_name=None,
                     output_filename=None):
  config_set_input_gamma(float(input_gamma))
  if image_pack_name is None:
    data = None
  else:
    data = load_data(image_pack_name.split(','))
  with get_session() as sess:
    fcn = FCN(sess=sess, name=name)
    fcn.load(ckpt)
    _, _, _, _, ret = fcn.test(
        scales=[0.5], summary=True, summary_key=123, data=data)
    if output_filename is not None:
      with open('outputs/%s.pkl' % output_filename, 'wb') as f:
        pickle.dump(ret, f)
      print ret
      print 'results dumped'


def dump_result(name, ckpt, image_pack_name=None):
  if image_pack_name is None:
    data = None
  else:
    data = load_data(image_pack_name.split(','))
  outputs = []
  gts = []
  with get_session() as sess:
    fcn = FCN(sess=sess, name=name)
    fcn.load(ckpt)
    _, _, outputs, gts = fcn.test(
        scales=[0.5], summary=True, summary_key=123, data=data)
  result = {
      'outputs': np.array(outputs),
      'gts': np.array(gts),
  }
  pickle.dump(result,
              open("outputs/%s-%s-%s.pkl" % (name, ckpt, image_pack_name),
                   "wb"))


def dump_errors(name,
                ckpt,
                fold,
                output_filename,
                method='full',
                samples=0,
                pooling='median'):
  samples = int(samples)
  with get_session() as sess:
    kwargs = {'dataset_name': 'gehler', 'subset': 0, 'fold': fold}
    fcn = FCN(sess=sess, name=name, kwargs=kwargs)
    fcn.load(ckpt)
    for i in range(4):
      if method == 'full':
        errors, t, _, _, _ = fcn.test(scales=[0.5])
      elif method == 'resize':
        errors, t = fcn.test_resize()
      elif method == 'patches':
        errors, t = fcn.test_patch_based(
            scale=0.5, patches=samples, pooling=pooling)
      else:
        assert False
  utils.print_angular_errors(errors)
  with open(output_filename, 'w') as f:
    pickle.dump({'e': errors, 't': t}, f)


def test_multi(name, ckpt):
  with get_session() as sess:
    fcn = FCN(sess=sess, name=name)
    fcn.load(ckpt)
    fcn.test_multi()


def test_network(name, ckpt):
  with get_session() as sess:
    fcn = FCN(sess=sess, name=name)
    fcn.load(ckpt)
    fcn.test_network()


def cont(name, preload, key):
  with get_session() as sess:
    fcn = FCN(sess=sess, name=name)
    sess.run(tf.global_variables_initializer())
    if preload is not None:
      fcn.load(name=preload, key=key)
    fcn.train(EPOCHS)


def train(name, *args):
  kwargs = {}
  for arg in args:
    key, val = arg.split('=')
    kwargs[key] = val
    OVERRODE[key] = val
  with get_session() as sess:
    fcn = FCN(sess=sess, name=name, kwargs=kwargs)
    sess.run(tf.global_variables_initializer())
    fcn.train(EPOCHS)


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


def dump_gehler():
  from datasets import GehlerDataSet
  ds = GehlerDataSet()
  ds.regenerate_meta_data()
  ds.regenerate_image_packs()


def dump_cheng(start, end):
  start = int(start)
  end = int(end)
  from datasets import ChengDataSet
  for i in range(start, end + 1):
    ds = ChengDataSet(i)
    ds.regenerate_meta_data()
    ds.regenerate_image_packs()


def override_global(key, val):
  assert False
  if type(globals()[key]) == str:
    globals()[key] = val
  elif type(globals()[key]) == int:
    globals()[key] = int(val)
  elif type(globals()[key]) == float:
    globals()[key] = float(val)
  else:
    assert False
  print "Overriding ", key, '=', val
  OVERRODE[key] = val
  print globals()[key]
  initialize_dataset_config()


def test_naive():
  return FCN.test_naive()


def dump_multi():
  from datasets import MultiDataSet
  ds = MultiDataSet()
  ds.regenerate_meta_data()
  ds.regenerate_image_packs()


if __name__ == '__main__':
  if len(sys.argv) < 2:
    print 'Usage: ./fccc.py [func]'
    exit(-1)
  filename = __file__[2:]
  mode = sys.argv[1]
  globals()[mode](*sys.argv[2:])
