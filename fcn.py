# FC4 is by design an FCN.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Things we tried but not turned out to work: (so you don't have to understand them)
#   FCN.build_shallow, FEED_SHALLOW
#   per_patch_weight, per_patch_loss
#   smooth_l1
#
# Things to clarify:
#   per_pixel_est is actually per-patch estimation, each patch is the receptive field of the FCN.

import cPickle as pickle
import math
import time
import os
import cv2
import sys
from utils import Tee
import shutil
import numpy as np
import tensorflow as tf
from data_provider import load_data, DataProvider
from utils import angular_error, print_angular_errors, LowestTrigger

from config import *
#from alexnet import create_alexnet as create_convnet
from squeeze_net import create_convnet
from summary_utils import *
import random
slim = tf.contrib.slim


# An alternative loss function to original angular loss
# not used any more
def smooth_l1(x):
  return tf.maximum(tf.minimum(0.5 * x**2, 0.5), x - 0.5)


class FCN:

  def __init__(self, sess=None, name=None, kwargs={}):
    global TRAINING_FOLDS, TEST_FOLDS
    self.name = name
    self.wd = GLOBAL_WEIGHT_DECAY
    TRAINING_FOLDS, TEST_FOLDS = initialize_dataset_config(**kwargs)
    self.training_data_provider = None
    self.sess = sess
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer=slim.l2_regularizer(GLOBAL_WEIGHT_DECAY)):
      self.build()
    tf.global_variables_initializer()
    # Store the test-time networks, for images of different resolution
    self.test_nets = {}
    self.saver = tf.train.Saver(max_to_keep=CKPTS_TO_KEEP)

  # There are two branches:
  # 1. AlexNet branch
  # 2. A shallower shortcut branch (*not* affected by USE_SHORTCUT), which proves to be not working

  # images: 0-65535 linear RGB
  @staticmethod
  def build_branches(images, dropout):
    images = tf.clip_by_value(images, 0.0, 65535.0)
    if USE_SHORTCUT:
      # Apply grey-world first
      image_mean = tf.reduce_mean(images, axis=(1, 2), keep_dims=True)
      images = images * (1.0 / tf.maximum(image_mean, 1e-10))
      image_max = tf.reduce_max(images, axis=(1, 2, 3), keep_dims=True)
      images = images * (1.0 / tf.maximum(image_max, 1e-10))
    else:
      images = images * (1.0 / 65535)

    #self.prob = alex_outputs['prob']
    feed_to_fc = []

    # Build the AlexNet branch
    if FEED_ALEX:
      with tf.variable_scope('AlexNet'):
        # RGB (0~1) to sRGB (0~255) with gamma
        # alex_images = (tf.pow(images, 1.0/config_get_input_gamma()) * 255.0)
        # the caffe model takes BGR, 0~255 image
        alex_images = (tf.pow(images, 1.0 / config_get_input_gamma()) *
                       255.0)[:, :, :, ::-1]
        #alex_images = alex_images - np.array([103.939, 116.779, 123.68])[None, None, None,:]
        alex_outputs = create_convnet(alex_images)
      alexnet_output = alex_outputs['features_out']
      feed_to_fc.append(alexnet_output)

    # Build the shallow network branch
    if FEED_SHALLOW:
      with tf.variable_scope('Shallow'):
        shallow_outputs = FCN.build_shallow(images)
      feed_to_fc.append(shallow_outputs)

    feed_to_fc = tf.concat(axis=3, values=feed_to_fc)
    print('Feed to FC shape', feed_to_fc.get_shape())

    # The FC1 here is actually a convolutional layer, since we use FCN
    fc1 = slim.conv2d(
        feed_to_fc, FC1_SIZE, [FC1_KERNEL_SIZE, FC1_KERNEL_SIZE], scope='fc1')
    print('FC1 shape', fc1.get_shape())
    # Use dropout, training time only
    fc1 = slim.dropout(fc1, dropout)

    ##############
    # There are two ways to do confidence-weighted pooling
    #   1. Output (normalized) R, G, B and weight, average RGB by weight
    #   2. Just output unnormalized R, G, B, and simply take the sum and normalize. Thus
    #      we are weighting using the length.
    ##############

    if not SEPERATE_CONFIDENCE:
      # Way 2
      fc2 = slim.conv2d(fc1, 3, [1, 1], scope='fc2', activation_fn=None)
    else:
      # Way 1
      print("Using sperate fc2")
      fc2_mid = slim.conv2d(fc1, 4, [1, 1], scope='fc2', activation_fn=None)
      fc2_mid = tf.nn.relu(fc2_mid)
      rgb = tf.nn.l2_normalize(fc2_mid[:, :, :, :3], 3)
      w, h = map(int, fc2_mid.get_shape()[1:3])
      print(w, h)
      confidence = fc2_mid[:, :, :, 3:4]
      confidence = tf.reshape(confidence, shape=[-1, w * h])
      print(confidence.get_shape())
      confidence = tf.nn.softmax(confidence)
      confidence = tf.reshape(confidence, shape=[-1, w, h, 1])
      fc2 = rgb * confidence

    print('FC2 shape', fc2.get_shape())
    if USE_SHORTCUT:
      fc2 = fc2 * image_mean[:, :, :, ::-1]
    if not WEIGHTED_POOLING:
      # Simply average pooling
      fc2 = tf.nn.l2_normalize(fc2, 3)
    print('FC2 shape', fc2.get_shape())
    if FC_POOLING:
      # WEIGHTED_POOLING should be true, since we should not normalize anything before feeding into FC
      assert WEIGHTED_POOLING
      assert RESIZE_TEST
      # Pooling using a simple FC, as noted in the paper
      fc2 = tf.nn.l2_normalize(
          slim.conv2d(
              tf.nn.relu(fc2),
              3, [15, 15],
              scope='fc_pooling',
              activation_fn=None,
              padding='VALID'), 3)
      print('FC2 shape', fc2.get_shape())
    return fc2

  # Build the network
  def build(self):
    self.dropout = tf.placeholder(tf.float32, shape=(), name='dropout')
    # We don't use per_patch_weight any more.
    self.per_patch_weight = tf.placeholder(
        tf.float32, shape=(), name='pre_patch_weight')
    per_patch_weight = self.per_patch_weight
    self.learning_rate = tf.placeholder(
        tf.float32, shape=(), name='learning_rate')
    # ground truth, actually
    self.illums = tf.placeholder(tf.float32, shape=(None, 3), name='illums')
    # input images
    self.images = tf.placeholder(
        tf.float32,
        shape=(None, FCN_INPUT_SIZE, FCN_INPUT_SIZE, 3),
        name='images')

    with tf.variable_scope('FCN'):
      fc2 = self.build_branches(self.images, self.dropout)

    self.per_pixel_est = fc2

    self.illum_normalized = tf.nn.l2_normalize(
        tf.reduce_sum(fc2, axis=(1, 2)), 1)

    self.train_visualization = get_visualization(
        self.images, self.per_pixel_est, self.illum_normalized, self.illums,
        (VISUALIZATION_SIZE, VISUALIZATION_SIZE))

    self.global_loss = self.get_angular_loss(
        tf.reduce_sum(fc2, axis=(1, 2)), self.illums, LENGTH_REGULARIZATION)
    self.per_patch_loss = self.get_angular_loss(
        fc2, self.illums[:, None, None, :], LENGTH_REGULARIZATION)

    self.loss = (
        1 - per_patch_weight
    ) * self.global_loss + self.per_patch_weight * self.per_patch_loss

    scalar_summaries = []

    scalar_summaries.append(
        tf.summary.scalar('per_patch_loss', self.per_patch_loss))
    scalar_summaries.append(
        tf.summary.scalar('full_image_loss', self.global_loss))
    scalar_summaries.append(tf.summary.scalar('loss', self.loss))

    self.scalar_summaries = tf.summary.merge(scalar_summaries)
    conv_scopes = []
    if FEED_ALEX:
      conv_scopes.append('FCN/AlexNet/conv1')
    if FEED_SHALLOW:
      conv_scopes.append('FCN/Shallow/conv1')

    image_summaries = []
    '''
        for scope in conv_scopes:
            with tf.variable_scope(scope, reuse=True):
                image_summaries.append(conv_summary(tf.get_variable("weights"), scope))
        self.image_summaries = tf.summary.merge(image_summaries)
        '''

    self.merge_summaries = tf.summary.merge_all()

    reg_losses = tf.add_n(slim.losses.get_regularization_losses())
    self.total_loss = self.loss + reg_losses
    self.train_step_adam = tf.train.AdamOptimizer(self.learning_rate).minimize(
        self.total_loss)

    var_list1 = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='FCN/AlexNet')
    var_list2 = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='FCN/fc1') + tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='FCN/fc2')

    for v in var_list1:
      print('list1', v.name)
    for v in var_list2:
      print('list2', v.name)

    opt1 = tf.train.AdamOptimizer(self.learning_rate * FINE_TUNE_LR_RATIO)
    opt2 = tf.train.AdamOptimizer(self.learning_rate)
    grads = tf.gradients(self.total_loss, var_list1 + var_list2)
    grads1 = grads[:len(var_list1)]
    grads2 = grads[len(var_list1):]
    train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
    train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
    self.train_step_sgd = tf.group(train_op1, train_op2)

  # The shallow branch - not used any more
  @staticmethod
  def build_shallow(images):
    net = images
    net = slim.conv2d(
        net,
        SHALLOW_CHANNELS[0], [11, 11],
        stride=4,
        scope='conv1',
        padding='SAME')
    net = slim.max_pool2d(net, [3, 3], padding='VALID')
    net = slim.conv2d(
        net,
        SHALLOW_CHANNELS[1], [5, 5],
        stride=1,
        scope='conv2',
        padding='SAME')
    net = slim.max_pool2d(net, [3, 3], padding='VALID')
    net = slim.max_pool2d(net, [3, 3], padding='VALID')
    return net

  def duplicate_output_to_log(self):
    self.tee = Tee(self.get_ckpt_folder() + 'log.txt')

  def get_summary_variables(self, i, j):
    summary_variables = []
    if i >= 2 and WRITE_SUMMARY and j == 0:
      summary_variables.append(self.scalar_summaries)
      if i % IMAGE_SUMMARY_INT == 0:
        #summary_variables.append(self.image_summaries)
        pass
    return summary_variables

  def get_train_step(self, i):
    if i <= FORCE_ADAM or OPTIMIZER == 'ADAM':
      return self.train_step_adam
    elif OPTIMIZER == 'SGD':
      return self.train_step_sgd
    else:
      assert False

  def train(self, epochs, backup=True):
    trigger = LowestTrigger()
    if not self.try_make_ckpt_folder():
      print('Warning: folder exists!!!')
    if backup:
      self.backup_scripts()
      print('Backup succeeded')
    self.train_writer = tf.summary.FileWriter(
        logdir=self.get_ckpt_folder() + '/training/', graph=self.sess.graph)
    self.validation_writer = tf.summary.FileWriter(
        logdir=self.get_ckpt_folder() + '/validation/')
    self.duplicate_output_to_log()
    training_batch_size = TRAINING_BATCH_SIZE
    print("TF", TRAINING_FOLDS)
    self.training_data_provider = DataProvider(True, TRAINING_FOLDS)
    self.training_data_provider.set_batch_size(training_batch_size)
    test_summary_input = tf.placeholder(tf.float32, shape=())
    test_summary = tf.summary.scalar('loss', test_summary_input)
    batches_per_training_epoch = self.training_data_provider.data_count // training_batch_size

    for i in range(1, epochs + 1):
      # Actually we use a fixed learning rate
      learning_rate = BASE_LEARNING_RATE * pow(LR_DECAY,
                                               1.0 * i / LR_DECAY_INTERVAL)
      epoch_starting_time = time.time()
      training_losses = []
      for j in range(batches_per_training_epoch):
        batch = self.training_data_provider.get_batch()
        summary_variables = self.get_summary_variables(i, j)
        visualization = []
        # visualize some training images for monitoring the process
        if TRAINING_VISUALIZATION and i % TRAINING_VISUALIZATION == 0:
          visualization.append(self.train_visualization)
        loss, _, per_patch_loss, global_loss, summary, ppest, vis = self.sess.run(
            [
                self.loss,
                self.get_train_step(i), self.per_patch_loss, self.global_loss,
                summary_variables, self.per_pixel_est, visualization
            ],
            feed_dict={
                self.images: batch[0],
                self.illums: batch[2],
                self.dropout: DROPOUT,
                self.per_patch_weight: PER_PATCH_WEIGHT,
                self.learning_rate: learning_rate
            })
        for s in summary:
          self.train_writer.add_summary(s, i)
        if vis:
          folder = self.get_ckpt_folder() + '/training_visualization/'
          try:
            os.mkdir(folder)
          except:
            pass
          for k, merged in enumerate(vis[0]):
            summary_fn = '%s/%04d-%03d.jpg' % (folder, i, j * len(vis) + k)
            cv2.imwrite(summary_fn, merged[:, :, ::-1] * 255)

        training_losses.append(loss)

      training_loss = sum(training_losses) / len(training_losses)
      validation_loss = 10

      ending = ''
      info = "*%s* E %4d, TL %.3f, VL %.3f, D %.3f, t %4.1f%s" % (
          self.name, i, training_loss, validation_loss,
          validation_loss - training_loss, time.time() - epoch_starting_time,
          ending)
      print(info)
      saved = False
      if CKPT_PERIOD and i % CKPT_PERIOD == 0:
        self.save(i)
        saved = True
      # Do some test
      if TEST_PERIOD and i % TEST_PERIOD == 0:
        summary = i // TEST_PERIOD % 5 == 0
        if MULTIRES_TEST:
          self.test(summary=summary, scales=[0.25], summary_key=i)
          self.test(summary=summary, scales=[0.5], summary_key=i)
        if not RESIZE_TEST:
          errors = self.test(
              summary=summary, scales=[TEST_BASE_RES], summary_key=i)[0]
        else:
          errors = self.test_resize()[0]
        self.validation_writer.add_summary(
            test_summary.eval(feed_dict={test_summary_input: np.mean(errors)}),
            i)
        if MULTIRES_TEST:
          self.test(summary=summary, scales=[0.25, 0.5, 1.0], summary_key=i)
        # Does validation loss achieve a new minimum?
        # It is unfortunate that the datasets have only train and test separation...
        # We need a validation set to prevent overfitting...
        if trigger.push(np.mean(errors)):
          error_fn = self.get_ckpt_folder() + 'error%04d.pkl' % i
          with open(error_fn, 'wb') as f:
            pickle.dump(errors, f, protocol=-1)
          #self.test_patch_based()
          if not saved:
            self.save(i)
            saved = True
        print('mean(errors) from fcn.py line: 330', np.mean(errors))
    self.training_data_provider.stop()

  # Simply test the network
  def test_network(self, summary=False, summary_key=0):
    records = load_data(['m'])

    for r in records:
      scale = 1
      a = 1
      img = np.clip((r.img / r.img.max()), 0, 1)
      img = np.power(img, 2.2)
      img[:, :img.shape[1] // 2:, 0] *= 3
      #img = np.pad(img, ((112, 112), (112, 112), (0, 0)), 'constant')
      if scale != 1.0:
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
      shape = img.shape[:2]
      if shape not in self.test_nets:
        test_net = {}
        test_net['images'] = tf.placeholder(
            tf.float32, shape=(None, shape[0], shape[1], 3), name='test_images')
        with tf.variable_scope("FCN", reuse=True):
          test_net['pixels'] = FCN.build_branches(test_net['images'], 1.0)
        self.test_nets[shape] = test_net
      test_net = self.test_nets[shape]

      pixels = self.sess.run(
          test_net['pixels'],
          feed_dict={
              test_net['images']: img[None, :, :, :],
          })

      pixels = pixels[0].astype(np.float32)
      #pixels = cv2.resize(pixels, image.shape[0:2][::-1])
      pixels /= np.linalg.norm(pixels, axis=2)[:, :, None]

      #pixels = pixels[:,:,::-1]

      cv2.imshow('pixels', cv2.resize(pixels, (0, 0), fx=10, fy=10))
      cv2.imshow('image', cv2.resize(img, (0, 0), fx=0.5, fy=0.5))
      cv2.waitKey(0)

  # Test each image in multiple resolutions, and then average
  def test_multi(self, summary=False, summary_key=0):
    records = load_data(['m'])

    summaries = []

    errors = []
    for r in records:
      all_pixels = []
      scale = 1.0
      a = 1
      artificial = np.array((1.0, 1.0, 1.0))[None, None, :]
      img = r.img.copy()
      img = np.clip((img * artificial[:, :, ::-1] / img.max()), 0, 1)
      img = np.power(img, 2.2)
      #img[:,:img.shape[1] // 2:,0] *= 3
      #img = np.pad(img, ((112, 112), (112, 112), (0, 0)), 'constant')
      if scale != 1.0:
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
      shape = img.shape[:2]
      if shape not in self.test_nets:
        test_net = {}
        test_net['images'] = tf.placeholder(
            tf.float32, shape=(None, shape[0], shape[1], 3), name='test_images')
        with tf.variable_scope("FCN", reuse=True):
          test_net['pixels'] = FCN.build_branches(test_net['images'], 1.0)
        self.test_nets[shape] = test_net
      test_net = self.test_nets[shape]

      pixels = self.sess.run(
          test_net['pixels'],
          feed_dict={
              test_net['images']: img[None, :, :, :],
          })

      pixels = pixels[0].astype(np.float32) / artificial
      illum = r.illum.astype(np.float32)
      pixels = cv2.resize(pixels, illum.shape[0:2][::-1])
      pixels /= np.linalg.norm(pixels, axis=2)[:, :, None]
      illum /= np.linalg.norm(illum, axis=2)[:, :, None]

      pixels = pixels[:, :, ::-1]

      cv2.imshow('pixels', cv2.resize(pixels, (0, 0), fx=2, fy=2))
      cv2.imshow('illum', cv2.resize(illum, (0, 0), fx=2, fy=2))
      cv2.imshow('image', img)
      cv2.waitKey(0)

      illum = illum.astype(np.float32)
      pixels = pixels.astype(np.float32)

      dot = np.sum(pixels * illum, axis=2)

      error = np.mean(np.arccos(np.clip(dot, -1, 1)) * 180 / math.pi)
      print(error)
      errors.append(error)
    errors = sorted(errors)

    print('Mean:', np.mean(errors), 'Median', errors[len(errors) // 2])

  # Test grey world performance
  @staticmethod
  def test_naive():
    t = time.time()

    import scipy.io
    std = scipy.io.loadmat('/home/yuanming/colorchecker_shi_greyworld.mat')
    names = map(lambda x: x[0].encode('utf8'), std['all_image_names'][0])
    #print(names)
    records = load_data(TEST_FOLDS)

    errors = []
    for r in records:
      est = np.mean(r.img, axis=(0, 1))[::-1]
      est /= np.linalg.norm(est)
      #print(r.fn, est)
      #est=np.array((1, 1, 1))
      #est2= std['estimated_illuminants'][names.index(r.fn[:-4])]
      gt2 = std['groundtruth_illuminants'][names.index(r.fn[:-4])]
      #print(est2)
      error = math.degrees(angular_error(est, gt2))
      errors.append(error)

    print("Full Image:")
    ret = print_angular_errors(errors)
    print('Test time:', time.time() - t, 'per image:',
          (time.time() - t) / len(records))

    return errors

  # Just get the estimated illumination
  def inference(self, fn, datapacks=['g0', 'g1', 'g2']):
    records = load_data(datapacks)
    for r in records:
      if r.fn != fn:
        continue
      scale = 0.5
      img = cv2.resize(r.img, (0, 0), fx=scale, fy=scale)
      shape = img.shape[:2]
      test_net = {}
      test_net['images'] = tf.placeholder(
          tf.float32, shape=(None, shape[0], shape[1], 3), name='test_images')
      with tf.variable_scope("FCN", reuse=True):
        test_net['pixels'] = FCN.build_branches(test_net['images'], 1.0)
        test_net['est'] = tf.reduce_sum(test_net['pixels'], axis=(1, 2))
      self.test_nets[shape] = test_net

      pixels, est = self.sess.run(
          [test_net['pixels'], test_net['est']],
          feed_dict={
              test_net['images']: img[None, :, :, :],
          })
      pixels = pixels[0]
      est = est[0]
      return img, pixels, est, r.illum
    assert False, "Image not found"

  def test(self,
           summary=False,
           scales=[1.0],
           weights=[],
           summary_key=0,
           data=None,
           eval_speed=False, visualize=False):
    if not TEST_FOLDS:
      return [0]
    if data is None:
      records = load_data(TEST_FOLDS)
    else:
      records = data
    avg_errors = []
    median_errors = []
    t = time.time()

    summaries = []
    if weights == []:
      weights = [1.0] * len(scales)

    outputs = []
    ground_truth = []
    avg_confidence = []

    errors = []
    for r in records:
      all_pixels = []
      for scale, weight in zip(scales, weights):
        img = r.img
        if scale != 1.0:
          img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        shape = img.shape[:2]
        if shape not in self.test_nets:
          aspect_ratio = 1.0 * shape[1] / shape[0]
          if aspect_ratio < 1:
            target_shape = (MERGED_IMAGE_SIZE, MERGED_IMAGE_SIZE * aspect_ratio)
          else:
            target_shape = (MERGED_IMAGE_SIZE / aspect_ratio, MERGED_IMAGE_SIZE)
          target_shape = tuple(map(int, target_shape))

          test_net = {}
          test_net['illums'] = tf.placeholder(
              tf.float32, shape=(None, 3), name='test_illums')
          test_net['images'] = tf.placeholder(
              tf.float32,
              shape=(None, shape[0], shape[1], 3),
              name='test_images')
          with tf.variable_scope("FCN", reuse=True):
            test_net['pixels'] = FCN.build_branches(test_net['images'], 1.0)
            test_net['est'] = tf.reduce_sum(test_net['pixels'], axis=(1, 2))
          test_net['merged'] = get_visualization(
              test_net['images'], test_net['pixels'], test_net['est'],
              test_net['illums'], target_shape)
          self.test_nets[shape] = test_net
        test_net = self.test_nets[shape]

        pixels, est, merged = self.sess.run(
            [test_net['pixels'], test_net['est'], test_net['merged']],
            feed_dict={
                test_net['images']: img[None, :, :, :],
                test_net['illums']: r.illum[None, :]
            })

        if eval_speed:
          eval_batch_size = 1
          eval_packed_input = img[None, :, :, :].copy()
          eval_packed_input = np.concatenate(
              [eval_packed_input for i in range(eval_batch_size)], axis=0)
          eval_packed_input = np.ascontiguousarray(eval_packed_input)
          eval_start_t = time.time()
          print(eval_packed_input.shape)
          eval_rounds = 100
          images_variable = tf.Variable(
              tf.random_normal(
                  eval_packed_input.shape, dtype=tf.float32, stddev=1e-1))
          print(images_variable)
          for eval_t in range(eval_rounds):
            print(eval_t)
            pixels, est = self.sess.run(
                [test_net['pixels'], test_net['est']],
                feed_dict={
                    test_net['images']:  #images_variable,
                        eval_packed_input,
                })
          eval_elapsed_t = time.time() - eval_start_t
          print('per image evaluation time',
                eval_elapsed_t / (eval_rounds * eval_batch_size))

        pixels = pixels[0]
        #est = est[0]
        merged = merged[0]

        all_pixels.append(weight * pixels.reshape(-1, 3))

      all_pixels = np.sum(np.concatenate(all_pixels, axis=0), axis=0)
      est = all_pixels / (np.linalg.norm(all_pixels) + 1e-7)
      outputs.append(est)
      ground_truth.append(r.illum)
      error = math.degrees(angular_error(est, r.illum))
      errors.append(error)
      avg_confidence.append(np.mean(np.linalg.norm(all_pixels)))

      summaries.append((r.fn, error, merged))
    print("Full Image:")
    ret = print_angular_errors(errors)
    ppt = (time.time() - t) / len(records)
    print('Test time:', time.time() - t, 'per image:', ppt)

    if summary:
      for fn, error, merged in summaries:
        folder = self.get_ckpt_folder() + '/test%04dsummaries_%4f/' % (
            summary_key, scale)
        try:
          os.mkdir(folder)
        except:
          pass
        summary_fn = '%s/%5.3f-%s.png' % (folder, error, fn)
        cv2.imwrite(summary_fn, merged[:, :, ::-1] * 255)
        
    if visualize:
      for fn, error, merged in summaries:
        cv2.imshow('Testing', merged[:, :, ::-1])
        cv2.waitKey(0)
      
    return errors, ppt, outputs, ground_truth, ret, avg_confidence

  # Test external images, such as sixteen or some jpegs
  # without ground truth
  # images are BGR
  # TODO: move the net work creation here
  def test_external(self, images, scale=1.0, fns=None, show=True, write=True):
    illums = []
    confidence_maps = []
    for img, filename in zip(images, fns):
      if scale != 1.0:
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
      shape = img.shape[:2]
      if shape not in self.test_nets:
        aspect_ratio = 1.0 * shape[1] / shape[0]
        if aspect_ratio < 1:
          target_shape = (MERGED_IMAGE_SIZE, MERGED_IMAGE_SIZE * aspect_ratio)
        else:
          target_shape = (MERGED_IMAGE_SIZE / aspect_ratio, MERGED_IMAGE_SIZE)
        target_shape = tuple(map(int, target_shape))

        test_net = {}
        test_net['illums'] = tf.placeholder(
            tf.float32, shape=(None, 3), name='test_illums')
        test_net['images'] = tf.placeholder(
            tf.float32, shape=(None, shape[0], shape[1], 3), name='test_images')
        with tf.variable_scope("FCN", reuse=True):
          test_net['pixels'] = FCN.build_branches(test_net['images'], 1.0)
          test_net['est'] = tf.reduce_sum(test_net['pixels'], axis=(1, 2))
        test_net['merged'] = get_visualization(
            test_net['images'], test_net['pixels'], test_net['est'],
            test_net['illums'], target_shape)
        self.test_nets[shape] = test_net

      test_net = self.test_nets[shape]

      pixels, est, merged = self.sess.run(
          [test_net['pixels'], test_net['est'], test_net['merged']],
          feed_dict={
              test_net['images']: img[None, :, :, :],
              test_net['illums']: [[1, 1, 1]]
          })
      est = est[0]
      est /= np.linalg.norm(est)

      pixels = pixels[0]
      confidences = np.linalg.norm(pixels, axis=2)
      confidence_maps.append(confidences)
      ind = int(confidences.flatten().shape[0] * 0.95)
      print(confidences.mean(), confidences.max(),
            sorted(confidences.flatten())[ind])
      merged = merged[0]
      illums.append(est)

      if show:
        cv2.imshow('Ret', merged[:, :, ::-1])
        k = cv2.waitKey(0) % (2**20)
      #if k == ord('s'):
      if write:
        cv2.imwrite('/data/common/outputs/%s' % filename,
                    merged[:, :, ::-1] * 255.0)
    return illums, confidence_maps

  # Test how the network performs, if we just resize the image to fix the network input.
  def test_resize(self):
    records = load_data(TEST_FOLDS)
    t = time.time()

    errors = []
    for r in records:
      img = cv2.resize(r.img, (FCN_INPUT_SIZE, FCN_INPUT_SIZE))
      illum_est = self.sess.run(
          self.illum_normalized,
          feed_dict={self.images: [img],
                     self.dropout: 1.0})
      avg_error = math.degrees(angular_error(illum_est, r.illum))
      errors.append(avg_error)
    print_angular_errors(errors)
    ppt = (time.time() - t) / len(records)
    print('Test time:', time.time() - t, 'per image:', ppt)
    return errors, ppt

  # Patch-based test
  def test_patch_based(self, scale, patches, pooling='median'):
    records = load_data(TEST_FOLDS)
    avg_errors = []
    median_errors = []
    t = time.time()

    def sample_patch(img):
      s = FCN_INPUT_SIZE
      x = random.randrange(0, img.shape[0] - s + 1)
      y = random.randrange(0, img.shape[1] - s + 1)
      return img[x:x + s, y:y + s]

    for r in records:
      img = cv2.resize(r.img, (0, 0), fx=scale, fy=scale)
      img = [sample_patch(img) for i in range(patches)]
      illum_est = []
      batch_size = 4
      for j in range((len(img) + batch_size - 1) // batch_size):
        illum_est.append(
            self.sess.run(
                self.illum_normalized,
                feed_dict={
                    self.images: img[j * batch_size:(j + 1) * batch_size],
                    self.dropout: 1.0
                }))
      illum_est = np.vstack(illum_est)
      med = len(illum_est) // 2
      illum_est_median = np.array(
          [sorted(list(illum_est[:, i]))[med] for i in range(3)])
      illum_est_avg = np.mean(illum_est, axis=0)
      avg_error = math.degrees(angular_error(illum_est_avg, r.illum))
      median_error = math.degrees(angular_error(illum_est_median, r.illum))
      avg_errors.append(avg_error)
      median_errors.append(median_error)
    print("Avg pooling:")
    print_angular_errors(avg_errors)
    print("Median pooling:")
    print_angular_errors(median_errors)
    ppt = (time.time() - t) / len(records)
    print('Test time:', time.time() - t, 'per image:', ppt)
    if pooling == 'median':
      errors = median_errors
    else:
      errors = avg_errors
    return errors, ppt

  def get_ckpt_folder(self, name=None):
    if name is None:
      name = self.name
    return MODEL_PATH + '/' + name + '/'

  def backup_scripts(self):
    for fn in os.listdir('.'):
      if fn.endswith('.py'):
        shutil.copy(fn, self.get_ckpt_folder())
    with open(self.get_ckpt_folder() + 'config2.txt', 'w') as f:
      f.write(str(OVERRODE) + '\n')

  def try_make_ckpt_folder(self):
    try:
      os.mkdir(self.get_ckpt_folder())
    except:
      '''
            if raw_input('Warning: folder exists. Overwrite?\n') == 'yes':
                shutil.rmtree(self.get_ckpt_folder())
                os.mkdir(self.get_ckpt_folder())
            else:
                exit(-1)
            '''
      return False
    return True

  def get_filename(self, key, name=None):
    if name is None:
      name = self.name
    return self.get_ckpt_folder(name) + str(key) + '.ckpt'

  def save(self, key):
    save_path = self.saver.save(self.sess, self.get_filename(key))
    print("Model saved in file: %s" % save_path)

  def load(self, key, name=None):
    fn = self.get_filename(key, name=name)
    self.load_absolute(fn)
    
  def load_absolute(self, fn):
    self.saver.restore(self.sess, fn)
    print("Model %s restored." % fn)

  def get_angular_loss(self, vec1, vec2, length_regularization=0.0):
    with tf.name_scope('angular_error'):
      safe_v = 0.999999
      if len(vec1.get_shape()) == 2:
        illum_normalized = tf.nn.l2_normalize(vec1, 1)
        _illum_normalized = tf.nn.l2_normalize(vec2, 1)
        dot = tf.reduce_sum(illum_normalized * _illum_normalized, 1)
        dot = tf.clip_by_value(dot, -safe_v, safe_v)
        length_loss = tf.reduce_mean(
            tf.maximum(tf.log(tf.reduce_sum(vec1**2, axis=1) + 1e-7), 0))
      else:
        assert len(vec1.get_shape()) == 4
        illum_normalized = tf.nn.l2_normalize(vec1, 3)
        _illum_normalized = tf.nn.l2_normalize(vec2, 3)
        dot = tf.reduce_sum(illum_normalized * _illum_normalized, 3)
        dot = tf.clip_by_value(dot, -safe_v, safe_v)
        length_loss = tf.reduce_mean(
            tf.maximum(tf.log(tf.reduce_sum(vec1**2, axis=3) + 1e-7), 0))
      angle = tf.acos(dot) * (180 / math.pi)
      if SMOOTH_L1:
        angle = smooth_l1(angle)

      if ANGULAR_LOSS:
        return tf.reduce_mean(angle) + length_loss * length_regularization
      else:
        dot = tf.reduce_sum(
            (illum_normalized - _illum_normalized)**2,
            axis=len(illum_normalized.get_shape()) - 1)
        return tf.reduce_mean(dot) * 1000 + length_loss * length_regularization
