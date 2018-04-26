import tensorflow as tf
import numpy as np
import cv2
import cPickle as pkl

POOLING_LAYERS = [1, 3, 5]
MODEL_PATH = 'data/squeeze_net/model.pkl'

class SqueezeNet(object):

  def __init__(self, imgs):
    self.imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    self.imgs = imgs
    self.weights = {}
    self.net = {}
    self.build_model()

  # take  BGR 0~255 image, i.e. like the ones loaded by open CV
  def build_model(self):
    net = {}
    self.net = net
    self.model = pkl.load(open(MODEL_PATH, 'r'))
    for k in self.model.keys():
      print k, self.model[k].shape
    # Caffe order is BGR, this model is RGB.
    # The mean values are from caffe protofile from DeepScale/SqueezeNet github repo.
    # self.mean = tf.constant([123.0, 117.0, 104.0],
    #                         dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
    self.mean = tf.constant(
        [104.0, 117.0, 123.0],
        dtype=tf.float32,
        shape=[1, 1, 1, 3],
        name='img_mean')
    images = self.imgs - self.mean
    # images = self.imgs-np.array([123.0, 117.0, 104.0]).reshape([1,1,1,3])
    # images = self.imgs-self.imgs

    # images = tf.transpose(images, [0,2,1,3])

    net['input'] = images
    # conv1_1
    net['conv1'] = self.conv_layer(
        'conv1',
        net['input'],
        W=self.weight_variable(
            [3, 3, 3, 64],
            name='conv1_w',
            init=np.transpose(self.model['conv1_weights'], [2, 3, 1, 0])),
        stride=[1, 2, 2, 1],
        padding='VALID') + self.model['conv1_bias'][None, None, None, :]

    net['relu1'] = self.relu_layer(
        'relu1', net['conv1'], b=self.bias_variable([64], 'relu1_b', value=0.0))
    net['pool1'] = self.pool_layer('pool1', net['relu1'])

    net['fire2'] = self.fire_module('fire2', net['pool1'], 16, 64, 64)
    net['fire3'] = self.fire_module('fire3', net['fire2'], 16, 64, 64)
    net['pool3'] = self.pool_layer('pool3', net['fire3'], padding='SAME')

    net['fire4'] = self.fire_module('fire4', net['pool3'], 32, 128, 128)
    net['fire5'] = self.fire_module('fire5', net['fire4'], 32, 128, 128)
    net['pool5'] = self.pool_layer('pool5', net['fire5'], padding='SAME')

    net['fire6'] = self.fire_module('fire6', net['pool5'], 48, 192, 192)
    net['fire7'] = self.fire_module('fire7', net['fire6'], 48, 192, 192)
    net['fire8'] = self.fire_module('fire8', net['fire7'], 64, 256, 256)
    net['pool8'] = self.pool_layer('pool8', net['fire8'])
    net['fire9'] = self.fire_module('fire9', net['fire8'], 64, 256, 256)
    print net['fire9'].shape

    # 50% dropout removed
    #net['dropout9'] = tf.nn.dropout(net['fire9'], self.dropout)
    net['conv10'] = self.conv_layer(
        'conv10',
        net['fire9'],
        W=self.weight_variable(
            [1, 1, 512, 1000],
            name='conv10',
            init=np.transpose(self.model['conv10_weights'], [2, 3, 1, 0])),
        padding='VALID') + self.model['conv10_bias'][None, None, None, :]
    print net['conv10'].shape
    net['relu10'] = self.relu_layer(
        'relu10',
        net['conv10'],
        b=self.bias_variable([1000], 'relu10_b', value=0.0))
    print net['relu10'].shape
    net['pool10'] = self.pool_layer('pool10', net['relu10'], pooling_type='avg')
    print net['pool10'].shape
    avg_pool_shape = tf.shape(net['pool10'])

    net['pool_reshaped'] = tf.reshape(net['pool10'], [avg_pool_shape[0], -1])
    self.fc2 = net['pool_reshaped']
    self.logits = net['pool_reshaped']

    self.probs = tf.nn.softmax(self.logits)
    self.net = net

  def bias_variable(self, shape, name, value=0.1, from_caffe=False):
    if not from_caffe:
      self.weights[name] = tf.get_variable(
          'bias_' + name,
          initializer=tf.constant_initializer(value),
          shape=shape)
    else:
      self.weights[name] = tf.get_variable(
          'bias_' + name,
          initializer=tf.constant_initializer(value=value),
          shape=shape)
    return self.weights[name]

  def weight_variable(self, shape, name=None, init='xavier'):
    if init == 'variance':
      assert False
      initial = tf.get_variable(
          'W' + name,
          shape,
          initializer=tf.contrib.layers.variance_scaling_initializer())
    elif init == 'xavier':
      assert False
      initial = tf.get_variable(
          'W' + name, shape, initializer=tf.contrib.layers.xavier_initializer())
    else:
      assert isinstance(init, np.ndarray)
      print name, init.shape
      initial = tf.get_variable(
          'W' + name, shape, initializer=tf.constant_initializer(value=init))

    self.weights[name] = initial
    return self.weights[name]

  def relu_layer(self, layer_name, layer_input, b=None):
    if b:
      layer_input += b
    relu = tf.nn.relu(layer_input)
    return relu

  def pool_layer(self,
                 layer_name,
                 layer_input,
                 pooling_type='max',
                 padding='VALID'):
    if pooling_type == 'avg':
      pool = tf.nn.avg_pool(
          layer_input,
          ksize=[1, 14, 14, 1],
          strides=[1, 1, 1, 1],
          padding=padding)
    elif pooling_type == 'max':
      pool = tf.nn.max_pool(
          layer_input,
          ksize=[1, 3, 3, 1],
          strides=[1, 2, 2, 1],
          padding=padding)
    return pool

  def conv_layer(self,
                 layer_name,
                 layer_input,
                 W,
                 stride=[1, 1, 1, 1],
                 padding='VALID'):
    return tf.nn.conv2d(layer_input, W, strides=stride, padding=padding)

  def fire_module(self,
                  layer_name,
                  layer_input,
                  s1x1,
                  e1x1,
                  e3x3,
                  residual=False):
    """ Fire module consists of squeeze and expand convolutional layers. """
    fire = {}

    shape = layer_input.get_shape()

    # squeeze np.transpose(self.model['conv1_weights'], [2,3,1,0])),
    s1_weight = self.weight_variable(
        [1, 1, int(shape[3]), s1x1], layer_name + '_s1_weight',
        np.transpose(
            self.model[layer_name + '/' + 'squeeze1x1_weights'],
            axes=[2, 3, 1, 0]))

    # expand
    e1_weight = self.weight_variable(
        [1, 1, s1x1, e1x1], layer_name + '_e1',
        np.transpose(
            self.model[layer_name + '/' + 'expand1x1_weights'],
            axes=[2, 3, 1, 0]))
    e3_weight = self.weight_variable(
        [3, 3, s1x1, e3x3], layer_name + '_e3',
        np.transpose(
            self.model[layer_name + '/' + 'expand3x3_weights'],
            axes=[2, 3, 1, 0]))

    fire['s1'] = self.conv_layer(
        layer_name + '_s1', layer_input, W=s1_weight, padding='SAME')
    fire['relu1'] = self.relu_layer(
        layer_name + '_relu1',
        fire['s1'],
        b=self.bias_variable([s1x1], layer_name + '_fire_bias_s1'))

    fire['e1'] = self.conv_layer(
        layer_name + '_e1', fire['relu1'], W=e1_weight,
        padding='SAME')  # 'SAME' and 'VALID' padding should be the same here
    fire['e3'] = self.conv_layer(
        layer_name + '_e3', fire['relu1'], W=e3_weight, padding='SAME')
    fire['concat'] = tf.concat([
        tf.add(fire['e1'],
               self.bias_variable(
                   [e1x1],
                   name=layer_name + '_fire_bias_e1',
                   value=self.model[layer_name + '/' + 'expand1x1_bias'])),
        tf.add(fire['e3'],
               self.bias_variable(
                   [e3x3],
                   name=layer_name + '_fire_bias_e3',
                   value=self.model[layer_name + '/' + 'expand3x3_bias']))
    ], 3)

    if residual:
      fire['relu2'] = self.relu_layer(layer_name + 'relu2_res',
                                      tf.add(fire['concat'], layer_input))
    else:
      fire['relu2'] = self.relu_layer(layer_name + '_relu2', fire['concat'])
    self.net[layer_name + '_debug'] = fire['relu2']
    return fire['relu2']

  def get_features_out(self):
    return self.net['pool8']


def create_convnet(imgs):
  sn = SqueezeNet(imgs)
  return {'features_out': sn.get_features_out()}

