import tensorflow as tf
import numpy as np
import cv2
from config import *


def _activation_summary(x):
  """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = x.op.name
  if tensor_name in _activation_summary.summarized:
    return
  _activation_summary.summarized.append(tensor_name)
  # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


_activation_summary.summarized = []


def put_kernels_on_grid(kernel, (grid_Y, grid_X), pad=1):
  '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.
    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)
    
    Return:
      Tensor of shape [(Y+pad)*grid_Y, (X+pad)*grid_X, NumChannels, 1].
    '''
  # pad X and Y
  k_min = tf.reduce_min(kernel, axis=(0, 1, 2), keep_dims=True)
  k_max = tf.reduce_max(kernel, axis=(0, 1, 2), keep_dims=True)

  kernel = (kernel - k_min) / (k_max - k_min)
  x1 = tf.pad(kernel, tf.constant([[pad, 0], [pad, 0], [0, 0], [0, 0]]))

  # X and Y dimensions, w.r.t. padding
  Y = kernel.get_shape()[0] + pad
  X = kernel.get_shape()[1] + pad

  # put NumKernels to the 1st dimension
  x2 = tf.transpose(x1, (3, 0, 1, 2))
  # organize grid on Y axis
  x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, 3]))

  # switch X and Y axes
  x4 = tf.transpose(x3, (0, 2, 1, 3))
  # organize grid on X axis
  x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, 3]))

  # back to normal order (not combining with the next step for clarity)
  x6 = tf.transpose(x5, (2, 1, 3, 0))

  # to tf.image_summary order [batch_size, height, width, channels],
  #   where in this case batch_size == 1
  x7 = tf.transpose(x6, (3, 0, 1, 2))

  # scale to [0, 1]
  x_min = tf.reduce_min(x7)
  x_max = tf.reduce_max(x7)
  x8 = (x7 - x_min) / (x_max - x_min)

  return x8


def _get_grid(weights):
  grid_x = 8
  grid_y = int(weights.get_shape()[3]) // grid_x
  return put_kernels_on_grid(weights, (grid_y, grid_x))


def conv_summary(weights, name):
  grid = _get_grid(weights)
  return tf.summary.image(name, grid)
  #tf.image_summary(name + 'random', tf.random_uniform(shape=grid.get_shape()), max_images=3)


# Output: RGB images
def get_visualization(images, illums_est, illums_pooled, illums_ground,
                      target_shape):
  confidence = tf.sqrt(tf.reduce_sum(illums_est**2, axis=3))

  vis_confidence = confidence[:, :, :,
                              None]  # / tf.reduce_max(confidence, axis=(1, 2), keep_dims=True)[:,:,:,None]

  color_thres = [tf.constant(250.0 * i) for i in range(1, 5)]
  mean_confidence_value = tf.reduce_mean(confidence, axis=(0, 1, 2))
  vis_confidence_colored = tf.cond( mean_confidence_value < color_thres[0],
      lambda: vis_confidence * np.array((0,0,1)).reshape(1, 1, 1, 3)/500.0,
      lambda: tf.cond( mean_confidence_value < color_thres[1],
          lambda: vis_confidence * np.array((0,1,1)).reshape(1, 1, 1, 3)/1000.0,
          lambda: tf.cond( mean_confidence_value < color_thres[2],
              lambda: vis_confidence * np.array((0,1,0)).reshape(1, 1, 1, 3)/2000.0,
              lambda: tf.cond( mean_confidence_value < color_thres[3],
                  lambda: vis_confidence * np.array((1,1,0)).reshape(1, 1, 1, 3)/3000.0,
                  lambda: vis_confidence * np.array((1,0,0)).reshape(1, 1, 1, 3)/4000.0
              )
          )
      )
                                  )

  vis_est = tf.nn.l2_normalize(illums_est, 3)
  
  exposure_boost = 5

  img = tf.pow(images[:, :, :, ::-1] / 65535 * exposure_boost, 1 / VIS_GAMMA)
  img_corrected = tf.pow(
      images[:, :, :, ::-1] / 65535 / illums_pooled[:, None, None, :] * exposure_boost *
      tf.reduce_mean(illums_pooled, axis=(1), keep_dims=True)[:, None, None, :],
      1 / VIS_GAMMA)

  visualization = [
      img,
      img_corrected,
      vis_confidence_colored,
      vis_confidence * vis_est,
      vis_est,
      #tf.nn.l2_normalize(illums_ground, 1)[:, None, None, :],
      tf.nn.l2_normalize(illums_pooled, 1)[:, None, None, :]
  ]

  fcn_padding = 0  # = int(224.0 / int(images.get_shape()[1])  * target_shape[0]) // 2 # For receptive field offsets

  ##################
  confidence_dist = confidence[:, :, :, None] / tf.reduce_sum(
      confidence, axis=(1, 2), keep_dims=True)[:, :, :, None]
  mean_est = tf.reduce_mean(vis_est, axis=(1, 2), keep_dims=True)
  sq_deviation = tf.pow(vis_est - mean_est, 2)
  weighted_sq_dev = confidence_dist * sq_deviation
  variance = tf.reduce_sum(weighted_sq_dev, axis=(1, 2))
  ##################

  for i in range(len(visualization)):
    vis = visualization[i]
    if i == 0:
      padding = 0
    else:
      padding = fcn_padding
    if int(vis.get_shape()[3]) == 1:
      vis = vis * np.array((1, 1, 1)).reshape(1, 1, 1, 3)
    vis = tf.image.resize_images(
        vis, (target_shape[0] - padding * 2, target_shape[1] - padding * 2),
        method=tf.image.ResizeMethod.AREA)

    vis = tf.pad(vis,
                 tf.constant([[0, 0], [padding, padding], [padding, padding],
                              [0, 0]]))
    vis = tf.pad(vis - 1, tf.constant([[0, 0], [4, 4], [4, 4], [0, 0]])) + 1
    visualization[i] = vis

  visualization[3] = visualization[0] * visualization[2]

  visualization_lines = []
  images_per_line = 3
  for i in range(len(visualization) // images_per_line):
    visualization_lines.append(
        tf.concat(
            axis=2,
            values=visualization[i * images_per_line:(i + 1
                                                     ) * images_per_line]))
  visualization = tf.maximum(0.0, tf.concat(axis=1, values=visualization_lines))
  print 'visualization shape', visualization.shape

  return visualization


def get_weighted_variance(image, illums_est):
  confidence = tf.sqrt(tf.reduce_sum(illums_est**2, axis=3))
  vis_est = tf.nn.l2_normalize(illums_est, 3)
  ##################
  confidence_dist = confidence[:, :, :, None] / tf.reduce_sum(
      confidence, axis=(1, 2), keep_dims=True)[:, :, :, None]
  mean_est = tf.reduce_mean(vis_est, axis=(1, 2), keep_dims=True)
  sq_deviation = tf.pow(vis_est - mean_est, 2)
  weighted_sq_dev = confidence_dist * sq_deviation
  variance = tf.reduce_sum(weighted_sq_dev, axis=(1, 2))
  ##################
  return variance


def get_gram_matrix(illum_est):
  #    assert illum_est.shape[0] == 1
  width, height = illum_est.get_shape().as_list()[
      1], illum_est.get_shape().as_list()[2]
  print illum_est.shape
  est_points = tf.reshape(illum_est[0], [width * height, 3])
  gram = tf.matmul(tf.transpose(est_points), est_points)
  # todo: we should take the average
  return gram


# draw text on the bottom right corner of an image,
# lines like ['line1', 'line2']
def put_text_on_image(image, lines):
  for i, line in enumerate(lines[::-1]):
    text_width, text_height = cv2.getTextSize(line, cv2.FONT_HERSHEY_TRIPLEX,
                                              0.4, 1)[0]
    cv2.putText(image, line, (image.shape[1] - text_width,
                              image.shape[0] - 2 * i * text_height - 10),
                cv2.FONT_HERSHEY_TRIPLEX, 0.4, [0, 0, 0])
