MODEL_PATH = "models/fc4"

FCN_INPUT_SIZE = 512

##################################
# Data augmentation
##################################

# Use data augmentation?
AUGMENTATION = True
# Rotation angle
AUGMENTATION_ANGLE = 30
# Patch scale
AUGMENTATION_SCALE = [0.1, 1.0]
# Random left-right flip?
AUGMENTATION_FLIP_LEFTRIGHT = True
# Random top-down flip?
AUGMENTATION_FLIP_TOPDOWN = False
# Color rescaling?
AUGMENTATION_COLOR = 0.0
# Cross-channel terms
AUGMENTATION_COLOR_OFFDIAG = 0.0
# Augment Gamma?
AUGMENTATION_GAMMA = 0.0
# Augment using a polynomial curve?
USE_CURVE = False
# Apply different gamma and curve to left/right halves?
SPATIALLY_VARIANT = False


def config_get_input_gamma():
  return INPUT_GAMMA


def config_set_input_gamma(value):
  global INPUT_GAMMA
  INPUT_GAMMA = value


# The gamma used in the AlexNet branch to make patches in sRGB
INPUT_GAMMA = 2.2

# The gamma for visualization
VIS_GAMMA = 2.2

# Shuffle the images, after each epoch?
DATA_SHUFFLE = True

VISUALIZE = 0
SMOOTHNESS_REGULARIZATION = 0
GLOBAL_WEIGHT_DECAY = 5.7e-5
FEED_ALEX = True
FEED_SHALLOW = False
USE_SHORTCUT = False
ALEX_OUTPUT = 5
SHOW_CONV1 = False
DROPOUT = 0.5
TRAINING_BATCH_SIZE = 16
TRAIN_UPSCORE = False
FC1_SIZE = 64
FC1_KERNEL_SIZE = 6
# We only store ckpts that are good...
CKPT_PERIOD = 0
# How often (in epochs) do we test?
TEST_PERIOD = 20
TEST_SAMPLES = 8
PER_PATCH_WEIGHT = 0.0
LENGTH_REGULARIZATION = 0.0
SHALLOW_CHANNELS = [32, 32, 64, 128, 256]
CONV_DEPTH = 5
TRIGGER_STARTING_POINT = 60
VISUALIZATION_SIZE = 512
WRITE_SUMMARY = True
IMAGE_SUMMARY_INT = 10

USE_UV = False
WEIGHTED_POOLING = True
FC_POOLING = False

#Optimizers
ALEX_FINE_TUNE = 0
FORCE_ADAM = 0
OPTIMIZER = 'ADAM'
#OPTIMIZER='SGD'
FINE_TUNE_LR_RATIO = 1e-1
BASE_LEARNING_RATE = 3e-4
LR_DECAY = 1
LR_DECAY_INTERVAL = 100
MOMENTUM = 0.9

RESIZE_TEST = False

#Visualization
MERGED_IMAGE_SIZE = 400

# Data Sets
DATASET_NAME = 'gehler'
SUBSET = 0
FOLD = 0
TRAINING_FOLDS = []
TEST_FOLDS = []


def initialize_dataset_config(dataset_name=None, subset=None, fold=None):
  global DATASET_NAME, SUBSET, FOLD
  if dataset_name is not None:
    DATASET_NAME = dataset_name
    SUBSET = subset
    FOLD = int(fold)
  global TRAINING_FOLDS, TEST_FOLDS
  if DATASET_NAME == 'gehler':
    T = FOLD
    print 'FOLD', FOLD
    if T != -1:
      TRAINING_FOLDS = ['g%d' % (T), 'g%d' % ((T + 1) % 3)]
      TEST_FOLDS = ['g%d' % ((T + 2) % 3)]
    else:
      TRAINING_FOLDS = []
      TEST_FOLDS = ['g0', 'g1', 'g2']
  elif DATASET_NAME == 'cheng':
    subset = SUBSET
    T = FOLD
    TRAINING_FOLDS = ['c%s%d' % (subset, T), 'c%s%d' % (subset, (T + 1) % 3)]
    TEST_FOLDS = ['c%s%d' % (subset, (T + 2) % 3)]
  elif DATASET_NAME == 'multi':
    TEST_FOLDS = ['multi']

  print(TRAINING_FOLDS)
  print(TEST_FOLDS)
  return TRAINING_FOLDS, TEST_FOLDS


# Saver
CKPTS_TO_KEEP = 0
EPOCHS = 1300

##########################
# Test
##########################

# Test the images are multiple resolution, and then do a weighted average? (not helping)
MULTIRES_TEST = False
ANGULAR_LOSS = True
SMOOTH_L1 = False
TRAINING_VISUALIZATION = 200
# Up/down scale images for testing? (Keeping the aspect ratio)
TEST_BASE_RES = 0.5
SEPERATE_CONFIDENCE = False

OVERRODE = {}
