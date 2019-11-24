from experiment import Experiment
from params import Params


params = Params()

params.EMBEDDING_DIM            = 24
params.BATCH_SIZE               = 1
params.NUM_SHAPE                = 6
params.NUM_CLASSES              = 4
params.NUM_FILTER               = [256, 128]
params.ETH_MEAN_SHIFT_THRESHOLD = 1.5
params.DELTA_VAR                = 0.5
params.DELTA_D                  = 1.5
params.IMG_SIZE                 = 256
params.OUTPUT_SIZE              = 64
params.SEQUENCE_LEN             = 100
params.TRAIN_NUM_SEQ            = 2
params.TEST_NUM_SEQ             = 2
params.RANDOM_SIZE              = True
params.OPTICAL_FLOW_WEIGHT      = 5
params.BACKBONE                 = 'xception'
params.GITHUB_DIR               = 'C:/Users/yliu60/Documents/GitHub'
params.LEARNING_RATE            = 1e-4
params.EPOCHS                   = 100
params.EPOCHS_PER_SAVE          = 5
params.STEP_PER_VISUAL          = 20
params.MODEL_SAVE_DIR           = 'model'
params.MODEL_SAVE_NAME          = 'amodal_track.h5'
params.TRAIN_SET_PATH           = f'dataset/{params.NUM_SHAPE}_shapes/train'
params.VAL_SET_PATH             = f'dataset/{params.NUM_SHAPE}_shapes/val'
params.TEST_SET_PATH            = f'dataset/{params.NUM_SHAPE}_shapes/test'
params.IOU_THRESHOLD            = 0.5
params.MASK_AREA_THRESHOLD      = 20
params.FEATURE_STRING           = f'{params.NUM_SHAPE}_shape'

experiment = Experiment(params)
experiment.run()