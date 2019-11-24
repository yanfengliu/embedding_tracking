from experiment import Experiment
from params import Params


params = Params()

params.EMBEDDING_DIM            = 24
params.BATCH_SIZE               = 1
params.NUM_SHAPE                = 6
params.NUM_CLASSES              = 4 # (3 shapes + 1 background)
params.NUM_FILTER               = [256, 128]
params.ETH_MEAN_SHIFT_THRESHOLD = 1.5
params.DELTA_VAR                = 0.5
params.DELTA_D                  = 1.5
params.IMG_SIZE                 = 256
params.OUTPUT_SIZE              = 64
params.SEQUENCE_LEN             = 100
params.TRAIN_NUM_SEQ            = 500
params.VAL_NUM_SEQ              = 50
params.TEST_NUM_SEQ             = 50
params.RANDOM_SIZE              = True
params.OPTICAL_FLOW_WEIGHT      = 0
params.BACKBONE                 = 'xception'
params.GITHUB_DIR               = 'C:/Users/yliu60/Documents/GitHub'
params.LEARNING_RATE            = 1e-4
params.EPOCHS                   = 30
params.EPOCHS_PER_SAVE          = 5
params.STEPS_PER_VISUAL         = 1000
params.FEATURE_STRING           = f'{params.NUM_SHAPE}_shapes'
params.MODEL_SAVE_DIR           = f'model/{params.NUM_SHAPE}_shapes'
params.TRAIN_SET_PATH           = f'dataset/{params.NUM_SHAPE}_shapes/train'
params.VAL_SET_PATH             = f'dataset/{params.NUM_SHAPE}_shapes/val'
params.TEST_SET_PATH            = f'dataset/{params.NUM_SHAPE}_shapes/test'
params.IOU_THRESHOLD            = 0.5
params.MASK_AREA_THRESHOLD      = 20

experiment = Experiment(params)
experiment.run()