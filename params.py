import os
import numpy as np


class Params(object):
    def __init__(self):
        self.EMBEDDING_DIM            = None
        self.BATCH_SIZE               = None
        self.NUM_CLASSES              = None
        self.NUM                      = None
        self.NUM_FILTER               = None
        self.ETH_MEAN_SHIFT_THRESHOLD = None
        self.DELTA_VAR                = None
        self.DELTA_D                  = None
        self.IMG_SIZE                 = None
        self.BACKBONE                 = None
    
    def display_values(self):
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")