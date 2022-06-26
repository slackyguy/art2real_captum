import numpy as np
import torch
import torchvision

class CustomClassifierOutputTarget:
    def __init__(self, category):
        self.category = category
    def __call__(self, model_output):
        print(model_output.shape)
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]