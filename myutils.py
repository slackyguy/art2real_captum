import numpy as np
import torch
import torchvision

class CustomClassifierOutputTarget:
    def __init__(self, category):
        self.category = category
    def __call__(self, model_output):
        return model_output;
        # print(model_output.shape)
        # print(model_output[0])
        # print(model_output[1])
        # if len(model_output.shape) == 1:
        #     print(model_output[self.category])
        #     return model_output[self.category]
        # return model_output[:, self.category]