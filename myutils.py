import numpy as np
import torch
import torchvision

from PIL import Image

class CustomClassifierOutputTarget:
    def __init__(self, category):
        self.category = category
    def __call__(self, model_output):
        print(model_output.shape)
        img = tensor_to_image(model_output)
        img.save('output.jpg')
        # print(model_output[0])
        # print(model_output[1])
        if len(model_output.shape) == 1:
            print(model_output[self.category])
            return model_output[self.category]
        return model_output[:, self.category]

def tensor_to_image(tensor):
    #arr = np.ndarray(tensor)#This is your tensor
    #arr_ = np.squeeze(arr) # you can give axis attribute if you wanna squeeze in specific dimension
    #plt.imshow(arr_)
    img_data = tensor.detach().cpu().numpy()[0] #Image.fromarray(arr_)
    img_data = (img_data + 1) * 127.5

    img = Image.fromarray(img_data)
    return img.convert('RGB')
    # tensor = tensor*255
    # tensor = np.array(tensor.cpu().detach().numpy(), dtype=np.uint8)
    # if np.ndim(tensor)>3:
    #     assert tensor.shape[0] == 1
    #     tensor = tensor[0]
    # return Image.fromarray(tensor)

activation = {}
def get_activation(name):
    def hook(model, input, output):
        print(len(input[0]))
        activation[name] = {}
        activation[name]['input'] = input
        activation[name]['output'] = output.detach()
    return hook