import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torchvision.utils import make_grid, save_image

# from gradcam.utils import visualize_cam
# from gradcam import GradCAM, GradCAMpp

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget, ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from torchvision.models import resnet50

from myutils import CustomClassifierOutputTarget, tensor_to_image

# #from captum.attr import Saliency
# import torch
# import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as transforms

# from captum.insights import AttributionVisualizer, Batch
# from captum.insights.attr_vis.features import ImageFeature

# from captum.attr import IntegratedGradients
# from captum.attr import GradientShap
# from captum.attr import Saliency
# from captum.attr import NoiseTunnel
# from captum.attr import visualization as viz

def baseline_func(input):
    return input * 0

transform = transforms.Compose([
 transforms.Resize(256),
 transforms.CenterCrop(224),
 transforms.ToTensor()
])
transform_normalize = transforms.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
 )


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.verbose = False
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options

    sample_index = 0
    sample = list(dataset)[sample_index]

    model.setup(opt)               # regular setup: load and print networks; create schedulers
    resnet = model.nets[0] # model.load_networks(opt.epoch)

    # model_methods = [method_name for method_name in dir(model)
    #               if callable(getattr(model, method_name))]
    # print(model_methods)

    # resnet_methods = [method_name for method_name in dir(model)
    #               if callable(getattr(model, method_name))]
    # print(resnet_methods)
    # print(resnet.get_current_losses())

    # RUN TEST
    # model.set_input(sample)  # unpack data from data loader
    # model.test()           # run inference
    # visuals = model.get_current_visuals()

    print(resnet.model)

    #resnet = models.resnet101(pretrained=True) #resnet50

    #target_layers = [resnet.layer4[-1]]
    target_layers = [resnet.model[18]] #torch.nn.Sequential(*list(resnet.children())[:layers_len]) #:-1
    #print(len(list(target_layers)))

    activations_and_grads = ActivationsAndGradients(
            resnet.model, target_layers, None) # reshape_transform
    outputs = activations_and_grads(sample['A'].cuda())

    i = 0
    for output in outputs:
        print(len(output))
        my_img = tensor_to_image(output)
        my_img.save("out" + str(i) + ".jpg")
        i = i + 1
    
    #print(visuals)
    #pil_img = Image.open(sample['A_paths'][0])

    #https://github.com/vickyliin/gradcam_plus_plus-pytorch/blob/master/example.ipynb
    # torch_img = transforms.Compose([
    # transforms.Resize((224, 224)),
    # transforms.ToTensor()
    #     ])(pil_img).to(device)
    
    # normed_torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None]
    # input_tensor = normed_torch_img



    
    # image = np.array(img)
    # rgb_img = np.float32(image) / 255
    # transformed_img = transform(img)
    # input = transform_normalize(transformed_img)
    
    # input_tensor = input.unsqueeze(0)
    # input_tensor = preprocess_image(rgb_img,
    #                             mean=[0.485, 0.456, 0.406],
    #                             std=[0.229, 0.224, 0.225])

    #cam = GradCAM(model=resnet, target_layers=target_layers, use_cuda=True)

    # # Construct the CAM object once, and then re-use it on many images:
    # #cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    # cam = EigenCAM(model, target_layers, use_cuda=True)
    # grayscale_cam = cam(input_tensor)[0, :, :]
    # cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    # Image.fromarray(cam_image)

    # You can also use it within a with statement, to make sure it is freed,
    # In case you need to re-create it inside an outer loop:
    # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
    #   ...

    #print(visuals)
    #torch.cuda.empty_cache()

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category
    # will be used for every image in the batch.
    # Here we use ClassifierOutputTarget, but you can define your own custom targets
    # That are, for example, combinations of categories, or specific outputs in a non standard model.
    # targets = [CustomClassifierOutputTarget(284)]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    # grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    # rgb_img = Image.new('RGB', (224,224))

    # In this example grayscale_cam has only one image in the batch:
    # grayscale_cam = grayscale_cam[0, :]
    # visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    # im1 = rgb_img.save("./gradmap.jpg")