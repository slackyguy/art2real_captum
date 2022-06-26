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
from torchvision.models import resnet50

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

    model.setup(opt)               # regular setup: load and print networks; create schedulers

    #model = resnet50(pretrained=True)
    #resnet = models.resnet101(pretrained=True)
    resnet = list(model.load_networks(opt.epoch))[0]

    #target_layers = [net.layer4[-1]]
    target_layers = torch.nn.Sequential(*list(resnet.children())[:-1])

    pil_img = Image.open(list(dataset)[0]['A_paths'][0])

    #https://github.com/vickyliin/gradcam_plus_plus-pytorch/blob/master/example.ipynb
    torch_img = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
        ])(pil_img).to(device)
    
    normed_torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None]
    input_tensor = normed_torch_img
    
    # configs = [
    #     dict(model_type='resnet', arch=resnet, layer_name='layer4')
    # ]
    # for config in configs:
    #     config['arch'].to(device).eval()
    # cams = [
    #     [cls.from_config(**config) for cls in (GradCAM, GradCAMpp)]
    #     for config in configs
    # ]
    
    # images = []
    # for gradcam, gradcam_pp in cams:
    #     mask, _ = gradcam(normed_torch_img)
    #     heatmap, result = visualize_cam(mask, torch_img)
    #     mask_pp, _ = gradcam_pp(normed_torch_img)
    #     heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)
    #     images.extend([torch_img.cpu(), heatmap, heatmap_pp, result, result_pp])
    # grid_image = make_grid(images, nrow=5)
    # output_maps = transforms.ToPILImage()(grid_image)
    # im1 = output_maps.save("./gradmap.jpg")
    # print(im1)
    
    
    # image = np.array(img)
    # rgb_img = np.float32(image) / 255
    # # transformed_img = transform(img)
    # # input = transform_normalize(transformed_img)
    
    # # Create an input tensor image for your model.
    # # Note: input_tensor can be a batch tensor with several images!
    
    # input_tensor = input.unsqueeze(0)
    # input_tensor = preprocess_image(rgb_img,
    #                             mean=[0.485, 0.456, 0.406],
    #                             std=[0.229, 0.224, 0.225])

    cam = GradCAM(model=resnet, target_layers=target_layers, use_cuda=True)

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

    model.set_input(list(dataset)[0])  # unpack data from data loader
    model.test()           # run inference
    visuals = model.get_current_visuals()
    print(visuals)
    #torch.cuda.empty_cache()

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category
    # will be used for every image in the batch.
    # Here we use ClassifierOutputTarget, but you can define your own custom targets
    # That are, for example, combinations of categories, or specific outputs in a non standard model.
    targets = [ClassifierOutputTarget(284)]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    rgb_img = Image.new('RGB', (255,255))

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)