import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

from PIL import Image

#from captum.attr import Saliency
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import ImageFeature

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Saliency
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

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
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options

    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # Captum
    # normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # visualizer = AttributionVisualizer(
    #     models=[model],
    #     score_func=lambda o: torch.nn.functional.softmax(o, 1),
    #     classes=[ "Portrait" ],
    #     features=[
    #         ImageFeature(
    #             "Photo",
    #             baseline_transforms=[baseline_func],
    #             input_transforms=[normalize],
    #         )
    #     ],
    #     dataset=dataset,
    # )
    # visualizer.render()
    
    # prediction_score, pred_label_idx = torch.topk(output, 1)
    # pred_label_idx.squeeze_()
    # predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
    # print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')


    #print(model)
    #saliency = Saliency(model)
    # show a screenshot if using notebook non-interactively
    # from IPython.display import Image
    # Image(filename='img/captum_insights.png')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    torch.cuda.empty_cache()

    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference

        visuals = model.get_current_visuals()  # get image results

        #print(list(visuals.values())[0])
        #print(data['A'])
        print(list(dataset)[i]['A_paths'][0]) #transform_normalize

        # Create IntegratedGradients object and get attributes
        output = F.softmax(list(visuals.values())[0], dim=1)
        #transformed_img = transform(data)
        #transformed_img = data['A']
        net = list(model.load_networks(opt.epoch))
        integrated_gradients = IntegratedGradients(net[0])
        
        #https://gilberttanner.com/blog/interpreting-pytorch-models-with-captum/
        img = Image.open('./datasets/monet2photo/testA/00001.jpg')
        transformed_img = transform(img)
        input = transform_normalize(transformed_img)
        input = input.unsqueeze(0)

        #attributions_ig = integrated_gradients.attribute(transformed_img.cuda(), n_steps=200) #target=pred_label_idx
        attributions_ig = integrated_gradients.attribute(input.cuda(), n_steps=200) #target=pred_label_idx

        # create custom colormap for visualizing the result
        default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                        [(0, '#ffffff'),
                                                        (0.25, '#000000'),
                                                        (1, '#000000')], N=256)


        # visualize the results using the visualize_image_attr helper method
        _ = viz.visualize_image_attr_multiple(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
                                    np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                    methods=["original_image", "heat_map"],
                                    signs=['all', 'positive'],
                                    cmap=default_cmap,
                                    show_colorbar=True)

        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  # save the HTML
