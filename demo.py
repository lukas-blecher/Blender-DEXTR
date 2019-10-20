import os
import sys
import torch
from collections import OrderedDict
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

from torch.nn.functional import interpolate

import networks.deeplab_resnet as resnet
from mypath import Path
from dataloaders import helpers as helpers
import argparse
import os
import glob
import easygui
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', type=str, default='ims/dog-cat.jpg', help='path to image')
    parser.add_argument('--model-name', type=str, default='dextr_pascal-sbd')
    parser.add_argument('-o', '--output', type=str, default='results', help='path where results will be saved')
    parser.add_argument('--pad', type=int, default=50, help='padding size')
    parser.add_argument('--thres', type=float, default=.9)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--anchors', type=int, default=5, help='amount of points to set')
    parser.add_argument('--anchor-points', type=str, default=None, help='path to folder of anchor points (tracking points)')
    parser.add_argument('--use-frame-info', type=bool, default=True, help='wheter to use the frame number from the csv file or not')
    parser.add_argument('--corrections', action='store_true', help='toggle popup message wheater to correct or not')
    parser.add_argument('--cut', action='store_true', help='if used, will save the cutted image instead of the mask as png')
    
    opt = parser.parse_args()
    modelName = opt.model_name
    pad = opt.pad
    thres = opt.thres
    gpu_id = opt.gpu_id
    device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")

    #  Create the network and load the weights
    net = resnet.resnet101(1, nInputChannels=4, classifier='psp')
    print("Initializing weights from: {}".format(os.path.join(Path.models_dir(), modelName + '.pth')))
    state_dict_checkpoint = torch.load(os.path.join(Path.models_dir(), modelName + '.pth'),
                                       map_location=lambda storage, loc: storage)
    # Remove the prefix .module from the model when it is trained using DataParallel
    if 'module.' in list(state_dict_checkpoint.keys())[0]:
        new_state_dict = OrderedDict()
        for k, v in state_dict_checkpoint.items():
            name = k[7:]  # remove `module.` from multi-gpu training
            new_state_dict[name] = v
    else:
        new_state_dict = state_dict_checkpoint
    net.load_state_dict(new_state_dict)
    net.eval()
    net.to(device)

    #  Read image and click the points
    if os.path.isfile(opt.image):
        images = [opt.image]
    else:
        images = sorted(glob.glob(opt.image+'/*.*'))
    if opt.anchor_points:
        tracks = sorted(glob.glob(opt.anchor_points+'/*.csv'))
        frames, X, Y = [], [], []
        for i in range(len(tracks)):
            f, x, y = np.loadtxt(tracks[i], delimiter=',', unpack=True)
            frames.append(f.tolist())
            X.append(x.tolist())
            Y.append(y.tolist())
        anchorPoints = []
        uframes = np.unique(np.hstack([np.array(a) for a in frames])).tolist()
        # print(uframes)
        for i in range(len(uframes)):
            extreme_points = []
            for j in range(len(frames)):
                try:
                    ind = frames[j].index(uframes[i])
                    extreme_points.append([X[j][ind], Y[j][ind]])
                except ValueError:
                    continue
            anchorPoints.append(np.array(extreme_points))

    for i, img in enumerate(images):

        if opt.use_frame_info and opt.anchor_points is not None:
            file_number = int(re.sub(r'\D', '', img))
            if not file_number in uframes:
                print(img, 'skipped')
                continue

        if opt.anchor_points is None:
            plt.figure()
        while True:
            image = np.array(Image.open(img))
            mask_path = os.path.join(opt.output, os.path.split(img)[1])
            if opt.anchor_points is None:
                plt.ion()
                plt.axis('off')
                plt.imshow(image)
                plt.title('Click the four extreme points of the objects\nHit enter/middle mouse button when done (do not close the window)')

            results = []

            with torch.no_grad():
                # while 1:
                if opt.anchor_points:
                    if opt.use_frame_info:
                        try:
                            index = uframes.index(file_number)
                        except ValueError:
                            print('Could not find data for frame %i. Use frame %i instead.' % (file_number, i))
                            index = i
                    else:
                        index = i
                    extreme_points_ori = anchorPoints[index].astype(np.int)
                else:
                    extreme_points_ori = np.array(plt.ginput(opt.anchors, timeout=0)).astype(np.int)

                # print(extreme_points_ori,extreme_points_ori.shape)
                #  Crop image to the bounding box from the extreme points and resize
                bbox = helpers.get_bbox(image, points=extreme_points_ori, pad=pad, zero_pad=False)
                crop_image = helpers.crop_from_bbox(image, bbox, zero_pad=True)
                resize_image = helpers.fixed_resize(crop_image, (512, 512)).astype(np.float32)

                #  Generate extreme point heat map normalized to image values
                extreme_points = extreme_points_ori - [np.min(extreme_points_ori[:, 0]), np.min(extreme_points_ori[:, 1])] + [pad, pad]
                extreme_points = (512 * extreme_points * [1 / crop_image.shape[1], 1 / crop_image.shape[0]]).astype(np.int)
                extreme_heatmap = helpers.make_gt(resize_image, extreme_points, sigma=10)
                extreme_heatmap = helpers.cstm_normalize(extreme_heatmap, 255)

                #  Concatenate inputs and convert to tensor
                input_dextr = np.concatenate((resize_image, extreme_heatmap[:, :, np.newaxis]), axis=2)
                inputs = torch.from_numpy(input_dextr.transpose((2, 0, 1))[np.newaxis, ...])

                # Run a forward pass
                inputs = inputs.to(device)
                outputs = net.forward(inputs)
                outputs = interpolate(outputs, size=(512, 512), mode='bilinear', align_corners=True)
                outputs = outputs.to(torch.device('cpu'))

                pred = np.transpose(outputs.data.numpy()[0, ...], (1, 2, 0))
                pred = 1 / (1 + np.exp(-pred))
                pred = np.squeeze(pred)
                result = helpers.crop2fullmask(pred, bbox, im_size=image.shape[:2], zero_pad=True, relax=pad) > thres
                
                results.append(result)
                # Plot the results
                plt.imshow(helpers.overlay_masks(image / 255, results))
                plt.plot(extreme_points_ori[:, 0], extreme_points_ori[:, 1], 'gx')

                if not opt.cut:
                    helpers.save_mask(results, mask_path)
                else:
                    Image.fromarray(np.concatenate((image, 255*result[..., None].astype(np.int)), 2).astype(np.uint8)).save(mask_path, 'png')
                '''if len(extreme_points_ori) < 4:
                        if len(results) > 0:
                            helpers.save_mask(results, 'demo.png')
                            print('Saving mask annotation in demo.png and exiting...')
                        else:
                            print('Exiting...')
                        sys.exit()'''
            if opt.anchor_points is None:
                plt.close()
            if opt.corrections:
                if easygui.ynbox(image=mask_path):
                    break
            else:
                break
        print(img, 'done')


if __name__ == '__main__':
    main()
