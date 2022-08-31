import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import SimpleITK as sitk

from lib.data import DatasetGenerator, LocatorDataGenerator
from lib.metrics import *
from lib.models.vnet import VNet
from lib.models.vnet_attention import VNetAttention
from lib.models.unet3d import UNet3D


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_dir',
        dest='root_dir',
        default='./testing',
        type=str,
        help='Directory of the LASC18 dataset')
    parser.add_argument(
        '--locator_path',
        dest='locator_path',
        default='./runs/weights/locator_vnetattn.pt',
        type=str,
        help='Best locator path to load the trained locator model before performing testing')
    parser.add_argument(
        '--segmentor_path',
        dest='segmentor_path',
        default='./runs/weights/vnetattn_best.pt',
        type=str,
        help='Best segmentation model path')
    parser.add_argument(
        '--gpu',
        action='store_true',
        dest='gpu',
        default=True,
        help='use cuda')
    parser.add_argument(
        '--num_layers',
        dest='num_layers',
        default=1,
        type=int,
        help='Number of convolution layers in addition to default layers at each level for both models')
    parser.add_argument(
        '--output_dir',
        default="./testing/test/preds",
        type=str,
        help='Output directory to store the model predictions')
    args = parser.parse_args()

    assert args.locator_path is not None, "Locator load path must be provided during testing mode"
    assert os.path.exists(
        args.locator_path), "Provided locator load path doesnt exist"
    assert args.segmentor_path is not None, "Segmentation model load path must be provided during testing mode"
    assert os.path.exists(
        args.segmentor_path), "Provided segmentor load path doesnt exist"
    assert args.root_dir is not None, "Root directory to load test image set must be provided during testing mode"

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    root_dir = args.root_dir
    use_cuda = torch.cuda.is_available() and args.gpu
    device = torch.device("cuda" if use_cuda else "cpu")

    net_locator = VNetAttention(in_channels=1, classes=1)
    print(f'Initialised Locator model.')

    net = VNetAttention(in_channels=1, classes=1)
    print(f'Initialised segmentation model.')

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"""Provided locator checkpoint load path: '{args.locator_path}'
                                    and segmentor load path: '{args.segmentor_path}' """)

    locator_checkpoint = torch.load(args.locator_path, map_location='cpu')
    segmentor_checkpoint = torch.load(args.segmentor_path, map_location='cpu')

    net_locator.load_state_dict(locator_checkpoint['locator_model_state'])
    net.load_state_dict(segmentor_checkpoint['model_state'])
    patch_size = segmentor_checkpoint['patch_size']
    scale_factor = segmentor_checkpoint['scale_factor']
    mean = segmentor_checkpoint['mean']
    std = segmentor_checkpoint['std']

    print(
        f"Checkpoint locator model weights loaded from resume path: '{args.locator_path}'")
    print(
        f"Checkpoint segmentation model weights loaded from resume path: '{args.segmentor_path}'")
    print(
        f"Using patch size: {patch_size}, mean:{mean} and std:{std} over the data.")

    if use_cuda:
        net_locator.to(device)
        net.to(device)
        print(f'Testing with {device} using 1 GPUs.')

    test_set_builder = LocatorDataGenerator(
        root_dir=root_dir, mode='test', scale_factor=scale_factor)
    test_builder_inputs = []

    # Roughly locate the LA structure in the test images
    net_locator.eval()
    with torch.no_grad():
        for idx in range(len(test_set_builder)):
            raw_file_name, raw_image = test_set_builder[idx]
            raw_image = torch.unsqueeze(raw_image, dim=0).to(device)

            train_output = net_locator(raw_image)

            test_builder_inputs.append(((raw_file_name, train_output), None))

    # Use localisation prediction for patch slice generation and to build
    # final test image tensors
    test_set = DatasetGenerator(
        mode='test',
        inputs=test_builder_inputs,
        pad=(
            30,
            30,
            30),
        scale_factor=scale_factor,
        mean=mean,
        std=std,
        patch_dims=patch_size)

    mean_iou = []
    mean_p = []
    mean_r = []
    mean_d = []

    net.eval()
    with torch.no_grad():
        sigmoid = nn.Sigmoid()
        for index in range(len(test_set)):
            raw_image_filename, raw_image, slice_range, output_dims = test_set[index]
            raw_image = torch.unsqueeze(raw_image, dim=0).to(device)

            prediction = net(raw_image)

            prediction = sigmoid(prediction)
            prediction[prediction < 1] = 0
            prediction = prediction[0, 0].to(
                dtype=torch.uint8).detach().cpu().numpy()

            prediction_helper = np.zeros((96, 640, 640), dtype=np.uint8)
            prediction_helper[slice_range] = prediction

            # Gather slice ranges to recreate original input data dimension for
            # each predicted array
            depth_diff = prediction_helper.shape[0] - output_dims[0]
            depth_start = int(depth_diff / 2)
            depth_end = int(prediction_helper.shape[0] - depth_diff / 2)

            height_diff = prediction_helper.shape[1] - output_dims[1]
            height_start = int(height_diff / 2)
            height_end = int(prediction_helper.shape[1] - height_diff / 2)

            width_diff = prediction_helper.shape[2] - output_dims[2]
            width_start = int(width_diff / 2)
            width_end = int(prediction_helper.shape[2] - width_diff / 2)

            # Select from the prediction array to create final output similar
            # to original image arrays
            prediction = prediction_helper[(slice(depth_start, depth_end, None),
                                            slice(height_start, height_end, None),
                                            slice(width_start, width_end, None)
                                            )]

            gt_img = sitk.GetArrayFromImage(sitk.ReadImage(
                f"{args.root_dir}/test/label/{raw_image_filename.split('_')[0]}_laendo.nrrd"))

            prediction = ((prediction - prediction.min()) * (1 / \
                          (prediction.max() - prediction.min()) * 255)).astype('uint8')

            iou = iou_score(prediction, gt_img)
            p = precision(prediction, gt_img)
            r = recall(prediction, gt_img)
            d = dice_score(prediction, gt_img)

            mean_iou.append(iou)
            mean_p.append(p)
            mean_r.append(r)
            mean_d.append(d)

    mean_iou = np.mean(mean_iou)
    mean_p = np.mean(mean_p)
    mean_r = np.mean(mean_r)
    mean_d = np.mean(mean_d)

    print("-" * 30)

    print(f"Mean IoU      : {mean_iou}")
    print(f"Mean Precision: {mean_p}")
    print(f"Mean Recall   : {mean_r}")
    print(f"Mean Dice   : {mean_d}")
