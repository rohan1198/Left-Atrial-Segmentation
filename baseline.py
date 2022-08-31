import os
import shutil
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.data import LocatorDataGenerator
from lib.utils import number_of_parameters, get_logger
from lib.losses import Dice, IoULoss
from lib.models.unet3d import UNet3D
from lib.models.vnet import VNet
from lib.models.vnetlight import VNetLight
from lib.models.vnet_attention import VNetAttention
from lib.metrics import SegmentationMetrics, dice_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_dir',
        default='~/Drive/lasc18_dataset',
        type=str,
        help='path to the LASC18 dataset')
    parser.add_argument(
        '--max_epochs',
        default=200,
        type=int,
        help='number of epochs')
    parser.add_argument('--batch_size', default=5, type=int, help='batch size')
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "unet3d",
            "vnet",
            "vnetlight",
            "vnetattn"],
        default="vnet")
    parser.add_argument(
        '--scale_factor',
        default=(
            0.5,
            0.25,
            0.25),
        type=tuple,
        help='scale down factor(D,H,W) for locator model training')
    parser.add_argument(
        '--lr',
        default=0.0001,
        type=float,
        help='optimizer learning rate')
    parser.add_argument(
        '--loss_criterion',
        default='dice',
        type=str,
        help='loss function to be used')
    parser.add_argument(
        '--save_after_epochs',
        default=100,
        type=int,
        help='number of epochs after which state is saved by default')
    parser.add_argument(
        '--validate_after_epochs',
        default=1,
        type=int,
        help='number of epochs after which validation occurs')
    parser.add_argument('--seed', default=42, type=int, help='seed for RNG')
    parser.add_argument(
        '--gpu',
        action='store_true',
        default=True,
        help='use cuda')
    parser.add_argument(
        '--locator_resume',
        default=None,
        type=str,
        help='name of stored model checkpoint state file to resume training')
    parser.add_argument(
        '--best_locator',
        default=None,
        type=str,
        help='File path for the finalised best locator model')
    parser.add_argument(
        '--patience',
        default=15,
        type=int,
        help='LR Scheduler patience')
    parser.add_argument(
        '--reduce',
        default=0.85,
        type=float,
        help='LRScheduler learning_rate reduction factor ')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    root_dir = args.root_dir
    max_epochs = args.max_epochs
    start_epoch = 0
    batch_size = args.batch_size
    loss_criterion = args.loss_criterion

    use_cuda = torch.cuda.is_available() and args.gpu
    device = torch.device("cuda" if use_cuda else "cpu")
    factor = args.reduce
    patience = args.patience
    alpha = 1

    locator_loss_function = Dice()

    logger = get_logger(f"{args.model}_locator_{loss_criterion}")
    writer = SummaryWriter(log_dir="runs")

    best_locator_validation_score = 0
    best_locator_available = False

    locator_training_error = {f'{args.loss_criterion}_loss': ([], [])}
    locator_validation_error = {f'{args.loss_criterion}_loss': ([], [])}

    if args.model == "unet3d":
        net_locator = UNet3D(in_channels=1, classes=1)
    elif args.model == "vnet":
        net_locator = VNet(in_channels=1, classes=1)
    elif args.model == "vnetlight":
        net_locator = VNetLight(in_channels=1, classes=1)
    elif args.model == "vnetattn":
        #net_locator = VNetAttention(in_channels = 1, classes = 1)
        net_locator = VNetAttention()

    logger.info(f'Initialised {args.model} locator model.')
    # logger.info(net_locator.test())

    locator_optimizer = optim.Adam(net_locator.parameters(), lr=args.lr)
    locator_scheduler = lr_scheduler.ReduceLROnPlateau(
        locator_optimizer, mode='min', factor=factor, patience=patience, verbose=True)

    if args.best_locator is None and args.locator_resume is not None:
        assert os.path.isfile(
            args.locator_resume), "Locator resume file path provided doesn't exist!"
        checkpoint = torch.load(args.locator_resume, map_location='cpu')

        start_epoch = int(checkpoint['epoch']) + 1
        max_epochs = int(checkpoint['max_epochs'])
        locator_optimizer.load_state_dict(
            checkpoint['locator_optimizer_state'])
        net_locator.load_state_dict(checkpoint['locator_model_state'])
        best_locator_validation_score = float(
            checkpoint['best_locator_validation_score'])
        locator_training_error = checkpoint['locator_training_error']
        locator_validation_error = checkpoint['locator_validation_error']

        logger.info(
            f"Checkpoint locator model state loaded from resume path: '{args.locator_resume}'")

    elif args.best_locator is not None and args.locator_resume is None:
        assert os.path.isfile(
            args.best_locator), "Best locator load path provided doesn't exist!"
        best_checkpoint = torch.load(args.best_locator, map_location='cpu')
        best_locator_available = True
        net_locator.load_state_dict(best_checkpoint['locator_model_state'])

        logger.info(f"Checkpoint loaded from path: {args.best_locator}")

    logger.info(
        f'Number of Trainable parameters: {number_of_parameters(net_locator)}')

    if use_cuda:
        net_locator.to(device)
        logger.info(f'GPU available: {torch.cuda.is_available()}')

    if not (best_locator_available):
        locator_train_set = LocatorDataGenerator(
            root_dir=root_dir, mode='train', scale_factor=args.scale_factor)
        locator_validation_set = LocatorDataGenerator(
            root_dir=root_dir, mode='validate', scale_factor=args.scale_factor)

        logger.info(f'Created locator dataset generator...')

        locator_train_set_loader = DataLoader(
            locator_train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=16)
        locator_validation_set_loader = DataLoader(
            locator_validation_set, batch_size=1, shuffle=True, num_workers=16)

        logger.info("Created locator Data Loader...")

        logger.info(
            f"""Loaded locator training and validation datasets from '{root_dir}'
                                             Batch Size: {batch_size}
                                             -----------------------------------------------------------------------------
                                             Beginning model training from epoch: {start_epoch} / {max_epochs}
                                             Best validation score: {best_locator_validation_score}
                                             Adam optimiser with lr: {'{:.7f}'.format(locator_optimizer.param_groups[0]['lr'])}
                                             Scheduler ReduceLROnPlateau with mode: 'min', factor: {factor}, patience: {patience}
                                             Loss Criterion: '{loss_criterion}'
                                             -----------------------------------------------------------------------------
                                    """)

        for epoch in range(start_epoch, max_epochs):
            net_locator.train()
            locator_training_score = {
                f'{args.loss_criterion}_loss': torch.zeros(
                    1, requires_grad=False, dtype=torch.float32)}
            locator_validation_score = {
                f'{args.loss_criterion}_loss': torch.zeros(
                    1, requires_grad=False, dtype=torch.float32)}

            iou_score = []
            precision = []
            recall = []
            f1_score = []
            dice_score = []

            logger.info(
                f"""-------Locator training Epoch: [{epoch} / {max_epochs}]-------""")

            for iteration, data in enumerate(locator_train_set_loader):
                raw_image, label = data
                raw_image = raw_image.to(device)
                label = label.to(device)

                output = net_locator(raw_image)
                locator_loss = locator_loss_function(output, label)

                locator_optimizer.zero_grad()
                locator_loss.backward()
                locator_optimizer.step()
                writer.add_scalar(
                    f"lr/locator_{args.model}_learning_rate",
                    locator_optimizer.param_groups[0]['lr'],
                    epoch)

                locator_training_score[f'{args.loss_criterion}_loss'] = torch.cat(
                    (locator_training_score[f'{args.loss_criterion}_loss'],
                     locator_loss.detach().view(1))) if iteration > 0 else locator_loss.detach().view(1)

            locator_train_error = torch.mean(
                locator_training_score[f'{args.loss_criterion}_loss'])
            locator_training_error[f'{args.loss_criterion}_loss'][0].append(
                locator_train_error)
            locator_training_error[f'{args.loss_criterion}_loss'][1].append(
                epoch)

            logger.info(
                f'''Locator training {args.loss_criterion} error for epoch {epoch} / {max_epochs}:  {locator_train_error}''')
            writer.add_scalar(
                f"train/locator_{args.model}_train_error",
                locator_train_error,
                epoch)

            if epoch % args.validate_after_epochs == 0:
                net_locator.eval()
                logger.info(
                    f"""-------Performing Validation for locator--------""")

                with torch.no_grad():
                    sigmoid = nn.Sigmoid()
                    for iteration, val_data in enumerate(
                            locator_validation_set_loader):
                        val_raw_image, val_label = val_data
                        val_label = val_label.to(device)
                        val_raw_image = val_raw_image.to(device)

                        val_output = net_locator(val_raw_image)

                        locator_loss = locator_loss_function(
                            val_output, val_label)

                        pred = sigmoid(val_output)
                        pred[pred < 1] = 0
                        pred = pred[0, 0].to(
                            dtype=torch.uint8).detach().cpu().numpy()

                        iou, p, r, f1, dice = SegmentationMetrics(
                            pred, val_label.cpu().numpy()).metrics()

                        iou_score.append(iou)
                        precision.append(p)
                        recall.append(r)
                        f1_score.append(f1)
                        dice_score.append(dice)

                        locator_validation_score[f'{args.loss_criterion}_loss'] = torch.cat(
                            (locator_validation_score[f'{args.loss_criterion}_loss'],
                             locator_loss.detach().view(1))) if iteration > 0 else locator_loss.detach().view(1)

                locator_valid_error = torch.mean(
                    locator_validation_score[f'{args.loss_criterion}_loss'])
                locator_validation_error[f'{args.loss_criterion}_loss'][0].append(
                    locator_valid_error)
                locator_validation_error[f'{args.loss_criterion}_loss'][1].append(
                    epoch)

                logger.info(
                    f"""Validation {args.loss_criterion} for Epoch {epoch}:  {locator_valid_error}
                            """)
                writer.add_scalar(
                    f"valid/locator_{args.model}_valid_{args.loss_criterion}_error",
                    locator_valid_error,
                    epoch)
                writer.add_scalar(
                    f"iou/locator_{args.model}_iou",
                    np.mean(iou_score),
                    epoch)
                writer.add_scalar(
                    f"precision/locator_{args.model}_precision",
                    np.mean(precision),
                    epoch)
                writer.add_scalar(
                    f"recall/locator_{args.model}_recall",
                    np.mean(recall),
                    epoch)
                writer.add_scalar(
                    f"f1/locator_{args.model}_f1_score",
                    np.mean(f1_score),
                    epoch)
                writer.add_scalar(
                    f"dice/locator_{args.model}_dice_score",
                    np.mean(dice_score),
                    epoch)

                logger.info(f"""------------------------
                                             | Mean IoU       : {np.mean(iou_score, axis = 0):.2f} |
                                             | Mean Precision : {np.mean(precision, axis = 0):.2f} |
                                             | Mean Recall    : {np.mean(recall, axis = 0):.2f} |
                                             | Mean Dice score: {np.mean(dice_score, axis = 0):.2f} |
                                             ------------------------

                                    """)

                locator_scheduler.step(locator_valid_error)

                iou_score.clear()
                precision.clear()
                recall.clear()
                f1_score.clear()
                dice_score.clear()

            best_locator_model = True if best_locator_validation_score < (
                1 - locator_valid_error) else False

            if best_locator_model:
                best_locator_validation_score = (1 - locator_valid_error)

            if best_locator_model or epoch % args.save_after_epochs == 0:
                model_state = net_locator.state_dict()

                locator_state = {
                    'epoch': epoch,
                    'max_epochs': max_epochs,
                    'locator_optimizer_state': locator_optimizer.state_dict(),
                    'locator_model_state': model_state,
                    'best_validation_score': best_locator_validation_score,
                    'locator_training_error': locator_training_error,
                    'locator_validation_error': locator_validation_error,
                    'scale_factor': args.scale_factor}

                os.makedirs("runs/weights/", exist_ok=True)
                checkpoint_path = f"runs/weights/locator_{args.model}.pt"
                torch.save(locator_state, checkpoint_path)

                logger.info(
                    f"""Saving locator model state to '{checkpoint_path}'
                                             Locator Training error: {locator_train_error}
                                             Locator Validation error: {locator_valid_error}
                                             Optimizer Learning Rate: {'{:.10f}'.format(locator_optimizer.param_groups[0]['lr'])}
                                             Is Best Locator model: {best_locator_model}
                                    """)

                if best_locator_model:
                    best_checkpoint_path = f"runs/weights/locator_{args.model}_best.pt"
                    shutil.copyfile(checkpoint_path, best_checkpoint_path)

    logger.info(f"""
                                    _______________________
                                    Finished model training
                                    =======================
                        """)
