import os
import shutil
import time
import argparse

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from lib.data import DatasetGenerator, LocatorDataGenerator
from lib.utils import *
from lib.losses import *
from lib.models.vnet_attention import VNetAttention
from lib.metrics import SegmentationMetrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_dir',
        dest='root_dir',
        default='~/Drive/lasc18_dataset',
        type=str,
        help='path to the LASC18 dataset')
    parser.add_argument(
        '--max_epochs',
        dest='max_epochs',
        default=201,
        type=int,
        help='number of epochs')
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        default=3,
        type=int,
        help='batch size for the segmentation model')
    parser.add_argument(
        '--locator_batch_size',
        dest='locator_batch_size',
        default=4,
        type=int,
        help='batch size for the locator model')
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "unet3d",
            "vnet",
            "vnetlight",
            "vnetattn"],
        default="vnetattn")
    parser.add_argument(
        '--scale_factor',
        dest='scale_factor',
        default=(
            0.5,
            0.25,
            0.25),
        type=tuple,
        help='scale down factor(D,H,W) for locator model training')
    parser.add_argument(
        '--padding',
        dest='padding',
        default=(
            35,
            35,
            35),
        type=tuple,
        help='padding along each axis for segmentation model inputs')
    parser.add_argument(
        '--locator_learning_rate',
        dest='locator_lr',
        default=0.0001,
        type=float,
        help='optimizer learning rate for the locator model')
    parser.add_argument(
        '--learning_rate',
        dest='lr',
        default=0.0001,
        type=float,
        help='optimizer learning rate for the segmentation model')
    parser.add_argument(
        '--loss_criterion',
        dest='loss_criterion',
        default='dice',
        type=str,
        help='loss function to be used for the segmentation model')
    parser.add_argument(
        '--save_after_epochs',
        dest='save_after_epochs',
        default=100,
        type=int,
        help='number of epochs after which state is saved by default')
    parser.add_argument(
        '--validate_after_epochs',
        dest='validate_after_epochs',
        default=1,
        type=int,
        help='number of epochs after which validation occurs')
    parser.add_argument(
        '--resume',
        dest='resume',
        default=None,
        type=str,
        help='file path of the stored checkpoint state to resume segmentation training')
    parser.add_argument(
        '--locator_resume',
        dest='locator_resume',
        default=None,
        type=str,
        help='file path of the stored locator checkpoint state to resume locator training')
    parser.add_argument(
        '--best_locator',
        dest='best_locator',
        default='runs/weights/locator_vnetattn.pt',
        type=str,
        help='file path of the best locator checkpoint state to use before segmentation')
    parser.add_argument(
        '--seed',
        dest='seed',
        default=42,
        type=int,
        help='seed for RNG')
    parser.add_argument(
        '--gpu',
        action='store_true',
        dest='gpu',
        default=True,
        help='use cuda')
    parser.add_argument(
        '--patience',
        dest='patience',
        default=7,
        type=int,
        help='LR Scheduler patience')
    parser.add_argument(
        '--reduce',
        dest='reduce',
        default=0.8,
        type=float,
        help='LRScheduler learning_rate reduction factor ')

    args = parser.parse_args()

    # set RNG for both CPU and CUDA
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    root_dir = args.root_dir
    max_epochs = args.max_epochs
    start_epoch = 0
    locator_batch_size = args.locator_batch_size
    lr_reduce_factor = args.reduce
    scheduler_patience = args.patience
    locator_lr_reduce_factor = 0.85
    locator_scheduler_patience = 15
    locator_loss_function = Dice()
    locator_training_error = {f'{args.loss_criterion}_loss': ([], [])}
    locator_validation_error = {f'{args.loss_criterion}_loss': ([], [])}

    logger = get_logger(f'{args.model}_{args.loss_criterion}')
    writer = SummaryWriter(log_dir="runs")

    best_locator_validation_score = 0
    use_cuda = torch.cuda.is_available() and args.gpu
    device = torch.device("cuda" if use_cuda else "cpu")

    net_locator = VNetAttention(in_channels=1, classes=1)
    logger.info(f'Initialised locator model.')

    locator_optimizer = optim.Adam(
        net_locator.parameters(),
        lr=args.locator_lr)

    locator_scheduler = lr_scheduler.ReduceLROnPlateau(
        locator_optimizer,
        mode='min',
        factor=locator_lr_reduce_factor,
        patience=locator_scheduler_patience,
        verbose=True)

    if args.best_locator is not None:
        assert os.path.isfile(
            args.best_locator), "Best locator load path provided doesn't exist!"
        best_checkpoint = torch.load(args.best_locator, map_location='cpu')

        net_locator.load_state_dict(best_checkpoint['locator_model_state'])

        logger.info(f'Loaded best locator model weights.')

    elif args.best_locator is None and args.locator_resume is not None:
        assert os.path.isfile(
            args.locator_resume), "Locator resume file path provided doesn't exist!"
        checkpoint = torch.load(args.locator_resume, map_location='cpu')

        logger.info(
            f"Loading checkpoint locator model state from resume path: '{args.locator_resume}'")

        start_epoch = int(checkpoint['epoch']) + 1
        max_epochs = int(checkpoint['max_epochs'])
        locator_optimizer.load_state_dict(
            checkpoint['locator_optimizer_state'])
        net_locator.load_state_dict(checkpoint['locator_model_state'])
        best_locator_validation_score = float(
            checkpoint['best_locator_validation_score'])
        locator_training_error = checkpoint['locator_training_error']
        locator_validation_error = checkpoint['locator_validation_error']

    elif args.best_locator is None and args.locator_resume is None:
        logger.info(f'Initialised model weights.')

    logger.info(
        f'Number of Trainable parameters for locator model: {number_of_parameters(net_locator)}')

    if use_cuda:
        net_locator.to(device)
        logger.info(
            f'Training locator with {device} using {torch.cuda.device_count()} GPUs.')

    start_epoch = 0
    batch_size = args.batch_size
    training_error = {f'{args.loss_criterion}_loss': ([], [])}
    validation_error = {f'{args.loss_criterion}_loss': ([], [])}

    best_validation_score = 0

    if args.loss_criterion == "dice":
        loss_function = Dice()
    else:
        loss_function = IoULoss()

    net = VNetAttention(in_channels=1, classes=1)
    logger.info(f'Initialised segmentation model.')

    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=lr_reduce_factor,
        patience=scheduler_patience,
        verbose=True)

    if args.resume is not None:
        assert os.path.isfile(
            args.resume), "Resume file name provided for segmentation model doesn't exist!"
        checkpoint = torch.load(args.resume, map_location='cpu')

        start_epoch = int(checkpoint['epoch']) + 1
        max_epochs = int(checkpoint['max_epochs'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        net.load_state_dict(checkpoint['model_state'])
        best_validation_score = float(checkpoint['best_validation_score'])
        training_error = checkpoint['training_error']
        validation_error = checkpoint['validation_error']

        logger.info(
            f"Checkpoint segmentation model state loaded from resume path: '~/{args.resume}'")
    else:
        logger.info(f'Initialised segmentation model weights.')

    logger.info(
        f'Number of Trainable parameters for segmentation model: {number_of_parameters(net)}')

    if use_cuda:
        net.to(device)
        logger.info(
            f'Training with {device} using {torch.cuda.device_count()} GPUs.')

    train_set_builder = LocatorDataGenerator(
        root_dir=root_dir,
        mode='paths_train',
        scale_factor=args.scale_factor)
    validation_set_builder = LocatorDataGenerator(
        root_dir=root_dir,
        mode='paths_validate',
        scale_factor=args.scale_factor)
    train_builder_inputs = []
    validation_builder_inputs = []

    net_locator.eval()
    with torch.no_grad():
        for idx in range(len(train_set_builder)):
            train_raw_data, train_label_data = train_set_builder[idx]
            raw_file_name, raw_image = train_raw_data
            raw_image = torch.unsqueeze(raw_image, dim=0).to(device)

            train_output = net_locator(raw_image)

            train_builder_inputs.append(
                ((raw_file_name, train_output), (train_label_data)))

        for idx in range(len(validation_set_builder)):
            validate_raw_data, validate_label_data = validation_set_builder[idx]
            raw_file_name, raw_image = validate_raw_data
            raw_image = torch.unsqueeze(raw_image, dim=0).to(device)

            validate_output = net_locator(raw_image)

            validation_builder_inputs.append(
                ((raw_file_name, validate_output), (validate_label_data)))

    train_set1 = DatasetGenerator(
        mode='train',
        inputs=train_builder_inputs,
        pad=args.padding,
        scale_factor=args.scale_factor,
        loss_criterion=args.loss_criterion)

    dataset_mean = train_set1.mean
    dataset_std = train_set1.std
    patch_size = train_set1.patch_size

    # Resuming training from checkpoint will affect reproducibility for random
    # rotate and translate sets
    train_set_rotate = DatasetGenerator(
        mode='train',
        inputs=train_builder_inputs,
        pad=args.padding,
        scale_factor=args.scale_factor,
        loss_criterion=args.loss_criterion,
        seed=args.seed,
        transform='random_rotate',
        mean=dataset_mean,
        std=dataset_std,
        patch_dims=patch_size)

    train_set_translate = DatasetGenerator(
        mode='train',
        inputs=train_builder_inputs,
        pad=args.padding,
        scale_factor=args.scale_factor,
        loss_criterion=args.loss_criterion,
        seed=args.seed,
        transform='random_translate',
        mean=dataset_mean,
        std=dataset_std,
        patch_dims=patch_size)

    train_set_translate2 = DatasetGenerator(
        mode='train',
        inputs=train_builder_inputs,
        pad=args.padding,
        scale_factor=args.scale_factor,
        loss_criterion=args.loss_criterion,
        seed=args.seed * 2,
        transform='random_translate',
        mean=dataset_mean,
        std=dataset_std,
        patch_dims=patch_size)

    train_set = ConcatDataset(
        [train_set1, train_set_rotate, train_set_translate, train_set_translate2])

    logger.info(f"""Created segmentation train dataset generator objects.
                                        Dataset mean: {dataset_mean}
                                        Dataset std: {dataset_std}
                                        Patch size: {patch_size}""")

    validation_set = DatasetGenerator(
        mode='validate',
        inputs=validation_builder_inputs,
        pad=args.padding,
        scale_factor=args.scale_factor,
        loss_criterion=args.loss_criterion,
        mean=dataset_mean,
        std=dataset_std,
        patch_dims=patch_size)

    logger.info(f'Created segmentation validation dataset generator object.')

    train_set_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16)

    validation_set_loader = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16)

    logger.info(
        f"""Length of train set: {len(train_set)} and validation set: {len(validation_set)}
                                      Batch Size: {batch_size}
                                      -----------------------------------------------------------------------------
                                      Beginning model training from epoch: {start_epoch} / {max_epochs}
                                      Best validation score: {best_validation_score}
                                      Adam optimiser with lr: {'{:.7f}'.format(optimizer.param_groups[0]['lr'])}
                                      Scheduler ReduceLROnPlateau with mode: 'min', factor: {lr_reduce_factor}, patience: {scheduler_patience}
                                      Loss Criterion: '{args.loss_criterion}'
                                      -----------------------------------------------------------------------------
                        """)

    for epoch in range(start_epoch, max_epochs):
        net.train()
        a = {
            'surface_avg': torch.zeros(
                1,
                requires_grad=False,
                dtype=torch.float32)}
        training_score = {
            f'{args.loss_criterion}_loss': torch.zeros(
                1, requires_grad=False, dtype=torch.float32)}
        validation_score = {
            f'{args.loss_criterion}_loss': torch.zeros(
                1, requires_grad=False, dtype=torch.float32)}

        iou_score = []
        precision = []
        recall = []
        f1_score = []
        dice_score = []

        logger.info(
            f"""-------Segmentation training Epoch: [{epoch} / {max_epochs}]-------""")

        for iteration, data in enumerate(train_set_loader):
            raw_image_patches, label_patches, dist_maps = data
            raw_image_patches = raw_image_patches.to(device)
            label_patches = label_patches.to(device)

            output_patches = net(raw_image_patches)

            loss = loss_function(output_patches, label_patches)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar(
                f"lr/{args.model}_learning_rate",
                optimizer.param_groups[0]['lr'],
                epoch)

            training_score[f'{args.loss_criterion}_loss'] = torch.cat(
                (training_score[f'{args.loss_criterion}_loss'],
                 loss.detach())) if iteration > 0 else loss.detach()

        train_error = torch.mean(training_score[f'{args.loss_criterion}_loss'])

        training_error[f'{args.loss_criterion}_loss'][0].append(train_error)

        for loss_name in training_error.keys():
            training_error[loss_name][1].append(epoch)

        logger.info(
            f"""Training {args.loss_criterion} error for epoch {epoch} / {max_epochs}:  {train_error}""")
        writer.add_scalar(
            f"train/{args.model}_train_error",
            train_error,
            epoch)

        net.eval()
        logger.info(f"""-------Performing Validation--------""")

        with torch.no_grad():
            for iteration, val_data in enumerate(validation_set_loader):
                val_raw_image_patches, val_label_patches, val_dist_maps = val_data
                val_label_patches = val_label_patches.to(device)
                val_raw_image_patches = val_raw_image_patches.to(device)

                val_output_patches = net(val_raw_image_patches)

                loss = loss_function(val_output_patches, val_label_patches)

                pred = torch.sigmoid(val_output_patches)
                #print(pred.shape, val_output_patches.shape)
                pred[pred < 1] = 0
                #print(pred.shape, val_output_patches.shape)
                pred = pred.to(dtype=torch.uint8).detach().cpu().numpy()

                #print(pred.shape, val_label_patches.shape, "\n")

                iou, p, r, f1, dice = SegmentationMetrics(
                    pred, val_label_patches.cpu().numpy()).metrics()

                iou_score.append(iou)
                precision.append(p)
                recall.append(r)
                f1_score.append(f1)
                dice_score.append(dice)

                validation_score[f'{args.loss_criterion}_loss'] = torch.cat(
                    (validation_score[f'{args.loss_criterion}_loss'],
                     loss.detach())) if iteration > 0 else loss.detach()

            valid_error = torch.mean(
                validation_score[f'{args.loss_criterion}_loss'])

            logger.info(
                f'''Validation {args.loss_criterion} epoch {epoch}:  {valid_error}
                            ''')
            writer.add_scalar(
                f"valid/{args.model}_train_error",
                valid_error,
                epoch)
            writer.add_scalar(
                f"iou/{args.model}_iou",
                np.mean(iou_score),
                epoch)
            writer.add_scalar(
                f"precision/{args.model}_precision",
                np.mean(precision),
                epoch)
            writer.add_scalar(
                f"recall/{args.model}_recall",
                np.mean(recall),
                epoch)
            writer.add_scalar(
                f"f1/{args.model}_f1_score",
                np.mean(f1_score),
                epoch)
            writer.add_scalar(
                f"dice/{args.model}_dice_score",
                np.mean(dice_score),
                epoch)

            logger.info(f"""------------------------
                                             | Mean IoU       : {np.mean(iou_score, axis = 0):.2f} |
                                             | Mean Precision : {np.mean(precision, axis = 0):.2f} |
                                             | Mean Recall    : {np.mean(recall, axis = 0):.2f} |
                                             | Mean F1 score  : {np.mean(f1_score, axis = 0):.2f} |
                                             | Mean Dice score: {np.mean(dice_score, axis = 0):.2f} |
                                             ------------------------

                                    """)

            scheduler.step(valid_error)

            validation_error[f'{args.loss_criterion}_loss'][0].append(
                valid_error)

        best_model = True if best_validation_score < (
            1 - valid_error) else False

        if best_model:
            best_validation_score = (1 - valid_error)

        if best_model or epoch % args.save_after_epochs == 0:

            if isinstance(net, nn.DataParallel):
                model_state = net.module.state_dict()
            else:
                model_state = net.state_dict()

            state = {'epoch': epoch,
                     'max_epochs': max_epochs,
                     'optimizer_state': optimizer.state_dict(),
                     'model_state': model_state,
                     'best_validation_score': best_validation_score,
                     'training_error': training_error,
                     'validation_error': valid_error,
                     'patch_size': patch_size,
                     'scale_factor': args.scale_factor,
                     'mean': dataset_mean,
                     'std': dataset_std
                     }

            assert os.path.exists(
                "./runs/weights"), "Runs path does not exist! Please run locate.py first"
            checkpoint_path = f'runs/weights/{args.model}_latest.pt'
            torch.save(state, checkpoint_path)

            logger.info(f'''Saving model state to '{checkpoint_path}'
                                    Training error: {train_error}
                                    Validation error: {valid_error}
                                    Optimizer Learning Rate: {'{:.10f}'.format(optimizer.param_groups[0]['lr'])}
                                    Is Best model: {best_model}
                            ''')

            if best_model:
                best_checkpoint_path = f'runs/weights/{args.model}_best.pt'
                shutil.copyfile(checkpoint_path, best_checkpoint_path)

    logger.info(f"""
                                      ____________________________________
                                      Finished segmentation model training
                                      ====================================
                         """)
