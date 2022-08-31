import torch
import torch.nn as nn


class Dice(nn.Module):
    """
    Calculates the dice loss between prediction and ground truth label tensors. Prediction tensor must be normalised using sigmoid function before
    calculation.
    """

    def __init__(self):
        super(Dice, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, predicted_output, label):
        assert predicted_output.size() == label.size(
        ), 'predicted output and label must have the same dimensions'
        predicted_output = self.sigmoid(predicted_output)
        # Resizes or flattens the predicted and label tensors to calculate
        # intersect between them
        predicted_output = predicted_output.view(1, -1)
        label = label.view(1, -1).float()
        intersect = (predicted_output * label).sum(-1)
        denominator = (predicted_output).sum(-1) + (label).sum(-1)
        dice_score = 2 * (intersect / denominator.clamp(min=1e-6))

        return 1.0 - dice_score


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


class FocalLoss(nn.Module):
    """
        Implements calculation of Focal loss as  FocalLoss(pt) = −(1 − pt)γ log(pt)
        specified in "Lin, T. Y. et al. (2020) ‘Focal Loss for Dense Object Detection’, IEEE Transactions on Pattern Analysis and Machine Intelligence, 42(2), pp. 318–327."
        doi: 10.1109/TPAMI.2018.2858826.
    """

    def __init__(self, gamma=2, eps=1e-6, alpha=1.0, **kwargs):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.alpha = alpha
        self.dice = Dice()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, predicted_output, label):
        error = self.dice(predicted_output, label)
        BCE = self.bce(predicted_output, label, reduction='none')
        pt = torch.exp(-BCE)
        #focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE
        focal_loss = (1 + (label * 199)) * (1 - pt) ** self.gamma * BCE

        return error, focal_loss.mean().view(1)


class HybridLoss(nn.Module):
    def __init__(self, loss_type='Dice', alpha=1):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.dice = Dice()
        self.loss_1 = Dice() if loss_type == 'Dice' else FocalLoss()
        self.surface_loss = IoULoss()

    def forward(self, predicted_output, label, distance_map, alpha):
        self.alpha = alpha
        error = self.dice(predicted_output, label)
        self.dsc = self.alpha * self.loss_1(predicted_output, label)
        self.surface = (1 - self.alpha) * \
            self.surface_loss(predicted_output, distance_map)
        return error, self.dsc + self.surface
