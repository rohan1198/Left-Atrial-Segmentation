import numpy
from numpy import ndarray


def iou_score(predicted_result: ndarray, reference_ground: ndarray):
    """
        The Jaccard coefficient between the object(s) in `result` and the object(s) in `reference`. It ranges from 0 (no overlap) to 1 (perfect overlap).
    """
    result = predicted_result.astype(bool)
    reference = reference_ground.astype(bool)

    intersection = numpy.count_nonzero(result & reference)
    union = numpy.count_nonzero(result | reference)

    jaccard_similarity = float(intersection) / float(union)

    return jaccard_similarity


def precision(predicted_result: ndarray, reference_ground: ndarray):
    """
        The precision between two binary datasets, here mostly binary objects in images,
        which is defined as the fraction of retrieved instances that are relevant. The
        precision is not symmetric.
    """
    result = predicted_result.astype(bool)
    reference = reference_ground.astype(bool)

    tp = numpy.count_nonzero(result & reference)
    fp = numpy.count_nonzero(result & ~reference)

    try:
        precision = tp / float(tp + fp)
    except ZeroDivisionError:
        precision = 0.0

    return precision


def recall(predicted_result: ndarray, reference_ground: ndarray):
    """
        The recall defined as the fraction of relevant instances between two binary datasets
        across a batch of predicted and reference images.
    """
    result = predicted_result.astype(bool)
    reference = reference_ground.astype(bool)

    tp = numpy.count_nonzero(result & reference)
    fn = numpy.count_nonzero(~result & reference)

    try:
        recall = tp / float(tp + fn)
    except ZeroDivisionError:
        recall = 0.0

    return recall


def f1_score(predicted_result: ndarray, reference_ground: ndarray):
    p = precision(predicted_result, reference_ground)
    r = recall(predicted_result, reference_ground)

    try:
        f1_score = float((p * r) / (p + r))
    except BaseException:
        f1_score = 0.0

    return f1_score


def dice_score(predicted_result: ndarray, reference_ground: ndarray):
    intersection = numpy.sum(predicted_result[reference_ground == 255]) * 2.0
    dice = intersection / (numpy.sum(predicted_result) +
                           numpy.sum(reference_ground))
    return dice


class SegmentationMetrics(object):
    def __init__(self, pred: ndarray, gt: ndarray):
        self.pred = pred
        self.gt = gt

    def metrics(self):
        iou = iou_score(self.pred, self.gt)
        p = precision(self.pred, self.gt)
        r = recall(self.pred, self.gt)
        f1 = f1_score(self.pred, self.gt)
        d = dice_score(self.pred, self.gt)

        return iou, p, r, f1, d
