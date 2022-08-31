import os
import cv2
import argparse
import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw",
        type=str,
        required=True,
        help="Path to raw MRI")
    parser.add_argument("--gt", type=str, required=True,
                        help="Path to Ground truth mask")
    parser.add_argument(
        "--pred",
        type=str,
        required=True,
        help="Path to Predicted mask")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Transparency value for the mask")
    parser.add_argument(
        "--out",
        type=str,
        default="prediction.gif",
        help="Name of the output file. Please use extension '.gif'")

    args = parser.parse_args()

    assert os.path.splitext(args.out)[-1] == ".gif"

    raw_array = sitk.GetArrayFromImage(sitk.ReadImage(args.raw))
    gt_array = sitk.GetArrayFromImage(sitk.ReadImage(args.gt))
    pred_array = sitk.GetArrayFromImage(sitk.ReadImage(args.pred))

    pred_array = ((pred_array - pred_array.min()) * (1 /
                  (pred_array.max() - pred_array.min()) * 255)).astype('uint8')

    assert raw_array.shape == gt_array.shape == pred_array.shape, \
        f"""
     Raw file, Ground truth mask and prediction mask should have the same shape.
     Got:-
     Raw shape               : {raw_array.shape}
     Ground truth mask shape : {gt_array.shape}
     Prediction mask shape   : {pred_array.shape}
     """

    canvas = []

    for i, (raw_slice, gt_slice, pred_slice) in enumerate(
            zip(raw_array, gt_array, pred_array)):
        raw_slice = cv2.cvtColor(raw_slice, cv2.COLOR_GRAY2RGB)

        gt_slice = cv2.cvtColor(gt_slice, cv2.COLOR_GRAY2RGB)
        gt_slice[(gt_slice == 255).all(-1)] = [255, 0, 0]

        pred_slice = cv2.cvtColor(pred_slice, cv2.COLOR_GRAY2RGB)
        pred_slice[(pred_slice == 255).all(-1)] = [255, 0, 0]

        gt_blended = cv2.addWeighted(
            raw_slice, args.alpha, gt_slice, float(
                1 - args.alpha), 0)
        pr_blended = cv2.addWeighted(
            raw_slice, args.alpha, pred_slice, float(
                1 - args.alpha), 0)

        final = np.concatenate((gt_blended, pr_blended), axis=1)
        final = Image.fromarray(final)

        gt_text = ImageDraw.Draw(final)
        gt_text.text((10, 10), "Ground Truth", size=50)
        del gt_text

        pr_text = ImageDraw.Draw(final)
        pr_text.text((gt_blended.shape[0] + 10, 10), "Prediction")
        del pr_text

        canvas.append(final)

    canvas[0].save(args.out, save_all=True, append_images=canvas[1:], loop=0)
