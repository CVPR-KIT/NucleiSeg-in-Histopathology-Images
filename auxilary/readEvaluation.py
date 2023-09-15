from utils import * 
import argparse


def arg_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expt_dir', type=str, default='none', help='Path to the experiment directory.')
    return parser.parse_args()

if __name__ == "__main__":

    args = arg_init()

    if args.expt_dir == 'none':
        print("Please specify experiment directory")
        sys.exit(1)

    path = args.expt_dir
    results = readMetrics(path)


    cm = results["confusionMatrix"]

    '''TP = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TN = cm[1, 1]

    dice_coefficient = (2 * TP) / (2 * TP + FP + FN)

    iou = TP / (TP + FP + FN)

    # Mean IoU (mIoU) is the same as IoU in a binary classification problem
    mIoU = iou
    '''

    print(f"CM: {cm}")
    #print(f"Dice Coefficient: {dice_coefficient}")
    #print(f"mIoU: {mIoU}")

    print(f"Calculated mIoU : {calc_mIoU(cm)}")
    print(f"Calculated dice : {calc_dice_score(cm)}")