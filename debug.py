import cv2
import numpy as np
import matplotlib.pyplot as plt
from auxilary.utils import *

results = readMetrics("Outputs/experiment_09-02_09.56.56_MBP_dropAfter/")
print(results)

cm = results["confusionMatrix"]
dice = calc_dice_score(cm)

print(results["bestValAccuracy"])
print(dice)


