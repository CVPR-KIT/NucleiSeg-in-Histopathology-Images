import cv2
import numpy as np
import matplotlib.pyplot as plt

path = "Dataset/test/0_label.png"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()