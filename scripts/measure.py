from sewar.full_ref import scc
from sewar.full_ref import psnrb
import sewar

import os
import cv2
import numpy as np
gt_dir = '/home/godgang/NS-Outpainting/logs/0817/2/origval'
result_dir = '/home/godgang/NS-Outpainting/logs/0817/2/result_structureflow'
gt_folder = os.listdir(gt_dir)
result_folder = os.listdir(result_dir)
scc_result = []

for files in gt_folder:
    gt = os.path.join(gt_dir,files)
    result = os.path.join(result_dir,files)
    gt = cv2.imread(gt,cv2.IMREAD_GRAYSCALE)
    result = cv2.imread(result,cv2.IMREAD_GRAYSCALE)
    scc_result.append(psnrb(gt,result))
    # print(scc(gt,result))
    print(np.mean(scc_result))

