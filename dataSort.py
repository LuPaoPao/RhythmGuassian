"""Sort BVP/HR result mats into per-subject files for downstream evaluation.

Paths are taken from environment variables so the script can be reused
on any machine:
- ``RPPG_INDEX_DIR``  — STMap index directory for the dataset (default: UBFC).
- ``RPPG_RESULT_DIR`` — directory containing ``WAVE_ALL.mat`` and ``WAVE_PR_ALL.mat``.
"""

import cv2
import os
import numpy as np
import shutil
import pandas as pd
import scipy.io as scio
from scipy import interpolate
import scipy.io as io


gt_name = 'BVP.mat'
savePath = r'./Wave_sort/UBFC/'
if not os.path.exists(savePath):
    os.makedirs(savePath)

_default_root = os.environ.get('RPPG_DATA_ROOT', './Data/STMap/')
_default_results = os.environ.get('RPPG_RESULT_DIR', './Result')
Idex_files = os.environ.get(
    'RPPG_INDEX_DIR', os.path.join(_default_root, 'STMap_Index/UBFC')
)
gt_path = os.path.join(_default_results, 'rPPGNet_UBFCSpatial0.5Temporal0.1WAVE_ALL.mat')
pr_path = os.path.join(_default_results, 'rPPGNet_UBFCSpatial0.5Temporal0.1WAVE_PR_ALL.mat')
pr = scio.loadmat(pr_path)['Wave']
pr = np.squeeze(np.array(pr.astype('float32')))
gt = scio.loadmat(gt_path)['Wave']
gt = np.squeeze(np.array(gt.astype('float32')))

files_list = os.listdir(Idex_files)
files_list = sorted(files_list)
temp = scio.loadmat(os.path.join(Idex_files, files_list[0]))
lastPath = str(temp['Path'][0])
pr_temp = []
gt_temp = []
print(pr.shape)
PERSON = 10000
for HR_index in range(pr.shape[0]):
    temp = scio.loadmat(os.path.join(Idex_files, files_list[HR_index]))
    nowPath = str(temp['Path'][0])
    Step_Index = int(temp['Step_Index'])
    if lastPath != nowPath:
        PERSON = PERSON + 1
        if pr_temp is None:
            print(nowPath)
            print(lastPath)
            pr_temp = []
            gt_temp = []
        else:
            print(lastPath)
            print(PERSON)
            io.savemat(savePath + str(PERSON) + 'pr_Wave.mat', {'Wave': pr_temp})
            io.savemat(savePath + str(PERSON) + 'gt_Wave.mat', {'Wave': gt_temp})
            pr_temp = []
            gt_temp = []
            pr_temp.append(pr[HR_index, :])
            gt_temp.append(gt[HR_index, :])
    else:
        pr_temp.append(pr[HR_index, :])
        gt_temp.append(gt[HR_index, :])
    lastPath = nowPath
# io.savemat('gt_ps.mat', {'HR': gt_ps})
# io.savemat('pr_ps.mat', {'HR': pr_ps})
# io.savemat('HR_rel.mat', {'HR': gt_av})
# io.savemat('HR_pr.mat', {'HR': pr_av})
# MyEval(gt_av, pr_av)
