# RhythmGaussian: 4D Gaussian Representation for Remote Physiological Measurement

Official implementation of **RhythmGaussian**, which addresses the entanglement between motion / illumination interference and physiological signals in remote photoplethysmography (rPPG) by explicitly decoupling 4D chromatic and geometric components with a Generalizable Gaussian Model (GGM).

> **Highlight.** The 4D Gaussian representation makes it possible to model geometry, motion, and chroma jointly under a single rendering formulation — without requiring real camera intrinsics/extrinsics, which the rPPG datasets do not provide.

---

## Abstract

Remote Photoplethysmography (rPPG) enables non-contact extraction of physiological signals, providing significant advantages in medical monitoring, emotion recognition, and face anti-spoofing. However, the extraction of reliable rPPG signals is hindered by motion variations in real-world environments, leading to an entanglement issue. To address this challenge, we employ the Generalizable Gaussian Model (GGM) to disentangle geometry and chroma components with 4D Gaussian representations. Employing the GGM for robust rPPG estimation is non-trivial. Firstly, there are no camera parameters in the dataset, resulting in the inability to render video from 4D Gaussian. The "4D virtual camera" is proposed to construct extra Gaussian parameters to describe view and motion changes, giving the ability to render video with fixed virtual camera parameters. Further, the chroma component is still not explicitly decoupled in 4D Gaussian representation. Explicit Motion Modeling (EMM) is designed to decouple the motion variation in an unsupervised manner. Explicit Chroma Modeling (ECM) is tailored to decouple specular, physiological, and noise signals respectively.

---

## Method

The pipeline takes an STMap derived from the input face video, encodes it through a shared backbone, and splits into two heads:

1. **Physiological Decoder** — predicts BVP signal and HR.
2. **4D Gaussian Adapter** — predicts a 4D Gaussian map `M_gs ∈ R^{22 × N × T}` consisting of:
   - depth `d_r ∈ R^1`
   - specular reflection `v_s ∈ R^3`
   - diffuse reflection (physiological) `v_d ∈ R^3`
   - motion noise `v_n ∈ R^3`
   - alpha `a ∈ R^1`, scale `s ∈ R^3`, rotation `r ∈ R^3`
   - motion flow `[Δh, Δw, Δs] ∈ R^5`

Final color: `c = v_s + v_d + v_n`. Final position: `p = K E (ud, vd, d)` rendered via the **4D Virtual Camera** (face center as focal point, half face size as focal length, identity extrinsics).

### Three core modules

- **4D Virtual Camera** — fixes the missing camera parameters of rPPG datasets so 4D Gaussian splatting becomes applicable.
- **Explicit Motion Modeling (EMM)** — predicts a per-point motion flow `[Δh, Δw, Δs]` and renders both the original points `p` and motion-corrected points `p̂ = K E ((u+1)d, (v + Δw + Δh·W)d, d)`. The corrected points are constrained to be free of chroma motion noise.
- **Explicit Chroma Modeling (ECM)** — disentangles `v_d`, `v_s`, `v_n` with two unsupervised constraints:
  - `L_st` — ContrastPhys-style spatio-temporal consistency on `v_d`, ensuring physiological signals are periodic and similar across face regions.
  - `L_m` — negative Pearson between the modulus of motion flow and the modulus of `v_n`, tying chroma motion noise to the predicted motion flow.

### Total objective

```
L_all = λ_rec · L_rec  +  λ_st · L_st  +  λ_m · L_m  +  L_physio
```

`L_physio` follows the supervision protocol of [NEST-rPPG](https://github.com/EnVision-Research/NEST-rPPG); the three remaining terms are unsupervised, so RhythmGaussian also operates under the unsupervised protocol when `L_physio` is removed.

---

## Datasets

We evaluate on the cross-domain rPPG benchmark from NEST-rPPG (VIPL, V4V, BUAA, UBFC, PURE) and additionally on **MMPD**, **MR-NIRP**, **VV100**, **UCLA-rPPG**, and **Phys** to cover diverse motion / illumination interferences. Pre-processed STMaps and labels follow the NEST-rPPG convention; see the [NEST-rPPG repo](https://github.com/EnVision-Research/NEST-rPPG) for download and pre-processing details.

After download, organise the data root as:

```
$ROOT/
  VIPL/<subject>/STMap/STMap_RGB_Align_CSI.png
  V4V/<subject>/STMap/STMap_RGB.png
  PURE/...           STMap/STMap.png
  BUAA/...           STMap/STMap_RGB.png
  UBFC/...           STMap/STMap.png
  MMPD/...           STMap/STMap_RGB.png
  MR-NIRP/...        STMap/STMap_NIR.png
  VV100/...          STMap/STMap_RGB.png
  UCLA-rPPG/...      STMap/STMap_RGB.png
  Phys/...           STMap/STMap_RGB.png
  STMap_Index/       # auto-generated on first run with -rD 1
```

Then update `root_file` in `train.py:110` (or pass via env variable, see below) to point at `$ROOT`.

---

## Installation

```bash
conda create -n rhythmgs python=3.10 -y
conda activate rhythmgs

# PyTorch (CUDA 11.8 example; pick a build matching your GPU/driver)
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118

pip install opencv-python scipy pandas pillow

# Differentiable 3D Gaussian rasterizer (the original Inria implementation)
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git
pip install ./gaussian-splatting/submodules/diff-gaussian-rasterization
```

Tested with PyTorch 2.1 / CUDA 11.8 on a single A100. Any GPU that runs `diff_gaussian_rasterization` should work.

---

## Train

Build the STMap index on first run (`-rD 1`), then drop the flag for subsequent runs:

```bash
# Target = VIPL (others are sources)
python train.py -g 0 -t VIPL -rD 1

# Resume / continue with the index already built
python train.py -g 0 -t VIPL -rD 0
```

Useful flags (see `utils.py` for the full list):

| flag | default | meaning |
|---|---|---|
| `-t`, `--tgt` | `VIPL` | target domain (`VIPL` or `V4V`) |
| `-b`, `--batch-size` | 32 | per-source batch size |
| `-mi`, `--max_iter` | 20000 | total iterations |
| `--lambda_rec` | 0.1 | weight of L_rec |
| `--lambda_st` | 1.0 | weight of L_st |
| `--lambda_m` | 1.0 | weight of L_m |
| `-sr`, `--spatial_aug_rate` | 0.5 | spatial-shuffle augmentation rate |
| `-tr`, `--temporal_aug_rate` | 0.1 | temporal-shift augmentation rate |
| `-k1` … `-k10` | various | per-dataset balance weights for L_physio |

Logs are written to `./Result_log/`, predictions to `./Result/`, models to `./Result_Model/`.

### Unsupervised protocol

Set `--lambda_rec --lambda_st --lambda_m` only and zero out the L_physio weights (`-k1 ... -k10` to 0); the model then trains entirely on the three unsupervised constraints.

---

## Evaluation

For VIPL / V4V (HR head):

```bash
python Eval.py
```

For BUAA / PURE / UBFC (BVP head): clip-level BVP `.mat` files are saved during training; downstream HR / HRV evaluation uses the same Matlab pipeline as NEST-rPPG.

---

## Repo Layout

```
4D-rPPG/
├── train.py            # multi-source DG training loop (target = leave-one-out)
├── model.py            # BaseNet: ResNet18 encoder + physio head + 4D GS adapter
├── gs.py               # GaussianRenderer wrapping diff_gaussian_rasterization
├── graphics_utils.py   # 4D virtual camera (Inria gaussian-splatting helpers)
├── MyDataset.py        # STMap dataset with spatial/temporal augmentation
├── MyLoss.py           # P_loss3, SP_loss, ST_loss, M_loss, get_loss
├── utils.py            # CLI args, Logger, MyEval (ME/STD/MAE/RMSE/MER/Pearson)
├── Eval.py             # per-video aggregation + final HR metrics
└── run.sh              # example launcher
```

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{rhythmgaussian,
  title     = {RhythmGaussian: 4D Gaussian Representation for Remote Physiological Measurement},
  author    = {Lu, Hao and others},
  booktitle = {(to appear)},
  year      = {2025}
}
```

We also build on prior cross-domain rPPG work; please cite NEST-rPPG and DOHA if you use the benchmark protocol:

```bibtex
@inproceedings{Lu_2023_CVPR,
  author    = {Lu, Hao and Yu, Zitong and Niu, Xuesong and Chen, Ying-Cong},
  title     = {Neuron Structure Modeling for Generalizable Remote Physiological Measurement},
  booktitle = {CVPR},
  year      = {2023}
}

@inproceedings{Sun_2023_DOHA,
  author    = {Sun, Weiyu and Zhang, Xinyu and Lu, Hao and Chen, Ying and Ge, Yun and Huang, Xiaolin and Yuan, Jie and Chen, Yingcong},
  title     = {Resolve Domain Conflicts for Generalizable Remote Physiological Measurement},
  booktitle = {ACM MM},
  year      = {2023}
}
```

---

## Acknowledgements

- The cross-domain rPPG benchmark, STMap pre-processing, and `L_physio` formulation follow [NEST-rPPG](https://github.com/EnVision-Research/NEST-rPPG).
- The differentiable Gaussian rasterizer is the [Inria 3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) implementation. `graphics_utils.py` is adapted from the same project.
- The activation choices for RGB / alpha / scale / rotation follow [LGM (Tang et al., 2024)](https://github.com/3DTopia/LGM).
- The `L_st` formulation borrows from [ContrastPhys](https://github.com/zhaodongsun/contrast-phys).

---

## Contact

Issues and pull requests are welcome.
