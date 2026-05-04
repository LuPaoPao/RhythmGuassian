"""Loss functions for RhythmGaussian.

Active losses used in train.py:
- ``P_loss3``  — negative Pearson loss on BVP signals (also used inside ``M_loss``-like helpers).
- ``SP_loss``  — spectral / heart-rate loss using a windowed FFT magnitude.
- ``ST_loss``  — ContrastPhys-style spatio-temporal consistency on the diffuse component ``v_d``.
- ``M_loss``   — negative Pearson between the modulus of the predicted motion flow
                 and the modulus of the chroma motion noise ``v_n`` (Eq. ``L_m`` in the paper).
- ``get_loss`` — per-dataset weighting wrapper for the supervised physiological loss ``L_physio``.
"""
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class P_loss3(nn.Module):
    """Negative Pearson loss along the temporal dimension.

    Inputs are expected to be 3-D tensors ``(M, N, T)`` — typically
    ``M`` = batch, ``N`` = number of channels (often 1), ``T`` = temporal length.
    Returns ``1 - mean(Pearson)`` so that minimising the loss maximises correlation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, gt_lable, pre_lable):
        M, N, A = gt_lable.shape
        gt_lable = gt_lable - torch.mean(gt_lable, dim=-1).view(M, N, 1)
        pre_lable = pre_lable - torch.mean(pre_lable, dim=-1).view(M, N, 1)
        aPow = torch.sqrt(torch.sum(torch.mul(gt_lable, gt_lable), dim=2))
        bPow = torch.sqrt(torch.sum(torch.mul(pre_lable, pre_lable), dim=2))
        pearson = torch.sum(torch.mul(gt_lable, pre_lable), dim=2) / (aPow * bPow + 0.001)
        loss = 1 - torch.sum(torch.sum(pearson, dim=1), dim=0) / (gt_lable.shape[0] * gt_lable.shape[1])
        return loss


class SP_loss(nn.Module):
    """Spectral loss for heart-rate estimation.

    Computes the magnitude spectrum of a Hanning-windowed BVP signal at the
    candidate BPM bins ``[low_bound, high_bound)``. The classification loss
    pushes the bin matching the ground-truth HR to be the dominant peak.
    """

    def __init__(self, device, clip_length=256, delta=3, loss_type=1, use_wave=False):
        super().__init__()

        self.clip_length = clip_length
        self.time_length = clip_length
        self.device = device
        self.delta = delta
        self.delta_distribution = [0.4, 0.25, 0.05]
        self.low_bound = 40
        self.high_bound = 150

        self.bpm_range = torch.arange(self.low_bound, self.high_bound, dtype=torch.float).to(self.device)
        self.bpm_range = self.bpm_range / 60.0

        self.pi = 3.14159265
        two_pi_n = Variable(2 * self.pi * torch.arange(0, self.time_length, dtype=torch.float))
        hanning = Variable(
            torch.from_numpy(np.hanning(self.time_length)).type(torch.FloatTensor),
            requires_grad=True,
        ).view(1, -1)

        self.two_pi_n = two_pi_n.to(self.device)
        self.hanning = hanning.to(self.device)

        self.cross_entropy = nn.CrossEntropyLoss()
        self.nll = nn.NLLLoss()
        self.l1 = nn.L1Loss()

        self.loss_type = loss_type
        self.eps = 0.0001

        self.lambda_l1 = 0.1
        self.use_wave = use_wave

    def forward(self, wave, gt, pred=None, flag=None):
        fps = 30

        hr = gt.clone()
        hr[hr.ge(self.high_bound)] = self.high_bound - 1
        hr[hr.le(self.low_bound)] = self.low_bound

        if pred is not None:
            pred = torch.mul(pred, fps)
            pred = pred * 60 / self.clip_length

        batch_size = wave.shape[0]

        f_t = self.bpm_range / fps
        preds = wave * self.hanning

        preds = preds.view(batch_size, 1, -1)
        f_t = f_t.repeat(batch_size, 1).view(batch_size, -1, 1)

        tmp = self.two_pi_n.repeat(batch_size, 1)
        tmp = tmp.view(batch_size, 1, -1)

        complex_absolute = (
            torch.sum(preds * torch.sin(f_t * tmp), dim=-1) ** 2
            + torch.sum(preds * torch.cos(f_t * tmp), dim=-1) ** 2
        )

        target = hr - self.low_bound
        target = target.type(torch.long).view(batch_size)

        whole_max_val, whole_max_idx = complex_absolute.max(1)
        whole_max_idx = whole_max_idx + self.low_bound

        if self.loss_type == 1:
            loss = self.cross_entropy(complex_absolute, target)

        elif self.loss_type == 7:
            norm_t = torch.ones(batch_size).to(self.device) / torch.sum(complex_absolute, dim=1)
            norm_t = norm_t.view(-1, 1)
            complex_absolute = complex_absolute * norm_t

            loss = self.cross_entropy(complex_absolute, target)

            idx_l = target - self.delta
            idx_l[idx_l.le(0)] = 0
            idx_r = target + self.delta
            idx_r[idx_r.ge(self.high_bound - self.low_bound - 1)] = self.high_bound - self.low_bound - 1

            loss_snr = 0.0
            for i in range(0, batch_size):
                loss_snr = loss_snr + 1 - torch.sum(complex_absolute[i, idx_l[i]:idx_r[i]])

            loss_snr = loss_snr / batch_size

            loss = loss + loss_snr

        return loss, whole_max_idx


class ST_loss(nn.Module):
    """ContrastPhys-style spatio-temporal consistency on ``v_d``.

    ``v_d`` shape: ``(B, N=64, T=256, 3)``. The ``N`` rows are split into ``K`` regions
    (default 4) and averaged within each region to yield ``K`` candidate physiological
    signals per sample. Pairwise Pearson similarity between regions is maximised so
    that the diffuse component is periodic and consistent across face areas.
    """

    def forward(self, v_d, K=4):
        B, N, T, _ = v_d.shape
        sig = v_d.mean(-1)                              # (B, N, T)
        sig = sig.view(B, K, N // K, T).mean(2)         # (B, K, T)
        sig = sig - sig.mean(-1, keepdim=True)
        sig = sig / (sig.norm(dim=-1, keepdim=True) + 1e-6)
        sim = torch.einsum('bkt,bjt->bkj', sig, sig)    # (B, K, K)
        eye = torch.eye(K, device=sim.device, dtype=torch.bool)
        offdiag = sim.masked_select(~eye).view(B, -1).mean()
        return 1.0 - offdiag


def M_loss(motion_flow, v_n):
    """Negative-Pearson between motion-flow modulus and ``v_n`` modulus over time.

    motion_flow: ``(B, N, T, 5)`` containing ``[Δh, Δw, Δs(3)]``.
    v_n:         ``(B, N, T, 3)`` chroma motion-noise component.
    Both are reduced to a per-time modulus signal by L2 norm over the last dim
    followed by spatial averaging over ``N``. The loss ties the chroma motion
    noise to the predicted motion flow so that they vary together in time.
    """
    m_mod = motion_flow.norm(dim=-1).mean(1)            # (B, T)
    n_mod = v_n.norm(dim=-1).mean(1)                    # (B, T)
    m_mod = m_mod - m_mod.mean(-1, keepdim=True)
    n_mod = n_mod - n_mod.mean(-1, keepdim=True)
    pearson = (m_mod * n_mod).sum(-1) / (
        m_mod.norm(dim=-1) * n_mod.norm(dim=-1) + 1e-6
    )
    return (1.0 - pearson).mean()


def get_loss(bvp_pre, hr_pre, bvp_gt, hr_gt, dataName,
             loss_sig0, loss_hr, args, inter_num):
    """Per-dataset weighted physiological loss ``L_physio``.

    Different rPPG datasets provide different label quality (only HR vs HR+BVP),
    so the weights ``k1..k10`` from CLI scale signal vs HR supervision per dataset.
    The factor ``k`` ramps the HR contribution up smoothly over training.
    """
    k = 2.0 / (1.0 + np.exp(-10.0 * inter_num / args.max_iter)) - 0.999999
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = (
        args.k1, args.k2, args.k3, args.k4,
        args.k5, args.k6, args.k7, args.k8,
        args.k9, args.k10,
    )

    if dataName in ['PURE', 'UBFC', 'BUAA']:
        loss = (loss_sig0(bvp_pre, bvp_gt) + k * 0.1 * loss_hr(torch.squeeze(hr_pre), hr_gt)) / 2
    elif dataName in ['V4V', 'VIPL']:
        loss = 0.1 * k * loss_hr(torch.squeeze(hr_pre), hr_gt)
    elif dataName == 'MR-NIRP':
        loss = k1 * loss_sig0(bvp_pre, bvp_gt) + k * k2 * loss_hr(torch.squeeze(hr_pre), hr_gt)
    elif dataName == 'MMPD':
        loss = k3 * loss_sig0(bvp_pre, bvp_gt) + k * k4 * loss_hr(torch.squeeze(hr_pre), hr_gt)
    elif dataName == 'Phys':
        loss = k5 * loss_sig0(bvp_pre, bvp_gt) + k * k6 * loss_hr(torch.squeeze(hr_pre), hr_gt)
    elif dataName == 'VV100':
        loss = k7 * loss_sig0(bvp_pre, bvp_gt) + k * k8 * loss_hr(torch.squeeze(hr_pre), hr_gt)
    elif dataName == 'UCLA-rPPG':
        loss = k9 * loss_sig0(bvp_pre, bvp_gt) + k * k10 * loss_hr(torch.squeeze(hr_pre), hr_gt)

    if torch.sum(torch.isnan(loss)) > 0:
        loss = 0.0
    return loss
