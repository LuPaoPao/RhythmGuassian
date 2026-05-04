# -*- coding: UTF-8 -*-
import os
import time
import numpy as np
import scipy.io as io
import torch
import random
from datetime import datetime
from timeit import default_timer as timer
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import MyDataset
from models import MyLoss, model, utils
from models.utils import Logger, time_to_str
from torchvision.utils import save_image
# Define TARGET_DOMAIN and FILE_NAME mappings.
# Datasets supported as targets/sources; UCLA-rPPG is known to be noisy and may
# need extra filtering depending on label quality.
TARGET_DOMAIN = {
    'VIPL': ['V4V', 'PURE', 'BUAA', 'UBFC', 'VV100', 'MR-NIRP', 'UCLA-rPPG', 'Phys', "MMPD"],
    'V4V': ['VIPL', 'PURE', 'BUAA', 'UBFC', 'MR-NIRP', 'VV100', 'UCLA-rPPG', 'Phys', "MMPD"],
}

FILE_NAME = {
    'VIPL': ['VIPL', 'VIPL', 'STMap_RGB_Align_CSI.png'],
    'V4V': ['V4V', 'V4V', 'STMap_RGB.png'],
    'PURE': ['PURE', 'PURE', 'STMap.png'],
    'BUAA': ['BUAA', 'BUAA', 'STMap_RGB.png'],
    'UBFC': ['UBFC', 'UBFC', 'STMap.png'],
    'MMPD': ['MMPD', 'MMPD', 'STMap_RGB.png'],
    'MR-NIRP': ['MR-NIRP', 'MR-NIRP', 'STMap_NIR.png'],
    'VV100': ['VV100', 'VV100', 'STMap_RGB.png'],
    'UCLA-rPPG': ['UCLA-rPPG', 'UCLA-rPPG', 'STMap_RGB.png'],
    'Phys': ['Phys', 'Phys', 'STMap_RGB.png'],
}

def prepare_paths(root_file, domain_names):
    """
    Prepare file paths and related variables for given domain names.
    """
    paths = []
    for name in domain_names:
        file_name = FILE_NAME[name]
        file_root = os.path.join(root_file, file_name[0])
        save_root = os.path.join(root_file, 'STMap_Index', file_name[1])
        map_file = f"{file_name[2]}"
        paths.append({
            'name': name,
            'file_root': file_root,
            'save_root': save_root,
            'map_file': map_file
        })
    return paths

def load_datasets(paths, frames_num, args):
    """
    Load datasets based on the prepared paths.
    """
    datasets = []
    for path in paths:
        dataset = MyDataset.Data_DG(
            root_dir=path['save_root'],
            dataName=path['name'],
            STMap=path['map_file'],
            frames_num=frames_num,
            args=args
        )
        datasets.append(dataset)
    return datasets


def save_images(tensor, output_dir, base_filename="image"):
    """Save a batch of rendered images to ``output_dir`` as PNGs."""
    # ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # move tensor to CPU
    tensor = tensor.detach().cpu()

    # normalise to [0, 1] for visualisation; require float tensor
    if tensor.dtype == torch.float32 or tensor.dtype == torch.float64:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

    # save each image in the batch
    for i in range(tensor.size(0)):
        save_image(tensor[i], os.path.join(output_dir, f"{base_filename}_{i}.png"))

def iterate_data_loaders(loaders, iter_nums):
    """
    Reset and get iterators for data loaders if needed.
    """
    for i, loader in enumerate(loaders):
        if iter_nums[i] % len(loader) == 0:
            loaders[i] = iter(loader)
    return loaders

def main():
    args = utils.get_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Determine source domains based on target.
    # Set ``RPPG_DATA_ROOT`` env var to point at the STMap root directory; falls
    # back to ./Data/STMap/ for local smoke tests.
    source_domain_names = TARGET_DOMAIN[args.tgt]
    root_file = os.environ.get('RPPG_DATA_ROOT', './Data/STMap/')

    # Prepare paths for sources and target
    source_paths = prepare_paths(root_file, source_domain_names)  # Assuming first 4 are sources
    target_path = prepare_paths(root_file, [args.tgt])[0]

    # Training parameters
    batch_size_num = args.batchsize
    epoch_num = args.epochs
    learning_rate = args.lr
    num_workers = args.num_workers
    GPU = args.GPU
    input_form = args.form
    reTrain = args.reTrain
    frames_num = args.frames_num
    fold_num = args.fold_num
    fold_index = args.fold_index

    best_mae = 99

    print(f'batch num: {batch_size_num}, epoch_num: {epoch_num}, GPU Index: {GPU}')
    print(f'frames num: {frames_num}, learning rate: {learning_rate}')
    print(f'fold num: {fold_num}, fold index: {fold_index}')

    # Setup logging
    os.makedirs('./Result_log', exist_ok=True)
    rPPGNet_name = f"rPPGNet_{args.tgt}Spatial{args.spatial_aug_rate}Temporal{args.temporal_aug_rate}"
    para_name = str(args.k1) + '_' + str(args.k2)+ '_|' + str(args.k3)+ '_' + str(args.k4)+ '_|' \
                + str(args.k5)+ '_' + str(args.k6)+ '_|' + str(args.k7)+ '_' + str(args.k8)+ '_|' \
                + str(args.k9)+ '_' + str(args.k10)
    log = Logger()
    log_path = os.path.join('./Result_log', f"{rPPGNet_name}_{para_name}_log.txt")
    log.open(log_path, mode='a')
    log.write(f"\n----------------------------------------------- [START {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {'-' * 51}\n\n")

    # Setup device
    device = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu')
    print('Using device:', 'GPU' if torch.cuda.is_available() else 'CPU')

    # Prepare datasets
    if args.reData == 1:
        for path in source_paths + [target_path]:
            os.makedirs(path['save_root'], exist_ok=True)
            domain_indices = os.listdir(path['file_root'])
            MyDataset.getIndex(
                path['file_root'], domain_indices,
                path['save_root'], path['map_file'],
                10, frames_num
            )

    # Load datasets
    source_datasets = load_datasets(source_paths, frames_num, args)
    target_dataset = MyDataset.Data_DG(
        root_dir=target_path['save_root'],
        dataName=target_path['name'],
        STMap=target_path['map_file'],
        frames_num=frames_num,
        args=args
    )

    # Create DataLoaders
    source_loaders = [
        DataLoader(ds, batch_size=batch_size_num, shuffle=True, num_workers=num_workers)
        for ds in source_datasets
    ]
    tgt_loader = DataLoader(target_dataset, batch_size=batch_size_num, shuffle=False, num_workers=num_workers)

    # Initialize the network
    BaseNet = model.BaseNet()
    if reTrain == 1:
        model_path = os.path.join('./Result_Model', rPPGNet_name)
        BaseNet = torch.load(model_path, map_location=device)
        print(f'Loaded pretrained model from {model_path}')
    BaseNet.to(device=device)

    # Define optimizer and loss functions
    optimizer = torch.optim.Adam(BaseNet.parameters(), lr=learning_rate)
    loss_funcs = {
        'NP': MyLoss.P_loss3().to(device),
        'L1': nn.L1Loss().to(device),
        'SP': MyLoss.SP_loss(device, clip_length=frames_num).to(device),
        'ST': MyLoss.ST_loss().to(device),
    }

    # Initialize iterators
    source_iters = [iter(loader) for loader in source_loaders]
    tgt_iter = iter(tgt_loader)
    src_iter_per_epoch = [len(loader) for loader in source_loaders]
    tgt_iter_per_epoch = len(tgt_iter)

    max_iter = args.max_iter
    start_time = timer()

    # Training Loop
    BaseNet.train()
    for iter_num in range(max_iter + 1):
        # Reset iterators if needed
        for i in range(len(source_loaders)):
            if iter_num % src_iter_per_epoch[i] == 0:
                source_iters[i] = iter(source_loaders[i])

        # Prepare data from all sources
        inputs = []
        bvp_targets = []
        hr_rels = []
        inputs_aug = []
        bvp_targets_aug = []
        hr_rels_aug = []
        source_losses = []

        for i, source_iter in enumerate(source_iters):
            try:
                data, bvp, HR_rel, data_aug, bvp_aug, HR_rel_aug = next(source_iter)
            except StopIteration:
                source_iters[i] = iter(source_loaders[i])
                data, bvp, HR_rel, data_aug, bvp_aug, HR_rel_aug = next(source_iters[i])

            inputs.append(data.float().to(device))
            bvp_targets.append(bvp.float().to(device))
            hr_rels.append(torch.Tensor(HR_rel).float().to(device))

            inputs_aug.append(data_aug.float().to(device))
            bvp_targets_aug.append(bvp_aug.float().to(device))
            hr_rels_aug.append(torch.Tensor(HR_rel_aug).float().to(device))

        # Concatenate data from all sources
        inputs = torch.cat(inputs, dim=0)
        bvp_targets = torch.cat(bvp_targets, dim=0)
        hr_rels = torch.cat(hr_rels, dim=0)

        inputs_aug = torch.cat(inputs_aug, dim=0)
        bvp_targets_aug = torch.cat(bvp_targets_aug, dim=0).unsqueeze(1)
        hr_rels_aug = torch.cat(hr_rels_aug, dim=0)

        optimizer.zero_grad()

        # Forward pass
        bvp_pre, HR_pr, rendered = BaseNet(inputs)
        bvp_pre_aug, HR_pr_aug, rendered_aug = BaseNet(inputs_aug)

        # L_rec on both original positions p and motion-corrected positions p̂
        L_rec = (loss_funcs["L1"](inputs[rendered["indices_orig"]], rendered["image_orig"])
               + loss_funcs["L1"](inputs[rendered["indices_mod"]],  rendered["image_mod"]))

        # L_st: spatio-temporal consistency on v_d (physiological diffuse component)
        L_st = loss_funcs["ST"](rendered["v_d"])

        # L_m: motion-flow modulus vs motion-noise v_n modulus negative Pearson
        L_m = MyLoss.M_loss(rendered["motion_flow"], rendered["v_n"])

        if iter_num % 1000 == 0 and iter_num > 1000:
            save_images(rendered["image_orig"], './exp' + str(iter_num))
        # Split predictions back to individual sources if needed
        # (Optional: Depending on how MyLoss.get_loss handles multiple sources)

        # Compute losses for original and augmented data
        src_loss = 0
        src_loss_aug = 0
        for i in range(len(source_loaders)):
            start_idx = sum([args.batchsize for j in range(i)])
            end_idx = start_idx + args.batchsize

            src_loss += MyLoss.get_loss(
                bvp_pre[start_idx:end_idx],
                HR_pr[start_idx:end_idx],
                bvp_targets[start_idx:end_idx],
                hr_rels[start_idx:end_idx],
                source_domain_names[i],
                loss_funcs['NP'],
                loss_funcs['L1'],
                args,
                iter_num
            )

            src_loss_aug += MyLoss.get_loss(
                bvp_pre_aug[start_idx:end_idx],
                HR_pr_aug[start_idx:end_idx],
                bvp_targets_aug[start_idx:end_idx],
                hr_rels_aug[start_idx:end_idx],
                source_domain_names[i],
                loss_funcs['NP'],
                loss_funcs['L1'],
                args,
                iter_num
            )



        # Total loss: L_all = L_physio + λ_rec L_rec + λ_st L_st + λ_m L_m
        loss = (src_loss + src_loss_aug
                + args.lambda_rec * L_rec
                + args.lambda_st * L_st
                + args.lambda_m * L_m)

        # Check for NaNs
        if torch.isnan(loss).any():
            print('Encountered NaN in loss. Stopping training.')
            break

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Logging
        if iter_num % 200 == 0:
            elapsed = timer() - start_time
            log_message = (
                f'Train Iter: {iter_num} | loss: {loss.item():.4f} | '
                f'L_rec: {L_rec.item():.4f} | L_st: {L_st.item():.4f} | L_m: {L_m.item():.4f} | '
                f'Time: {time_to_str(elapsed, "min")}'
            )
            log.write(log_message + '\n')
            print(log_message)
        if iter_num % 2000 == 0 and iter_num>1000:
            # Testing Phase
            BaseNet.eval()
            HR_pr_temp = []
            HR_rel_temp = []
            BVP_ALL = []
            BVP_PR_ALL = []

            with torch.no_grad():
                for step, (data, bvp, HR_rel, _, _, _) in enumerate(tgt_loader):
                    data = data.float().to(device)
                    bvp = bvp.float().unsqueeze(1).to(device)
                    HR_rel = HR_rel.float().to(device)

                    Wave_pr, HR_pr, av = BaseNet(data)

                    HR_rel_temp.extend(HR_rel.cpu().numpy())
                    HR_pr_temp.extend(HR_pr.cpu().numpy())
                    BVP_ALL.extend(bvp.cpu().numpy())
                    BVP_PR_ALL.extend(Wave_pr.cpu().numpy())

            # Evaluation
            print('HR Evaluation:')
            ME, STD, MAE, RMSE, MER, P = utils.MyEval(HR_pr_temp, HR_rel_temp)
            evaluation_message = (
                f'Test Iter: {max_iter} | ME: {ME:.4f} | STD: {STD:.4f} | '
                f'MAE: {MAE:.4f} | RMSE: {RMSE:.4f} | MER: {MER:.4f} | P: {P:.4f}'
            )
            log.write(evaluation_message + '\n')
            print(evaluation_message)

            # Save Results
            os.makedirs('./Result_Model', exist_ok=True)
            os.makedirs('./Result', exist_ok=True)

            io.savemat(f'./Result/{rPPGNet_name}_HR_pr.mat', {'HR_pr': HR_pr_temp})
            io.savemat(f'./Result/{rPPGNet_name}_HR_rel.mat', {'HR_rel': HR_rel_temp})
            io.savemat(f'./Result/{rPPGNet_name}_WAVE_ALL.mat', {'Wave': BVP_ALL})
            io.savemat(f'./Result/{rPPGNet_name}_WAVE_PR_ALL.mat', {'Wave': BVP_PR_ALL})

            model_save_path = os.path.join('./Result_Model', rPPGNet_name)
            torch.save(BaseNet.state_dict(), model_save_path)
            print(f'Saved model as {model_save_path}')

if __name__ == '__main__':
    main()
