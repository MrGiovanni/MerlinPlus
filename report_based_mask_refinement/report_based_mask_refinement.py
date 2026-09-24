#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filter_predictions_debug.py · May‑2025

Adds: estimated tumour volumes / diameters and
      'organs with unknown tumour size' to each info.txt

Example:
 python r_super_pseudo_masks.py --pred_root /projects/bodymaps/Pedro/foundational/MedFormer/result/ufo_train_set/abdomenatlas/MULTI_TUMOR_db_Atlas_UCSF_no_liver_kidney_pancreas_basic_balanced_cropper \
    --output_folder '/projects/bodymaps/Pedro/data/r_super_pseudo_masks_ufo_debugging/' \
    --source_ct /projects/bodymaps/Data/UFO_27k_medformer/ \
    --overwrite --debug

python r_super_pseudo_masks.py --pred_root /projects/bodymaps/Pedro/foundational/MedFormer/result/CT_RATE_multi_tumor/abdomenatlas/MULTI_TUMOR_db_Atlas_UCSF_no_liver_kidney_pancreas_basic_balanced_cropper_ball_attenuation_MLP_trained_on_mask_w001/ --output_folder /projects/bodymaps/Pedro/data/ct_rate_pseudo_labels/MULTI_TUMOR_db_Atlas_UCSF_no_liver_kidney_pancreas_basic_balanced_cropper_ball_attenuation_MLP_trained_on_mask_w001/ --source_ct /projects/bodymaps/Data/image_only/AbdomenAtlasPro/AbdomenAtlasPro/ --meta /projects/bodymaps/Data/CT_RATE/multi_tumor_test_set_no_kidney_no_pancreas_no_liver.csv --reports /projects/bodymaps/Data/CT_RATE/per_tumor_metadata_with_BDMAP_ID_cleaned.csv

python r_super_pseudo_masks.py --pred_root /projects/bodymaps/Pedro/foundational/MedFormer/result/CT_RATE_pancreas_test_infer2/abdomenatlas/Atlas_JHH_UFO_training_large_tolerances --output_folder /projects/bodymaps/Pedro/data/r_super_pseudo_masks_ct_rate_pancreas --source_ct /projects/bodymaps/Data/image_only/AbdomenAtlasPro/AbdomenAtlasPro/ --meta /projects/bodymaps/Data/CT_RATE/metadata_filled_ct_rate.csv --reports /projects/bodymaps/Data/CT_RATE/per_tumor_metadata_with_BDMAP_ID_cleaned.csv --classes pancreatic
"""

from __future__ import annotations
import argparse, random, shutil, sys, unicodedata, math
from collections import Counter
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import yaml
import torch.nn.functional as F
import nibabel as nib
from tqdm import tqdm
import os
import csv

empty_pred_raw = []

def organs_from_metadata_row(
    row: pd.Series,
    allowed: set[str],
    min_lesions: int = 1,
) -> tuple[set[str], list[str]]:
    """
    Return organs (subset of `allowed`) where the metadata says
    `number of <organ> lesion instances` >= min_lesions.

    Returns:
      organs_set, problems_list
    """
    organs = set()
    probs = []

    for o in sorted(allowed):
        col = column_from_organ(o)

        if col not in row.index:
            probs.append(f"{o}: column '{col}' missing")
            continue

        v = row[col]

        # robust numeric conversion
        if pd.isna(v):
            n = 0
        else:
            try:
                n = int(float(v))
            except Exception:
                probs.append(f"{o}: non-numeric value '{v}' in '{col}'")
                n = 0

        if n >= min_lesions:
            organs.add(o)

    return organs, probs

def clean_subseg_list(tumor_segments):
        #split tumor segments that have /
        tmp=[]
        for segment in tumor_segments:
            if pd.isna(segment) or segment == 'u':
                continue
            else:
                sublist=segment.split(' / ')
                if sublist not in tmp:
                    tmp.append(sublist)
        tumor_segments = tmp
        tumor_segments_flat = list(set([item for sublist in tmp for item in sublist]))
        return tumor_segments, tumor_segments_flat

def read_report(id):
    global reports_df
    reports=reports_df
    if id not in reports['BDMAP_ID'].values:
        #print('ID is: ',id)
        raise ValueError('ID is not in the reports:', id, 'Length of reports:', len(reports))
        return None #no tumor
    else:
        tumors=reports[reports['BDMAP_ID']==id]
        #tumors=tumors.to_dict(orient='records')
        return tumors

def get_tumor_segment_labels(idx):
        """
        This function reads the LLM output for a given report, and its most importat outputs are subseg_with_only_known_sizes and organs_with_only_known_sizes_n_segments.
        These outputs represent organ/organ subsegments that contain tumors but do not contain tumors with unknown size.  
        """
        tumors=read_report(idx)
        if tumors is None:
            #no tumor, just do random crop
            retur = {'tumor_segments':[],
                    'tumor_segments_flat':[],
                    'tumor_organs':[],
                    'organs_with_unk_tumor_segment':[],
                    'organs_with_unk_tumor_size':[],
                    'organs_with_only_known_sizes_n_segments':[],
                    'subseg_with_only_known_sizes':[],
                    'subseg_with_unk_tumor_size':[],
                    'subsegs_in_organs_with_unk':[]}
            #print('No tumor found for:', img_list[idx], flush=True, file=sys.stderr)
            return retur,tumors
        else:
            #tumor is present
            tumor_segments = tumors['Standardized Location'].tolist()
            tumor_sizes = tumors['Tumor Size (mm)'].tolist()
            tumor_organs = tumors['Standardized Organ'].tolist()
            
            #add organ names to segments
            tmp=[]
            for i,s in enumerate(tumor_segments,0):
                if pd.isna(s) or s == 'u':
                    tmp.append(s)
                else:
                    tmp.append(tumor_organs[i]+'_'+s)
            tumor_segments = tmp


            #check which organs have tumors with unknown size or segment
            organs_with_unk_tumor_segment = []
            organs_with_unk_tumor_size = []
            #and which subsegments have unknown size
            subseg_with_unk_tumor_size = []
            for i in list(range(len(tumor_organs))):
                if pd.isna(tumor_sizes[i]) or tumor_sizes[i] == 'u' or tumor_sizes[i] == 'multiple' or not any(ch.isdigit() for ch in tumor_sizes[i]):
                    organs_with_unk_tumor_size.append(tumor_organs[i])
                    subseg_with_unk_tumor_size.append(tumor_segments[i])
                if pd.isna(tumor_segments[i]) or tumor_segments[i] == 'u':
                    organs_with_unk_tumor_segment.append(tumor_organs[i])

            #check which segments are in an organ with some unknown tumor size or segment
            subsegs_in_organs_with_unk = []
            for i in list(range(len(tumor_organs))):
                #check if the organ is not in the list of organs with unknown tumor segment
                if tumor_organs[i] in organs_with_unk_tumor_segment or tumor_organs[i] in organs_with_unk_tumor_size:
                    subsegs_in_organs_with_unk.append(tumor_segments[i])

            

            tumor_segments, tumor_segments_flat = clean_subseg_list(tumor_segments)
            subseg_with_unk_tumor_size, subseg_with_unk_tumor_size_flat = clean_subseg_list(subseg_with_unk_tumor_size)
            subsegs_in_organs_with_unk, subsegs_in_organs_with_unk_flat = clean_subseg_list(subsegs_in_organs_with_unk)

            tumor_organs = list(set(organ for organ in tumor_organs if not pd.isna(organ) and organ != 'u'))
            organs_with_unk_tumor_segment = list(set(organ for organ in organs_with_unk_tumor_segment if not pd.isna(organ) and organ != 'u'))
            organs_with_unk_tumor_size = list(set(organ for organ in organs_with_unk_tumor_size if not pd.isna(organ) and organ != 'u'))

            #subsegments with only known sizes
            subseg_with_only_known_sizes = list(set(tumor_segments_flat) - set(subseg_with_unk_tumor_size_flat) - set(subsegs_in_organs_with_unk_flat))
            #organs with only known sizes and locations of tumors
            organs_with_only_known_sizes_n_segments = list(set(tumor_organs) - set(organs_with_unk_tumor_segment) - set(organs_with_unk_tumor_size))
            organs_with_only_known_sizes = list(set(tumor_organs) - set(organs_with_unk_tumor_size))

            #for subseg_with_only_known_sizes, you must check tumor_segments, and consider segments that come in pairs
            #check if some sub-segment is in more than one item in the list, if so, merge the items
            tmp=[]
            for segment in subseg_with_only_known_sizes:
                #get all items that contain the segment in the list tumor_segments
                items = [item for item in tumor_segments if segment in item]
                #flatten
                items = list(set([item for sublist in items for item in sublist]))
                #items represent a list of sub-segments that share tumors with segment
                #check if any of them is in the list of prohibted segments
                if any(item in subseg_with_unk_tumor_size_flat for item in items) or \
                   any(item in subsegs_in_organs_with_unk_flat for item in items):
                    continue
                else:
                    tmp.append(items)
            subseg_with_only_known_sizes=tmp
                

            #create a big dict with the variables here
            retur = {'tumor_segments':tumor_segments,
                    'tumor_segments_flat':tumor_segments_flat,
                    'tumor_organs':tumor_organs,
                    'organs_with_unk_tumor_segment':organs_with_unk_tumor_segment,
                    'organs_with_unk_tumor_size':organs_with_unk_tumor_size,
                    'organs_with_only_known_sizes_n_segments':organs_with_only_known_sizes_n_segments,
                    'organs_with_only_known_sizes':organs_with_only_known_sizes,
                    'subseg_with_only_known_sizes':subseg_with_only_known_sizes,
                    'subseg_with_unk_tumor_size':subseg_with_unk_tumor_size,
                    'subsegs_in_organs_with_unk':subsegs_in_organs_with_unk}
            #print('Retur:', retur, flush=True, file=sys.stderr)
            #raise ValueError('You must change the handling of this function output everywhere it is used')
            #print('Tumor Dict:', tumors[['Standardized Location','Tumor Size (mm)','Standardized Organ']])
            #print('subseg_with_only_known_sizes:', retur['subseg_with_only_known_sizes'], flush=True, file=sys.stderr)
            #print('organs_with_only_known_sizes_n_segments:', retur['organs_with_only_known_sizes_n_segments'], flush=True, file=sys.stderr)
            #check if tumor_organs is not nan
            
            #if isinstance(retur['tumor_organs'],str):#if nan it is a normal case
            #    #print('XXXXXXXX Tumor Found for:', self.img_list[idx],f'tumor is: {tumors[["Standardized Location","Tumor Size (mm)","Standardized Organ"]]}', flush=True, file=sys.stderr)
            return retur,tumors

def estimate_tumor_volume(idx, tumor_segment_crop):
        """
        Estimates tumor volume from reports. For the segment in the crop.
        Always returns a list of 10 items, padding with 0.
        """
        _,tumor_dict=get_tumor_segment_labels(idx)
        #print('Tumor dict:', tumor_dict)
        #print all column names in tumor_dict
        #print(tumor_dict.columns)
        #print('Sizes:',tumor_dict['Tumor Size (mm)'])
        #print('Cropped on tumor segment:', tumor_segment_crop)
        if tumor_segment_crop is None or tumor_segment_crop=='random':
            return [0,0,0,0,0,0,0,0,0,0], torch.zeros((10,3)).float() #CT not cropped around a tumor segment

        
        if isinstance(tumor_segment_crop, list):
            pass
        elif isinstance(tumor_segment_crop, str):
            tumor_segment_crop=[tumor_segment_crop]
        else:
            raise ValueError('tumor_segment_crop must be a list or a string.')
        
        #is our tumor_segment_crop organ or segment:
        
        tpe='organ'
        col='Standardized Organ'
        
        tumors_in_crop=[]
        for row in tumor_dict.iterrows():
            location=row[1][col]
            #print('Location:',location)
            if not isinstance(location, str) or location.lower()=='u':
                continue
            if '/' in location:
                location=location.split(' / ')
            if not isinstance(location, list):
                location=[location]
            in_crop=True
            for loc in location:
                if loc not in tumor_segment_crop:
                    in_crop=False
                    break
            if in_crop:
                tumors_in_crop.append(row[1]['Tumor Size (mm)'])

            #print('Tumors in crop:', tumors_in_crop)#list of strings with sizes

            #print('Tumor dict:',tumor_dict[['Standardized Organ','Standardized Location','Tumor Size (mm)']])
                
        #estimate volumes for each tumor size
        volumes=[]
        diameters=[]
        for size in tumors_in_crop:
            if size=='u' or size=='U' or size=='multiple' or (isinstance(size,float) and np.isnan(size)):
                return 'no size'
            if 'x' not in size:
                #single diameter provided. Use ball.
                try:
                    diameter=float(size)
                except:
                    return 'no size'
                volume=(4/3) * math.pi * ((diameter/2) ** 3)#sphere. volume in mm3 (voxels)
                volumes.append(volume)
                diameters.append([diameter,diameter,diameter])
            else:
                #2 or 3 diameterts, use ellipsoid
                sizes=size.split(' x ')
                sizes=[float(s) for s in sizes]
                if len(sizes)==2:
                    #assume 3rd axis is the average of the other two
                    sizes.append(sum(sizes)/2)
                elif len(sizes) > 3:                
                    # more than 3 numbers, take top 3
                    sizes = sorted(sizes, reverse=True)[:3]
                #ellipsoid volume
                volume=(4/3) * math.pi * ((sizes[0]/2) * (sizes[1]/2) * (sizes[2]/2))
                volumes.append(volume)
                diameters.append(sizes)

        #print('Estimated volumes:',volumes)

        for i in range(len(volumes),10):
            volumes.append(0)
            diameters.append([0,0,0])
            
        return volumes,torch.tensor(diameters).float()

# ───────────────────── helpers (canon, STD_ORGANS, column_from_organ …)
ALIASES={"gall bladder":"gallbladder","gall_bladder":"gallbladder","gall‑bladder":"gallbladder"}
def canon(n:str)->str:n=unicodedata.normalize("NFKC",n).strip().lower();return ALIASES.get(n.replace(" ","_"),n.replace(" ","_"))
STD_ORGANS=[canon(o) for o in("bladder","duodenum","esophagus","gallbladder","prostate",
                               "spleen","stomach","uterus")]
def column_from_organ(o:str)->str:return f"number of {o.replace('_',' ').replace('adrenal','adrenal gland').replace('pancreas','pancreatic')} lesion instances"
def predicted_organs(p:Path)->set[str]:
    s=set()
    for f in p.iterdir():
        if f.is_dir():continue
        n=f.name.lower()
        if "lesion" not in n or not n.endswith((".nii.gz",".npz")):continue
        stem=n[:-7] if n.endswith(".nii.gz") else n.rsplit(".",1)[0]
        s.add(canon(stem.replace("_lesion","").replace('pancreatic','pancreas')))
    return s
def reports_agree(row:pd.Series,orgs:set[str])->tuple[bool,list[str]]:
    probs=[]
    for o in orgs:
        c=column_from_organ(o)
        if c not in row:probs.append(f"{o}: column '{c}' missing")
        #elif row[c]==0: probs.append(f"{o}: value is 0")
    return (len(probs)==0,probs)


def denoise_keep_edges(t: torch.Tensor,
                       erode_iters: int = 2,
                       dilate_iters: int = 3,
                       kernel_size: int = 3) -> torch.Tensor:
    """
    • takes a (D,H,W) *or* (B,C,D,H,W) binary/float mask `t`
    • applies 2 erosions → 3 dilations (kernel = ball of `kernel_size`)
    • multiplies the result with the original tensor (keeps edges)
    • returns a tensor on the same device & dtype
    """
    proc = t.clone()
    for _ in range(erode_iters):
        proc = dilate_volume(proc, kernel_size,erode=True)

    for _ in range(dilate_iters):
        proc = dilate_volume(proc, kernel_size)            # uses your existing fct.

    proc = proc * t                                        # element‑wise AND (×)
    proc = proc.type_as(t)                                  # ensure same dtype as input
    return proc

def dilate_volume(volume, kernel_size, full_pass_radius=3,erode=False):
    # ensure odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    # for small kernels, just do one pass
    if kernel_size <= (2*full_pass_radius+1):
        return dilate_volume_conv(volume, kernel_size,erode=erode)

    # compute how many "radius‑3" (kernel=7) passes we need
    # radius = (kernel_size‑1)//2  (an integer number of voxels)
    radius = (kernel_size - 1) // 2

    num_full = radius // full_pass_radius  # integer division
    remainder = radius % full_pass_radius  # 0, 1, or 2

    # apply all full radius‑3 passes
    for _ in range(num_full):
        volume = dilate_volume_conv(volume, 2*full_pass_radius + 1,erode=erode)

    # handle the leftover radius if any (1→kernel=3, 2→kernel=5)
    if remainder > 0:
        volume = dilate_volume_conv(volume, 2*remainder + 1,erode=erode)

    return volume



def dilate_volume_conv(volume, kernel_size,erode=False):
    """
    Applies binary dilation to a 3D binary volume using max pooling.

    Parameters:
        volume (torch.Tensor): The input binary volume with shape
            [batch, channels, depth, height, width]. The volume should be binary (0 or 1).
        kernel_size (int): The size of the cubic structuring element (must be an odd number).

    Returns:
        torch.Tensor: The dilated binary volume with the same shape as the input.
    """
    reduce=0
    if len(volume.shape) == 3:
        volume = volume.unsqueeze(0).unsqueeze(0)
        reduce=2
    if len(volume.shape) == 4:
        volume = volume.unsqueeze(0)
        reduce=1
    assert len(volume.shape) == 5, f"Input tensor should be 5D, got {volume.shape}"

    # Ensure the kernel size is odd for proper centering.
    if kernel_size % 2 == 0:
        kernel_size+=1



    # Apply max pooling with stride=1 and the computed padding.
    # This will output a 1 if any voxel in the kernel window is 1 (binary dilation).
    #we can use a maxpool or a ball convolution to dilate the volume. Maxpool should be faster, but it uses a cube kernel, while the ball kernel is more accurate.
    #dilated = F.max_pool3d(volume, kernel_size=kernel_size, stride=1, padding=padding)
    ball_kernel = create_ball_kernel(kernel_size).type_as(volume).unsqueeze(0).unsqueeze(0).repeat(volume.shape[1],1, 1, 1, 1)

    # Calculate padding such that the output size is the same as the input size.
    kernel_size = ball_kernel.shape[-1]
    padding = kernel_size // 2

    dilated = F.conv3d(volume, ball_kernel, padding=padding, groups=volume.shape[1])
    #binarize
    if not erode:
        dilated = (dilated > 0).float()
    else:
        full_hit = ball_kernel[0, 0].sum().type_as(dilated)
        dilated = (dilated==full_hit).float()

    assert dilated.shape == volume.shape, "Output shape must match input shape."

    if reduce == 1:
        dilated = dilated.squeeze(0)
    elif reduce == 2:
        # Reduce back to original shape if we added extra dimensions.
        dilated = dilated.squeeze(0).squeeze(0)

    return dilated

def create_ball_kernel(diameter, gaussian=False, gaussian_std=1.5):
    """
    Creates a 3D torch tensor (kernel) where there is a 'ball' of a given diameter.
    The diameter is first rounded up to the next odd integer. The kernel size is then
    computed to be 1.2 × (that odd diameter), rounded to the next odd integer.
    
    The ball is centered in this larger kernel. Inside the ball (hard cutoff at the
    ball boundary), values are set to 1 (or to a truncated Gaussian if `gaussian=True`).
    Outside the ball, values are 0. If `gaussian=True`, the Gaussian is centered at
    the ball center with standard deviation `gaussian_std * radius`.

    Parameters
    ----------
    diameter : float or int
        Desired diameter of the ball. Will be rounded up to the next odd integer.
    gaussian : bool, optional
        Whether to fill the ball with a Gaussian distribution, by default False.
    gaussian_std : float, optional
        Standard deviation factor (relative to the ball radius) if gaussian=True.
        For example, if the ball's radius is R and gaussian_std=1.5, the std is
        1.5*R, by default 1.5.

    Returns
    -------
    kernel : torch.FloatTensor
        A 3D tensor of shape (kernel_size, kernel_size, kernel_size) containing
        the ball (or Gaussian ball) centered in the kernel.
    """

    # --- Step 1: Round diameter to next odd integer ---
    diameter_ceil = math.ceil(diameter)
    if diameter_ceil % 2 == 0:
        diameter_ceil += 1
    diameter_odd = diameter_ceil  # The final odd diameter
    
    # --- Step 2: Compute kernel size as 1.2 * diameter_odd, also round up to next odd ---
    kernel_size_float = 1.2 * diameter_odd
    kernel_size_ceil = math.ceil(kernel_size_float)
    if kernel_size_ceil % 2 == 0:
        kernel_size_ceil += 1
    kernel_size = kernel_size_ceil  # The final odd kernel size
    
    # Ball radius (float)
    radius = diameter_odd / 2.0

    # --- Create 1D coordinate grid from 0..(kernel_size-1), shift so center is 0 ---
    center = (kernel_size - 1) / 2.0
    coords = torch.arange(kernel_size, dtype=torch.float32)
    coords_shifted = coords - center  # center at 0
    
    # --- Compute squared distance (3D) via broadcasting ---
    distance_squared = (coords_shifted[:, None, None] ** 2
                      + coords_shifted[None, :, None] ** 2
                      + coords_shifted[None, None, :] ** 2)
    
    # --- Hard cutoff mask for the ball ---
    mask = (distance_squared <= radius**2).float()
    
    if gaussian:
        # Scale std by the ball's actual radius
        std = gaussian_std * radius
        gaussian_values = torch.exp(-distance_squared / (2.0 * std**2))
        kernel = gaussian_values * mask
        # Normalize so that sum of kernel = 1
        kernel = kernel / kernel.sum()
    else:
        kernel = mask  # Binary ball kernel

    #assert the kernel size is odd
    assert kernel.shape[0] % 2 == 1, f'Kernel size should be odd, got {kernel.shape[0]}'
    
    return kernel




def DiceLossMultiClass(preds, targets, known_voxels, alpha = 0.5, beta=0.5, size_average=True, reduce=True, 
                       sigmoid=True, class_weights=None):

    if len(preds.shape)==3:
        preds=preds.unsqueeze(0).unsqueeze(0)
    if len(targets.shape)==3:
        targets=targets.unsqueeze(0).unsqueeze(0)
    if len(known_voxels.shape)==3:
        known_voxels=known_voxels.unsqueeze(0).unsqueeze(0)

    if len(preds.shape)==4:
        preds=preds.unsqueeze(0)
        targets=targets.unsqueeze(0)
        known_voxels=known_voxels.unsqueeze(0)

    assert len(preds.shape)==5
    assert (preds.shape == targets.shape) and (targets.shape == known_voxels.shape), f"Shapes do not match, pred, target and unk are: {preds.shape}, {targets.shape}, {known_voxels.shape}"

    N = preds.size(0)
    C = preds.size(1)
    
    if sigmoid:
        P = torch.sigmoid(preds)
    else:
        P = preds

    P = P * known_voxels
    targets = targets * known_voxels

    smooth = 1e-5

    class_mask = targets

    ones = torch.ones(P.shape).to(P.device)
    P_ = ones - P 
    class_mask_ = ones - class_mask

    TP = P * class_mask
    FP = P * class_mask_
    FN = P_ * class_mask

    alpha = FP.transpose(0, 1).reshape(C, -1).sum(dim=(1)) / ((FP.transpose(0, 1).reshape(C, -1).sum(dim=(1)) + FN.transpose(0, 1).reshape(C, -1).sum(dim=(1))) + smooth)
    alpha = alpha.unsqueeze(0).repeat(N, 1) # repeat for each batch item, now alpha is B,C

    alpha = torch.clamp(alpha, min=0.2, max=0.8) 
    #print('alpha:', alpha)
    beta = 1 - alpha
    num = torch.sum(TP, dim=(-1,-2,-3)).float()
    den = num + alpha * torch.sum(FP, dim=(-1,-2,-3)).float() + beta * torch.sum(FN, dim=(-1,-2,-3)).float()

    dice = num / (den + smooth)
    loss = 1 - dice
    if class_weights is not None:
        class_weights = class_weights.mean(dim=(-1,-2,-3))
        while len(class_weights.shape) < len(loss.shape):
            class_weights = class_weights.unsqueeze(0)
        assert class_weights.shape == loss.shape, f'Class weights shape {class_weights.shape} does not match the shape of dice loss {loss.shape}'
        # Apply class weights
        loss = loss * class_weights
    
    if not reduce:
        return loss

    if size_average:
        assert len(loss.shape) == 2, f'Loss should be 2D after reduction, but got {loss.shape}.'
        loss = loss.mean()  # Average over the batch size

    return loss

def isolate_tumor(x, diameter, gaussian, gaussian_std, tumor_volume,
                  diameter_margin=0.5,volume_margin=0.5):
    """
    Uses a ball convolution over x and applies a maximum operation to find the best
    fitting ball center. Then, it multiplies the input by a volume with the same size
    as the input, but with a binary ball placed at the given object center coordinate.
    Finally, after the multiplication, we find the top N voxels inside the remaining volume.
    N is the tumor volume.

    
    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W, D).
        diameter (int): Diameter of the ball kernel.
        gaussian (bool): Whether to use a Gaussian weighting inside the ball (for convolution).
        gaussian_std (float): Standard deviation of the Gaussian.
        tumor_volume (int): Number of voxels to select as the tumor volume.
    
    Returns:masked_volume should be within 
        torch.Tensor: A binary tumor mask of shape (H, W, D) with 1's in the top N voxels.
    """
    reduce=False
    if len(x.shape)==3:
        reduce=True
        x = x.unsqueeze(0).unsqueeze(0)
    assert len(x.shape) == 5, f"Input tensor should be 5D, got {x.shape}"


    #round diameter
    diameter = np.round(diameter).astype(int)
    #round tumor volume
    tumor_volume = np.round(tumor_volume).astype(int)

    MAX_DIAMETER = max(x.shape[-3:])
    # clamp diameter if larger than crop
    if diameter > MAX_DIAMETER:
        print(f"[isolate_tumor] Clamping diameter from {diameter} to {MAX_DIAMETER}")
        diameter = MAX_DIAMETER
        # max possible ball volume for this (clamped) diameter
        radius = diameter / 2.0
        max_ball_vol = int((4.0 / 3.0) * math.pi * (radius ** 3) * 1.2)
        if tumor_volume > max_ball_vol:
            print(f"[isolate_tumor] Clamping tumour volume from {tumor_volume} to {max_ball_vol}")
            tumor_volume = max_ball_vol

    # Ensure the diameter is odd.
    if diameter % 2 == 0:
        diameter += 1

    # Create the ball kernel for convolution.
    kernel = create_ball_kernel(diameter, gaussian, gaussian_std).type_as(x)
    # Convert kernel to a 5D tensor (shape: 1, 1, H, W, D).
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    #assert volume is not larger than the number of voxels in the ball
    if tumor_volume > 100000:
        assert tumor_volume <= (kernel>0).sum()*1.2, f'Tumor volume should be smaller than the number of voxels in the ball, got {tumor_volume} and {(kernel>0).sum()}'

    if (kernel>0).sum() > tumor_volume:
        #we tolerate numerical erros within a margin of 0.2
        tumor_volume = (kernel>0).sum()-1

    
    # Perform 3D convolution.
    out = F.conv3d(x, kernel, padding=kernel.shape[-1] // 2)

    assert out.shape == x.shape, f"Output shape should match input shape, got {out.shape} vs {x.shape}"

    # --- Step 1: Find the best fitting ball center ---
    # Assume x is of shape (1, 1, H, W, D); take the spatial part.
    out_spatial = out[0, 0]  # shape: (H, W, D)
    max_idx = torch.argmax(out_spatial)
    best_center = np.unravel_index(max_idx.item(), out_spatial.shape)  # (cx, cy, cz)
    
    # --- Step 2: Create a binary ball mask at the best center ---
    masked_volume = insert_ball(out_spatial,best_center,diameter,diameter_margin)
    new_dim = diameter
    while masked_volume.sum() < tumor_volume:
        #if the ball is in the border of the image, its volume may be less than the tumor volume, We increase the size of the ball until we reach the tumor volume.
        old_dim = new_dim
        new_dim = int(np.round(new_dim * 1.1))
        print(f'Increasing ball size to {new_dim}, current volume is {masked_volume.sum()}, tumor volume is {tumor_volume}')
        if old_dim == new_dim:
            new_dim += 1
        if new_dim % 2 == 0:
            new_dim += 1
        if new_dim >= max(x.shape[-1], x.shape[-2], x.shape[-3]):
            break
        masked_volume = insert_ball(out_spatial,best_center,new_dim,diameter_margin)
    if tumor_volume < (50**3):
        assert (masked_volume.sum() > tumor_volume*0.5), f'masked_volume should be within 20% of the tumor volume! got {masked_volume.sum()} and {tumor_volume}'
    if tumor_volume > (6**3):
        assert (masked_volume.sum() < tumor_volume*((1+diameter_margin)**3)*2), f'masked_volume should be within 20% of the tumor volume! got {masked_volume.sum()} and {tumor_volume} and diameter {diameter}'

    # --- Step 3: Multiply the input by the binary ball mask ---
    # x has shape (B, C, H, W, D); expand masked_volume to match.
    #assert no negative value in x
    assert (x >= 0).all(), f'Input tensor should not have negative values, got {x.min()}'
    masked_x = (x * masked_volume.unsqueeze(0).unsqueeze(0))

    # --- Step 4: Find the top N voxels in the masked volume ---
    # Remove batch and channel dimensions.
    masked_x_vol = masked_x[0, 0]
    flattened = masked_x_vol.reshape(-1)
    # Get indices of the top N voxel values.
    t=min(flattened.shape[-1]-1, tumor_volume)
    margin_small = min(0.5,volume_margin)
    t_small = int(t*(1-margin_small))
    t_small =  max(t_small, min(100,tumor_volume))  # Ensure at 4mm tumor
    t_big = min(flattened.shape[-1]-1,int(tumor_volume*(1+volume_margin)))
    topN_values, topN_indices = torch.topk(flattened, t)
    topN_values_small, topN_indices_small = torch.topk(flattened, t_small)
    topN_values_big, topN_indices_big = torch.topk(flattened, t_big)
    #how many indices? Assert this matches the tumor volume
    assert len(topN_indices) == t, f'Expected {tumor_volume} indices, got {len(topN_indices)}'
    # Create a binary volume: set top N positions to 1, rest to 0.
    tumor_mask_flat = torch.zeros_like(flattened)
    tumor_mask_flat[topN_indices] = 1
    tumor_mask_flat_small = torch.zeros_like(flattened)
    tumor_mask_flat_small[topN_indices_small] = 1
    tumor_mask_flat_big = torch.zeros_like(flattened)
    tumor_mask_flat_big[topN_indices_big] = 1
    
    # Reshape to original spatial dimensions.
    tumor_mask = tumor_mask_flat.view_as(masked_x_vol)
    tumor_mask_small = tumor_mask_flat_small.view_as(masked_x_vol)
    tumor_mask_big = tumor_mask_flat_big.view_as(masked_x_vol)
    # Assert the sum here still matches the tumor volume.
    assert tumor_mask.sum() == t, f'Tumor mask should have the same volume as the tumor volume, got {tumor_mask.sum()} and {tumor_volume}'

    
    #ensure no tumor_max value is outside the ball
    tumor_mask = tumor_mask * masked_volume
    tumor_mask_small = tumor_mask_small * masked_volume
    tumor_mask_big = tumor_mask_big * masked_volume

    if reduce:
        tumor_mask = tumor_mask.squeeze(0).squeeze(0)

    iters = 0
    while tumor_volume < (50**3) and tumor_mask.sum() < tumor_volume*0.7:
        #zero values inside the ball may not be chosen as the top N voxels. In such cases, we dilate the mask
        print(f'dilating tumor mask, iteration {iters}, current volume is {tumor_mask.sum()}, tumor volume is {tumor_volume}')
        if iters >3:
            return tumor_mask, tumor_mask_small, tumor_mask_big
        #dilate the mask
        tumor_mask = dilate_volume(tumor_mask, 7)*masked_volume
        tumor_mask_small = dilate_volume(tumor_mask_small, 7)*masked_volume
        tumor_mask_big = dilate_volume(tumor_mask_big, 7)*masked_volume
        iters += 1

    if tumor_volume < (50**3):
        assert (tumor_mask.sum() > tumor_volume*0.5), f'tumor_mask should have the same volume as the tumor volume, got {tumor_mask.sum()} and {tumor_volume}'
    if tumor_volume > (5**3):
        assert (tumor_mask.sum() < tumor_volume*((1+volume_margin)**3)*3), f'tumor_mask should have the same volume as the tumor volume, got {tumor_mask.sum()} and {tumor_volume}'

    #assert it is binary
    assert (tumor_mask == 0).sum() + (tumor_mask == 1).sum() == tumor_mask.numel(), f'Tumor mask should be binary, got {tumor_mask.sum()}'

    return tumor_mask, tumor_mask_small, tumor_mask_big

counter3=100
def GlobalWeightedRankPooling(x, N=1000, c=0.75, inverse=False, concentrate=1, return_weights=False,hard_cutoff=False):
    """
    Performs Global Weighted Rank Pooling (GWRP). The weights decay exponentially so that
    the top N voxels receive c% of the total weight.
    Ps: the raw weight at voxel N will be 1-c. 
    So, the inverse weight will be c.
    
    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W, D).
        N (int or torch.Tensor): Number of top voxels to concentrate. If an integer, a scalar
                                 value is used; if a tensor of shape (B, C), each (B,C) pair 
                                 uses its own N.
        c (float): Fraction (e.g. 0.9 for 90%) of the total weight to be concentrated in the top N voxels.
    
    Returns:
        torch.Tensor: The pooled tensor of shape (B, C).
    """
    reduce=False
    if len(x.shape)==3:
        x = x.unsqueeze(0).unsqueeze(0)
        reduce=True
    assert len(x.shape) == 5, f"Input tensor should be 5D, got {x.shape}"

    B, C, H, W, D = x.shape
    L = H * W * D  # total number of voxels per (B, C)
    
    # Sort the spatial elements in descending order.
    x_sorted, sort_indices = torch.sort(x.view(B, C, L), dim=-1, descending=True)
    
    # Compute the decay factor d.
    # If N is a scalar, convert it to a tensor of shape (B, C) with that constant.
    if not torch.is_tensor(N):
        N_tensor = torch.full((B, C), N, dtype=torch.float32, device=x.device)
    else:
        N_tensor = N.to(x.device).float()
    # Ensure N is at least 1.
    N_tensor = torch.clamp(N_tensor, min=1)
    
    # Compute d elementwise: d = (1-c)^(1/N).
    d = (1 - c) ** (1.0 / N_tensor)  # shape (B, C)
    # Reshape d to (B, C, 1) so it can broadcast.
    d = d.unsqueeze(-1)
    
    # Create an index tensor of shape (1, 1, L).
    indices = torch.arange(L, dtype=torch.float32, device=x.device).view(1, 1, L)
    
    # Compute weights: each weight is d^(i), broadcasting over (B, C).
    weights_raw = d ** indices  # shape (B, C, L)
    weights = weights_raw / weights_raw.sum(dim=-1, keepdim=True)  # normalize to sum to 1

    #assert that, for a random B,C element, the sum of the first N weights is equal to c
    #rand_b=torch.randint(0,B,(1,))
    #rand_c=torch.randint(0,C,(1,))
    #summed = weights[rand_b, rand_c, :int(N_tensor[rand_b, rand_c].item())].sum()  
    #assert abs(summed.item() - c) < 0.2

    if inverse:
        # For the inverse case we want to ignore the top N voxels.
        # Create a mask that is 0 for indices < N and 1 for indices >= N.
        mask_inv = (indices >= N_tensor.unsqueeze(-1)).float()  # shape (B, C, L)
        # Use the complementary weights for the background: here we use (1 - weights_raw)
        weights = mask_inv * (1 - weights_raw)
        # Note: We do not normalize these weights to sum to 1 because the goal here is to measure
        # the background (i.e. the voxels outside the top N).
    elif concentrate!=1:
        assert concentrate>1, 'concentrate must be greater than 1'
        # Create two masks: one for the top N voxels and one for the rest.
        mask_top = (indices < N_tensor.unsqueeze(-1)).float()      # 1 for indices < N, 0 otherwise
        mask_rest = (indices >= N_tensor.unsqueeze(-1)).float()     # 1 for indices >= N, 0 otherwise
        # Leave top N voxels unchanged and scale the rest by (1/concentrate)
        new_weights = mask_top * weights + mask_rest * (weights / concentrate)
        # Renormalize the weights so they sum to 1.
        weights = new_weights / new_weights.sum(dim=-1, keepdim=True)

    if return_weights:
        if hard_cutoff:
            #make all weights after N zero and re-normalize
            mask_top = (indices < N_tensor.unsqueeze(-1)).float()
            weights = mask_top * weights
            weights = weights / weights.sum(dim=-1, keepdim=True)
        # We need to return the weights reorganized into the original spatial order.
        # sort_indices tells us, for each (B, C, i), which voxel in the unsorted order that value came from.
        # Compute the inverse permutation.
        inverse_indices = sort_indices.argsort(dim=-1)
        # unsort the weights so that they align with the original order.
        weights_unsorted = weights.gather(dim=-1, index=inverse_indices)
        # Reshape to original spatial dimensions.
        weights_unsorted = weights_unsorted.view(B, C, H, W, D)
        if reduce:
            weights_unsorted = weights_unsorted.squeeze(0).squeeze(0)
        return weights_unsorted
    
    # Compute weighted sum and normalize by the sum of weights.
    pooled = (x_sorted * weights).sum(dim=-1)

    return pooled


def save_tensor_as_nifti(tensor: torch.Tensor, filename: str):
    """
    Saves a torch tensor as a NIfTI file, assuming a voxel spacing of 1x1x1 mm.

    Args:
        tensor (torch.Tensor): A torch tensor of shape (H, W, D) or (1, H, W, D).
        filename (str): The output filename (should end with .nii or .nii.gz).
    """
    if 'nii.gz' not in filename:
        filename += '.nii.gz'
        
    assert len(tensor.squeeze(0).shape)==3, f"Input tensor should be 3D, got {tensor.shape}"

    # Ensure tensor is on CPU and convert to numpy array.
    np_array = tensor.detach().cpu().numpy()
    
    # If the tensor has an extra channel dimension, squeeze it.
    if np_array.ndim == 4 and np_array.shape[0] == 1:
        np_array = np_array.squeeze(0)
    
    # Create an identity affine (voxel sizes = 1 mm in all directions).
    affine = np.eye(4)
    
    # Create the NIfTI image and save.
    nifti_img = nib.Nifti1Image(np_array, affine)
    nib.save(nifti_img, filename)
    print(f"Saved NIfTI file to {filename}")
    
def insert_ball(out_spatial, best_center, diameter, margin):
    """
    Places a 'ball' of size diameter * (1 + margin) into out_spatial at the 3D coordinate best_center.
    The 3D ordering is assumed to be (z, y, x).
    """
    # 1) Build the ball kernel for insertion
    binary_ball_kernel = create_ball_kernel(diameter*(1+margin), gaussian=False)

    # 2) Prepare an empty volume with same shape as out_spatial
    masked_volume = torch.zeros_like(out_spatial)
    
    # 3) Extract shape in (z, y, x) order
    Z, Y, X = masked_volume.shape
    
    # 4) The kernel half-width
    d_half = binary_ball_kernel.shape[-1] // 2
    
    # 5) Unpack best_center as (cz, cy, cx)
    cz, cy, cx = best_center
    
    # 6) Compute overlap in Z dimension
    vol_z_min = max(0, cz - d_half)
    vol_z_max = min(Z, cz + d_half + 1)
    mask_z_min = 0 if cz - d_half >= 0 else -(cz - d_half)
    mask_z_max = mask_z_min + (vol_z_max - vol_z_min)

    # 7) Compute overlap in Y dimension
    vol_y_min = max(0, cy - d_half)
    vol_y_max = min(Y, cy + d_half + 1)
    mask_y_min = 0 if cy - d_half >= 0 else -(cy - d_half)
    mask_y_max = mask_y_min + (vol_y_max - vol_y_min)

    # 8) Compute overlap in X dimension
    vol_x_min = max(0, cx - d_half)
    vol_x_max = min(X, cx + d_half + 1)
    mask_x_min = 0 if cx - d_half >= 0 else -(cx - d_half)
    mask_x_max = mask_x_min + (vol_x_max - vol_x_min)

    # 9) Place the kernel region into masked_volume
    masked_volume[
        vol_z_min:vol_z_max,
        vol_y_min:vol_y_max,
        vol_x_min:vol_x_max
    ] = binary_ball_kernel[
        mask_z_min:mask_z_max,
        mask_y_min:mask_y_max,
        mask_x_min:mask_x_max
    ]

    return masked_volume
def bbox_around_mask(mask: np.ndarray,
                     fallback: np.ndarray | None = None,
                     extra_ratio: float = 0.20,
                     min_size: int = 64
                     ) -> tuple[slice, slice, slice]:
    """
    3‑D bounding box tightly around **mask**, enlarged by *extra_ratio*,
    and then *grown* so that every edge is at least *min_size* voxels
    long (when the full volume along that axis is ≥ *min_size*).

    If *mask* is empty and *fallback* is supplied the box is computed on
    *fallback* instead.  If both are empty the full volume is returned.
    """
    assert mask.ndim == 3
    work = mask if np.count_nonzero(mask) > 0 or fallback is None else fallback

    zz, yy, xx = np.where(work > 0)
    if zz.size == 0:                      # still empty → whole volume
        return slice(0, work.shape[0]), slice(0, work.shape[1]), slice(0, work.shape[2])

    zmin, zmax = zz.min(), zz.max()
    ymin, ymax = yy.min(), yy.max()
    xmin, xmax = xx.min(), xx.max()

    def expand(lo: int, hi: int, length: int) -> tuple[int, int]:
        """
        • (*lo*,*hi*) are inclusive indices of the tight box.
        • First enlarge by *extra_ratio*.
        • Then grow to *min_size* if the axis is long enough.
        • Returned indices are **Python‑slice style**:  [lo, hi)  (hi exclusive).
        """
        size   = hi - lo + 1
        margin = int(round(size * extra_ratio / 2))

        lo -= margin
        hi += margin

        # convert to slice conventions (hi exclusive)
        hi += 1

        # clamp to volume
        lo = max(0, lo)
        hi = min(length, hi)

        # enforce minimum size
        cur_len = hi - lo
        target  = min_size if length >= min_size else cur_len
        if cur_len < target:
            # how much to add?
            need = target - cur_len
            add_lo = need // 2
            add_hi = need - add_lo
            lo = max(0, lo - add_lo)
            hi = min(length, hi + add_hi)

            # if one side hit the wall, extend on the other
            cur_len = hi - lo
            if cur_len < target:
                if lo == 0:
                    hi = min(length, target)
                elif hi == length:
                    lo = max(0, length - target)

        return lo, hi

    z0, z1 = expand(zmin, zmax, work.shape[0])
    y0, y1 = expand(ymin, ymax, work.shape[1])
    x0, x1 = expand(xmin, xmax, work.shape[2])

    return slice(z0, z1), slice(y0, y1), slice(x0, x1)

def ball_mask(lesion, organ, tumor_volumes, tumor_diameters, 
              apply_dice_loss=True,
              diameter_margin=0.5, volume_margin=0.5, gaussian=True, 
              gaussian_std=1.5, dilation_for_background=7,
              subseg_dilation=31,input_tensor=None, 
              standard_ce=False, 
              single_class=False):
    """
    This function is for using in postprocessing. It will produce an agreement score between the segmentation and the report, 
    and give an improved pseudo-mask. This is useful in active learning, to annotate a dataset with radiologists.
    Args:
    lesion: output raw mask (probabilities) for a lesion. Accepts one channel only. 
    organ: binary organ mask, one channel only. This is used to define the subsegment.
    tumor_diameter is a tensor of size B,T,3, batch, number of tumors in the crop, and 3 diameters
    diameter_margin: how much much we want the ball diameter to be bigger than the maximum tumor diameter
    gaussian: if a gaussian kernel is used in the ball convolution for better centering on the tumor
    gaussian_std: the higher, the smaller the difference between the ball kernel center and border values
    gwrp: wether to use GRWP to average each BCE loss. If so, more weight is given to increasing high confidence voxels.
    dilation_for_background: we apply a dilation kernel of this size to the tumor pseudo-mask, and define everything outside this mask as background, and use BCE loss to make the backgropund 0
    subseg_dilation: how much we dilate the tumor subsegment. Radiologists/AI may not be super precise when defining the subsegment, and tumors may grow out of organs, so we add a generous margin here.
    standard_ce: if True, we use a standard averaging for the BCE loss. Otherwise, we acerage the foreground and background voxel losses separately, and then sum the two losses.
    class_weights: optional 5D tensor to apply class weights. This is useful when dealing with imbalanced positives and negatives per class or datasets with many classes.
    Important: this loss assumes the output resolution is 1x1x1 mm, and that diammeters are in mm and volumes in mm^3. If the resolution is different, you should adjust the diameters and volumes accordingly or introduce a scaling factor.
    """
    global counter3
    out=lesion
    chosen_segment_mask = organ
    class_weights = None
    sigmoid = False #should have been already applied to the output
    gwrp=False
    gwrp_concentration=0.5

    #total tumor volume from the report
    #print('Volume in reports:', tumor_volumes)
    if len(out.shape)==3:
        out = out.unsqueeze(0)
    if len(chosen_segment_mask.shape)==3:
        chosen_segment_mask = chosen_segment_mask.unsqueeze(0)
    if len(out.shape)==4:
        out = out.unsqueeze(0)
    if len(chosen_segment_mask.shape)==4:
        chosen_segment_mask = chosen_segment_mask.unsqueeze(0)
        
    chosen_segment_mask = (chosen_segment_mask>0.5).float()
    
    assert len(tumor_volumes.shape) == 2 #batch and maximum of 10 tumors
    assert len(out.shape) == 5
    assert chosen_segment_mask.shape == out.shape
    

    #get only the channels with lesions
    assert out.shape[1] == 1, 'Output should have only one channel, the lesion channel'
    assert chosen_segment_mask.shape[1] == 1, 'organ should have only one channel, the lesion organ channel'
    unk_voxels = torch.zeros_like(out)
    labels = torch.zeros_like(out)

    chosen_segment_mask = dilate_volume(chosen_segment_mask,subseg_dilation)
    #dilate the unk voxels
    to_penalize = torch.ones_like(out)


    #let's get only the subsegment voxels
    assert out.shape == chosen_segment_mask.shape

    losses = []
    losses_dice = []
    refined_masks = []

    for B in range(out.shape[0]):#batch itens
        #assert diameters and violumes make sense
        assert torch.equal(tumor_diameters[B].sum(-1)>0, tumor_volumes[B]>0), f'Tumor diameters and volumes should be consistent, got {tumor_diameters[B]} and {tumor_volumes[B]}'
        
        #get correct batch and class
        x = out[B]
        tumor_seg = chosen_segment_mask[B]
        #current_x is still 4 D, with one class per tumor type. Assert at most one of these channels is non-zero (due to the chosen_segment_mask):
        assert (tumor_seg.sum((-1,-2,-3))>0).float().sum()<=1, f'Only one channel should be non-zero, got {tumor_seg.sum((-1,-2,-3))}'
        
        # if no tumor in this batch, create a zero pseudo label
        if tumor_seg.sum()==0 or tumor_volumes[B].sum()==0:
            # no tumor in this batch, create a zero pseudo label
            pseudo_mask = torch.zeros_like(x)
            if sigmoid:
                if not single_class:
                    #standard, use sigmoid
                    loss = F.binary_cross_entropy_with_logits(x, pseudo_mask, reduction='none')
                else:
                    #use softmax
                    loss = F.cross_entropy(x, pseudo_mask, reduction='none')
                #print('ball loss uses BCE with logits')
            else:
                if not single_class:
                    #assert x is in the range 0-1
                    assert (x>=0).all() and (x<=1).all(), f'Output is not in the range 0-1, its min is: {x.min()}, its max is: {x.max()}'
                    #assert pseudo_mask is in the range 0-1
                    assert (pseudo_mask>=0).all() and (pseudo_mask<=1).all(), f'Pseudo mask is not in the range 0-1, its min is: {pseudo_mask.min()}, its max is: {pseudo_mask.max()}'
                    loss = F.binary_cross_entropy(x, pseudo_mask, reduction='none')
                else:
                    #single class, but consider that softmax was already applied. Thus, use nll loss
                    #from one-hot to class indices: argmax
                    loss = F.nll_loss(x, pseudo_mask.argmax(dim=1), reduction='none')
            assert loss.shape == tumor_seg.shape
            loss = loss * to_penalize[B]
            if class_weights is not None:
                # apply class weights if provided
                loss = loss * class_weights[B]
            loss = loss.mean()
            if apply_dice_loss:
                if class_weights is not None:
                    w = class_weights[B]
                else:
                    w = None
                dice_loss = DiceLossMultiClass(preds=x, targets=pseudo_mask, known_voxels=to_penalize[B],sigmoid=sigmoid, class_weights=w).mean()
                losses_dice.append(dice_loss)
            losses.append(loss.mean())
            continue
        
        #get tumor class
        for c in range(x.shape[0]):
            if tumor_seg[c].sum()>0:
                x = x[c]
                penalize = to_penalize[B][c]
                if class_weights is not None:
                    c_weight = class_weights[B][c] #get the class weights for this batch and class
                else:
                    c_weight = None
                break
        tumor_seg = tumor_seg.sum(0)
        current_tumor_diameters = tumor_diameters[B]
        current_tumor_volumes = tumor_volumes[B]

        # Get the sort indices for tumor_volumes in descending order
        sorted_indices = torch.argsort(current_tumor_volumes, descending=True)

        # Filter indices to keep only those with volume > 0
        sorted_indices = sorted_indices[current_tumor_volumes[sorted_indices] > 0]
        #print('--------Sorted indices:', sorted_indices)
        #print('--------SORTED VOLUMES:', current_tumor_volumes[sorted_indices])
        #print('--------UNSORTED VOLUMES:', current_tumor_volumes)

        #Create the pseudo-mask
        pseudo_masks = []
        pseudo_masks_small = []
        pseudo_masks_big = []
        
        #update x for the next tumor: remove pseudo_mask, so that this tumor is not selected again.
        if sigmoid:
            x_iter = torch.sigmoid(x)*tumor_seg
        else:
            x_iter = x*tumor_seg
        for tumor_idx in sorted_indices:
            vol=current_tumor_volumes[tumor_idx].item()
            dia=current_tumor_diameters[tumor_idx]
            #get the maximum diameter
            max_diameter = torch.max(dia).item()
            assert max_diameter>0, f'Tumor diameter should be larger than 0, got {max_diameter}'
            assert vol>0, f'Tumor volume should be larger than 0, got {vol}'
            if vol==0 or max_diameter == 0:
                print('Found 0 tumor where it should not be')
                continue
            if max_diameter <= 1:
                print('Found 1mm diameter, increasing to 3')
                max_diameter = 3
            if vol <= 1:
                print('Found 1mm volume, increasing to 9')
                vol = 9
            #assert it is not zero
            #ball convolution: use isolate_tumor to get the top 'tumor_volume' voxels in the outpus, inside the best fitting ball position
            pseudo_mask,pseudo_mask_small,pseudo_mask_big = isolate_tumor(x_iter, diameter=max_diameter, 
                                                                          gaussian=gaussian, gaussian_std=gaussian_std, tumor_volume=vol,
                                                                          diameter_margin=diameter_margin,volume_margin=volume_margin)
            
            pseudo_masks_small.append(pseudo_mask_small)
            pseudo_masks_big.append(pseudo_mask_big)
            
            #which mask to keep? 
            #if the thresholded mask at 0.5 is between the pseudo_mask_small and pseudo_mask_big, use the thresholded mask
            #otherwise, use pseudo_mask
            
            th_mask = (x_iter > 0.5).float()
            diff_big = torch.clamp((pseudo_mask_big - th_mask), min=0) #this should be zero, or the mask is larger than pseudo_mask_big
            diff_big = diff_big.sum() == 0 #this should be true, or the mask is larger than pseudo_mask_big
            diff_small = th_mask.sum() >= pseudo_mask_small.sum() #this should be true, or the mask is smaller than pseudo_mask_small
            if diff_big and diff_small:
                #use the thresholded mask
                pseudo_masks.append(th_mask)
            else:
                #use the pseudo_mask
                pseudo_masks.append(pseudo_mask)
            
            x_iter = x_iter * (1 - pseudo_mask) #remove the pseudo mask from the output, so that it is not selected again
        #stack the pseudo masks
        pseudo_mask = torch.stack(pseudo_masks_small).sum(0)
        pseudo_mask = (pseudo_mask > 0).float()
        dilated_pseudo_mask = torch.stack(pseudo_masks_big).sum(0)
        dilated_pseudo_mask = (dilated_pseudo_mask > 0).float()
        
        pseudo_mask_refined = torch.stack(pseudo_masks).sum(0)
        pseudo_mask_refined = (pseudo_mask_refined > 0).float()
        refined_masks.append(pseudo_mask_refined)
        

        #we can add a tolerance margin around the pseudo mask, where we do not penalize the outputs for not being zero
        if dilation_for_background>0:
            dilated_pseudo_mask=dilate_volume(dilated_pseudo_mask, dilation_for_background)
            
        border = dilated_pseudo_mask - pseudo_mask
        #threshold at 0
        border = (border > 0).float()
            
        penalize=penalize * (1 - border)
        #penalize is a tensor with the voxels where we want to apply our losses to here

        #BCE loss with mask
        if sigmoid:
            if not single_class:
                BCE = F.binary_cross_entropy_with_logits(x, pseudo_mask, reduction='none')
            else:
                #single class
                BCE = F.cross_entropy(x, pseudo_mask, reduction='none')
        else:
            if not single_class:
                #assert x is in the range 0-1
                assert (x>=0).all() and (x<=1).all(), f'Output is not in the range 0-1, its min is: {x.min()}, its max is: {x.max()}'
                #assert pseudo_mask is in the range 0-1
                assert (pseudo_mask>=0).all() and (pseudo_mask<=1).all(), f'Pseudo mask is not in the range 0-1, its min is: {pseudo_mask.min()}, its max is: {pseudo_mask.max()}'
                BCE = F.binary_cross_entropy(x, pseudo_mask, reduction='none')
            else:
                #single class, but consider that softmax was already applied. Thus, use nll loss
                #from one-hot to class indices: argmax
                BCE = F.nll_loss(x, pseudo_mask.argmax(dim=1), reduction='none')
        assert (penalize.shape==BCE.shape), f'To penalize and BCE should have the same shape, got {penalize.shape} and {BCE.shape}'
        BCE = BCE * penalize #cut the loss gradient in the border. Remember that unk voxels were already removed from x

        #dice loss
        #dice loss
        if apply_dice_loss:
            #remove tumor surroundings, to avoid penalizing them: we are not super sure if this region is tumor or not.
            dice_loss = DiceLossMultiClass(preds=x, targets=pseudo_mask, known_voxels=penalize,sigmoid=sigmoid,class_weights=c_weight)
            if sigmoid:
                print('Dice loss:',dice_loss, 'Mean prediction:',torch.sigmoid(x).mean())
            else:
                print('Dice loss:',dice_loss, 'Mean prediction:',x.mean())
            #we make all voxels knwon because we alreay removed unknown voxels from x
            #print('Using dice loss inside the ball loss')

        if not standard_ce:
            #we separate foreground and background, calculate the average per-voxel loss for them separatelly, than sum it. We can use GRWP in the foreg. or not.
            if gwrp:
                #we do BCE for the entire channel, but we do not simply average it. We can use GWRP to average the tumor values (positive GT)
                #we add the pseudo-mask to boost its voxels values and concentrate GWRP there.
                assert pseudo_mask.sum() > 0, f'Pseudo mask should have at least one voxel, got {pseudo_mask.sum()}, volume is {vol} and diameter is {max_diameter}'
                if sigmoid:
                    foreg_weights = GlobalWeightedRankPooling(torch.sigmoid(x)*pseudo_mask+pseudo_mask, N=pseudo_mask.sum(), c=gwrp_concentration,return_weights=True,
                                                                hard_cutoff=True)
                else:
                    foreg_weights = GlobalWeightedRankPooling(x*pseudo_mask+pseudo_mask, N=pseudo_mask.sum(), c=gwrp_concentration,return_weights=True,
                                                                hard_cutoff=True)
                #print highest and lowest non-zero values in foreg_weights
                assert foreg_weights.sum() > 0.95 and foreg_weights.sum() < 1.05, f'GWRP weights should be normalized to 1, got {foreg_weights.sum()}'
                #renormlize gwrp weights so they sum to pseudo_mask.sum()
                foreg_weights = foreg_weights * pseudo_mask.sum()
                #print('GWRP Foreg weights range:', foreg_weights[foreg_weights>0].max(), foreg_weights[foreg_weights>0].min())
                #assert sum of foreg_weights is close to 1
                foreg_weights = foreg_weights*pseudo_mask
                assert BCE.shape == foreg_weights.shape, f'BCE and GWRP weights should have the same shape, got {BCE.shape} and {foreg_weights.shape}'
                loss_foreground = (BCE*foreg_weights)#.mean() #we can use mean here because 
            else:
                #print('Using simple mean for BCE loss')
                loss_foreground = (BCE*pseudo_mask)#.mean()
            
            #Background:
            bkg_weights = 1 - dilated_pseudo_mask
            loss_background = (BCE*bkg_weights)#.mean()
            
            if c_weight is not None:
                # apply class weights to the BCE loss
                assert len(c_weight.shape) == len(loss_background.shape), f'Class weights shape {c_weight.shape} does not match BCE shape {BCE.shape}'
                assert c_weight.shape[0] == loss_background.shape[0], f'Class weights {class_weights[B].shape} do not match loss_background shape {loss_background.shape}'
                loss_foreground = loss_foreground * c_weight
                loss_background = loss_background * c_weight
            loss_foreground = loss_foreground.mean()
            loss_background = loss_background.mean()

            loss = loss_foreground + loss_background
            losses.append(loss)#BCE loss
        else:
            #print('Using standard CE for BCE loss')
            if c_weight is not None:
                # apply class weights to the BCE loss
                assert len(c_weight.shape) == len(BCE.shape), f'Class weights shape {c_weight.shape} does not match BCE shape {BCE.shape}'
                assert c_weight.shape[0] == BCE.shape[0], f'Class weights {c_weight.shape} do not match BCE shape {BCE.shape}'
                BCE = BCE * c_weight
            BCE = BCE.mean()
            losses.append(BCE)#simple mean.

        if apply_dice_loss:
            losses_dice.append(dice_loss.mean())

        if counter3<10:

            counter3+=1
            os.makedirs('SanityBallLoss/'+str(counter3), exist_ok=True)
            if sigmoid:
                save_tensor_as_nifti(torch.sigmoid(x),'SanityBallLoss/'+str(counter3)+'/x')
            else:
                save_tensor_as_nifti(x,'SanityBallLoss/'+str(counter3)+'/x')
            save_tensor_as_nifti(pseudo_mask,'SanityBallLoss/'+str(counter3)+'/pseudo_mask')
            save_tensor_as_nifti(border,'SanityBallLoss/'+str(counter3)+'/border')
            save_tensor_as_nifti(tumor_seg,'SanityBallLoss/'+str(counter3)+'/tumor_segment')
            save_tensor_as_nifti((to_penalize[B].sum(0)>0).float(),'SanityBallLoss/'+str(counter3)+'/to_penalize')
            if input_tensor is not None:
                save_tensor_as_nifti(input_tensor[B].squeeze(),'SanityBallLoss/'+str(counter3)+'/input_volume')

            #save tumor volumes and diameters as yaml
            with open('SanityBallLoss/'+str(counter3)+'/tumor_volumes.yaml', 'w') as file:
                yaml.dump(tumor_volumes.tolist(), file)
            with open('SanityBallLoss/'+str(counter3)+'/tumor_diameters.yaml', 'w') as file:
                yaml.dump(tumor_diameters.tolist(), file)
            print('Saved to '+ 'SanityBallLoss/'+ str(counter3)+'/known_voxels')
            l=losses[-1].item()
            if apply_dice_loss:
                l+=losses_dice[-1].item()
            if sigmoid:
                info=f'Volume in output: {torch.sigmoid(x).sum().item()}, Volume in report: {vol}, Loss: {l}'
            else:
                info=f'Volume in output: {x.sum().item()}, Volume in report: {vol}, Loss: {l}'
            print(info)
            #save the loss as yaml
            with open('SanityBallLoss/'+str(counter3)+'/loss.yaml', 'w') as file:
                yaml.dump(l, file)
            #save the info as yaml
            with open('SanityBallLoss/'+str(counter3)+'/info.yaml', 'w') as file:
                yaml.dump(info, file)
            print('Saved to '+ 'SanityBallLoss/'+ str(counter3)+'/loss.yaml')

    
    refined_masks = torch.stack(refined_masks, dim=0)
    
    L = torch.stack(losses).mean()        # your avg CE loss (in nats)
    x = (L/torch.log(torch.tensor(2.0)))  # normalize so random=1
    # now x=0 --> perfect, x=1 --> random, x-->∞ --> terrible
    #   we want when x=0 → quality ~1, when x=1 → quality=0.5
    #   we can do quality = σ(−k*(x−1/2)) for some steepness k
    k = 1.0
    score = torch.sigmoid(-k*(x - 0.5))
    if apply_dice_loss:
        score += (1-torch.stack(losses_dice).mean())
        score = score/2
        
    score = score*10 # scale to 0-10 range
        
    #refined_masks = denoise_keep_edges(refined_masks)
    
    retur = {'ball_loss_bce':torch.stack(losses).mean(),
            'ball_loss_dice':torch.stack(losses_dice).mean() if apply_dice_loss else torch.zeros_like(torch.stack(losses).mean()),
            'refined_masks':refined_masks,
            'score': score}
    
    #print(retur,flush=True)
    print(f'Score: {score.item():.4f}, BCE loss: {retur["ball_loss_bce"].item():.4f}, Dice loss: {retur["ball_loss_dice"].item():.4f}', flush=True)
    
    return retur


# -------------------------------------------------------------------
# helper – robustly locate the binary organ mask inside “predictions/”
# -------------------------------------------------------------------
ORG_MASK_ALIASES: dict[str, str] = {
    # on‑disk name →          logical organ
    "prostate":      "uterus",        # uterus stored as prostate.nii.gz
    "gall_bladder":  "gallbladder",   # gallbladder stored as gall_bladder.nii.gz
}
INV_ALIASES = {v: k for k, v in ORG_MASK_ALIASES.items()}

# Bilateral organs are predicted as two files (left + right) with no single
# merged file on disk.  Map the logical organ → candidate on‑disk base names
# (each expanded with ``_left`` / ``_right``).  The first base that yields any
# component file wins; its left/right parts are OR‑merged by the caller.
BILATERAL_ORGAN_BASES: dict[str, list[str]] = {
    "kidney":        ["kidney"],
    "adrenal":       ["adrenal_gland", "adrenal"],
    "adrenal_gland": ["adrenal_gland", "adrenal"],
}


def resolve_organ_mask_files(pred_dir: Path, organ: str) -> list[Path]:
    """
    Return the list of *existing* files that together make up the mask for
    ``organ`` inside ``pred_dir`` (folder that contains the binary .nii.gz masks).

    1. Try the canonical single file  ``<organ>.nii.gz``;
    2. Try any known alias on disk (e.g. ``gall_bladder.nii.gz``);
    3. For bilateral organs (kidney, adrenal gland), gather the left/right
       component files (e.g. ``kidney_left.nii.gz`` + ``kidney_right.nii.gz``)
       so the caller can merge them;
    4. Otherwise raise ``FileNotFoundError`` (never returns an empty list).
    """
    # canonical single file
    cand = pred_dir / f"{organ}.nii.gz"
    if cand.exists():
        return [cand]

    # alias → physical filename
    alias = INV_ALIASES.get(organ)
    if alias:
        cand = pred_dir / f"{alias}.nii.gz"
        if cand.exists():
            return [cand]

    # bilateral organ → left/right component files
    for base in BILATERAL_ORGAN_BASES.get(organ, []):
        found = [pred_dir / f"{base}_{side}.nii.gz"
                 for side in ("left", "right")
                 if (pred_dir / f"{base}_{side}.nii.gz").exists()]
        if found:
            return found

    raise FileNotFoundError(
        f"Mask for organ '{organ}' not found in {pred_dir}. "
        "Checked canonical name, known aliases, and bilateral (left/right) parts."
    )


def resolve_organ_mask(pred_dir: Path, organ: str) -> Path:
    """Back‑compat single‑file resolver; returns the first component file."""
    return resolve_organ_mask_files(pred_dir, organ)[0]


def load_organ_mask(pred_dir: Path, organ: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load ``organ``'s binary mask as float32, OR‑merging bilateral component
    files (kidney, adrenal gland) into a single volume.

    Returns ``(mask, affine)`` where ``affine`` comes from the first component.
    """
    files = resolve_organ_mask_files(pred_dir, organ)
    img = nib.load(str(files[0]))
    mask = img.get_fdata().astype("float32")
    for extra in files[1:]:
        mask = np.maximum(mask, nib.load(str(extra)).get_fdata().astype("float32"))
    return mask, img.affine
    
import os, fcntl, math              # make sure these imports are present
from csv import DictWriter
from contextlib import contextmanager

# ────────────────────  CSV append with file‑locking  ────────────────────
@contextmanager
def _csv_lock(lockfile: Path):
    """
    Exclusive lock on *lockfile* (Path with ".lock" suffix).
    Several GPU processes can safely append to the same CSV.
    """
    fd = lockfile.open("w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()

def append_row(csv_path: Path, row: dict) -> None:
    """
    Atomically append *row* (a dict) to *csv_path*.
    Header is written automatically the first time.
    """
    lock_path = csv_path.with_suffix(csv_path.suffix + ".lock")
    with _csv_lock(lock_path):                 # ← wait for exclusive access
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with csv_path.open("a", newline="") as f:
            w = DictWriter(f, fieldnames=row.keys())
            if write_header:
                w.writeheader()
            w.writerow(row)
# ────────────────────────────  main  ──────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Filter predictions + collect bundles")
    ap.add_argument("--pred_root",    required=True, type=Path)
    ap.add_argument("--meta",         type=Path, default="/projects/bodymaps/Data/metadata_ucsf_batch_1_to_6_and_merlin.csv")
    ap.add_argument("--reports",      type=Path, default="/projects/bodymaps/Data/metadata_per_tumor_ucsf_batch_1_to_6_and_merlin.csv")
    ap.add_argument("--classes",      default=",".join(STD_ORGANS))
    ap.add_argument("--id-column",    default=None)
    ap.add_argument("--out",          type=Path, default="kept_ids.txt")
    ap.add_argument("--output_folder",required=True, type=Path)
    ap.add_argument("--source_ct",    required=True, type=Path)
    ap.add_argument("--debug",        action="store_true")
    ap.add_argument("--overwrite",    action="store_true",
                    help="re‑process IDs already listed in summary.csv")
    # --------------- new flags ---------------------------------------
    ap.add_argument("--gpu",   default="0",
                    help="GPU id visible to this process (CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--parts", type=int, default=1,
                    help="split workload into this many equal parts")
    ap.add_argument("--part",  type=int, default=0,
                    help="index (0‑based) of the part to process")
    ap.add_argument("--ids", type=str, default=None, help='Path to csv with BDMAP ID column and list of ids to process')
    ap.add_argument("--accept_binary", action="store_true", help="If predictions_raw is missing/empty, accept binary lesion masks from predictions/ as fallback")
    args = ap.parse_args()

    # choose GPU *before* importing torch.cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    try:
        torch.cuda.set_device(0)
    except:
        pass
    

    # ── load reports ────────────────────────────────────────────────
    global reports_df
    reports_df = pd.read_csv(args.reports, low_memory=False)

    # ── meta & filtering (same logic as your “last good” version) ───
    allowed = {canon(o) for o in args.classes.split(",") if o.strip()}
    if not allowed:
        sys.exit(" --classes resolved to an empty set")

    meta = pd.read_csv(args.meta, low_memory=False)
    id_col = "BDMAP_ID"
    if "BDMAP ID" in meta.columns:
        meta.rename(columns={'BDMAP ID': 'BDMAP_ID'}, inplace=True)
    if "BDMAP ID" in reports_df.columns:
        reports_df.rename(columns={'BDMAP ID': 'BDMAP_ID'}, inplace=True)
    if id_col not in meta.columns:
        sys.exit(f" no '{id_col}' column in {args.meta}")
    meta = meta.drop_duplicates(subset=[id_col])
    meta.set_index(id_col, inplace=True)

    kept, id_to_rawdir = {}, {}
    id_to_lesiondir = {}
    rejected = {
        "missing_pred_raw": 0,
        "empty_pred_raw": 0,
        "fallback_disallowed": 0,   # raw missing/empty but --accept_binary not set
        "fallback_used": 0,         # accepted fallback from predictions/
        "missing_lesion_files": 0,  # neither dir provides lesion files
        "extra_organs": 0,
        "report_mismatch": 0
    }

    tumor_num = {canon(o): 0 for o in allowed}
    mismatch_problems: Counter = Counter()  # aggregates reports_agree problems for diagnostics


    for id_dir in sorted(p for p in args.pred_root.iterdir() if p.is_dir()):
        bid_raw = id_dir.name
        bid = bid_raw[:len("BDMAP_00032591")]
        pred_raw = id_dir / "predictions_raw"
        pred_bin = id_dir / "predictions"
        # decide lesion source directory
        use_raw = pred_raw.exists() and any(pred_raw.iterdir())
        if use_raw:
            lesion_dir = pred_raw
        else:
            # track why raw is unusable
            if not pred_raw.exists():
                rejected["missing_pred_raw"] += 1
            else:
                empty_pred_raw.append(bid)
                rejected["empty_pred_raw"] += 1

            # fallback requires explicit opt-in
            if not args.accept_binary:
                rejected["fallback_disallowed"] += 1
                continue

            lesion_dir = pred_bin
            rejected["fallback_used"] += 1

        # verify lesion_dir exists and has lesion files
        if (not lesion_dir.exists()):
            rejected["missing_lesion_files"] += 1
            continue

        row = meta.loc[bid]
        organs,_ = organs_from_metadata_row(row, allowed, min_lesions=1)
        if not organs:
            rejected["missing_lesion_files"] += 1
            continue

        # keep only allowed organs
        organs = organs & allowed

        if bid not in meta.index:
            raise ValueError(f"{bid} not in metadata")

        ok, problems = reports_agree(meta.loc[bid], organs)
        if not ok:
            rejected["report_mismatch"] += 1
            mismatch_problems.update(problems)
            continue

        kept[bid] = organs
        id_to_rawdir[bid] = id_dir
        id_to_lesiondir[bid] = lesion_dir

        for o in organs:
            tumor_num[o] += 1

    args.out.write_text("\n".join(kept))
    print(f'Number of files found: {len(kept)}')
    print("Rejection summary:")
    for k, v in rejected.items():
        print(f"  {k:20s}: {v}")
    print(f"  kept                : {len(kept)}")
    print(f"Empty pred_raw dir: {empty_pred_raw}")

    # ── diagnostics for report_mismatch ──────────────────────────────
    # If reports_agree rejected any rows, print the top missing-column messages
    # and a sample of the meta columns that look related, so it's obvious when
    # --meta points at a CSV whose headers don't match column_from_organ().
    if rejected["report_mismatch"] > 0:
        print(f"\n[report_mismatch diagnostics]  --meta = {args.meta}")
        print(f"  reports_agree rejected {rejected['report_mismatch']} rows. Top problems:")
        for msg, n in mismatch_problems.most_common(10):
            print(f"    {n:6d}x  {msg}")
        related = [c for c in meta.columns
                   if "lesion" in c.lower() or c.lower().startswith("number of")]
        print(f"  Meta columns containing 'lesion' or starting with 'number of' "
              f"({len(related)} found):")
        for c in related[:20]:
            print(f"    - {c!r}")
        if not related:
            print("    (none — this --meta CSV has no lesion-count columns at all; "
                  "you are probably passing the wrong metadata file)")



    if args.ids is not None:
        ids_df = pd.read_csv(args.ids)
        id_set = set(ids_df['BDMAP ID'])

        # rebuild `kept` with only those entries whose key lives in id_set
        kept = {k: v for k, v in kept.items() if k in id_set}

    # ── honour summary.csv / overwrite flag ─────────────────────────
    csv_path = args.output_folder / "summary.csv"
    if csv_path.exists() and not args.overwrite:
        already = set(pd.read_csv(csv_path)["BDMAP_ID"].astype(str))
        kept = {k: v for k, v in kept.items() if k not in already}

    if not kept:
        print("Nothing found to predict!")
        return


    # ── split workload across parts ---------------------------------
    if args.parts < 1:
        sys.exit("--parts must be ≥1")
    if not 0 <= args.part < args.parts:
        sys.exit(f"--part must be in 0..{args.parts-1}")

    all_ids    = sorted(kept.keys())
    chunk_size = math.ceil(len(all_ids) / args.parts)
    s, e       = args.part * chunk_size, min(len(all_ids), (args.part + 1) * chunk_size)
    ids_this   = all_ids[s:e]
    kept       = {k: kept[k] for k in ids_this}
    print(f"GPU {args.gpu} processing part {args.part}/{args.parts-1}, "
          f"{len(kept)} IDs")

    # ── debug mode ──────────────────────────────────────────────────
    if args.debug:
        if args.output_folder.exists() and any(args.output_folder.iterdir()):
            if input("⚠️  delete ALL in output folder? type 'yes': ").strip().lower() != "yes":
                print("Aborted"); sys.exit(0)
            shutil.rmtree(args.output_folder)
        kept = dict(random.sample(list(kept.items()), min(10, len(kept))))

    # ── bundle creation ─────────────────────────────────────────────
    args.output_folder.mkdir(parents=True, exist_ok=True)

    for bid in tqdm(kept.keys(), desc="Creating bundles"):
        organs_here = kept[bid]
        root_dir    = id_to_rawdir[bid]
        dst         = args.output_folder / bid
        dst.mkdir(parents=True, exist_ok=True)

        # ---------- extra report info ------------------------------
        labels_dict, _ = get_tumor_segment_labels(bid)
        unks = labels_dict["organs_with_unk_tumor_size"]

        # ---------- ball‑mask score per organ & save refined masks -
        organ_scores   = []
        tumors_per_org = {}

        for organ in sorted(organs_here):
            # ---------- lesion probability ------------------------
            lesion_dir = id_to_lesiondir[bid]  # predictions_raw if available, else predictions (only if --accept_binary)

            base = organ.replace("pancreas", "pancreatic")
            lesion_path_nii = lesion_dir / f"{base}_lesion.nii.gz"
            lesion_path_npz = lesion_dir / f"{base}_lesion.npz"

            if lesion_path_nii.exists():
                lesion_img = nib.load(str(lesion_path_nii))
                lesion_np  = lesion_img.get_fdata().astype("float32")
                affine     = lesion_img.affine
            elif lesion_path_npz.exists():
                lesion_np  = np.load(lesion_path_npz)["arr_0"].astype("float32")
                affine     = None
            else:
                raise FileNotFoundError(f"No lesion mask for {organ} in {lesion_dir} (bid={bid})")

            # ---------- organ mask (bilateral organs OR‑merged) ----
            organ_np, organ_affine = load_organ_mask(root_dir / "predictions", organ)
            if affine is None:
                affine = organ_affine

            # ---------- crop bounding box --------------------------
            organ_empty  = np.count_nonzero(organ_np) == 0
            mask_for_box = organ_np if not organ_empty else (lesion_np > 0)
            zsl, ysl, xsl = bbox_around_mask(mask_for_box, extra_ratio=0.20)

            lesion_crop = lesion_np[zsl, ysl, xsl]
            organ_crop  = organ_np[zsl, ysl, xsl] if not organ_empty \
                          else np.ones_like(lesion_crop, dtype=np.float32)

            lesion_t = torch.from_numpy(lesion_crop).cuda()
            organ_t  = torch.from_numpy(organ_crop ).cuda()

            #assert lesion_t.shape == organ_t.shape, f'Shape mismatch: lesion {lesion_t.shape}, {lesion_path_nii}; organ: {organ_t.shape}, {organ}; CT shape: {nib.load(str(args.source_ct / bid / "ct.nii.gz")).get_fdata().shape}, {str(args.source_ct / bid / "ct.nii.gz")}'

            out = estimate_tumor_volume(bid, organ)
            if isinstance(out,str):
                continue
            else:
                vols, diams = out
            tumor_count = sum(v > 0 for v in vols)
            tumors_per_org[organ] = tumor_count



            with torch.no_grad():
                bm_out = ball_mask(
                    lesion_t, organ_t,
                    tumor_volumes=torch.tensor(vols).unsqueeze(0).cuda(),
                    tumor_diameters=diams.unsqueeze(0).cuda()
                )

            organ_scores.append(bm_out["score"].item())

            # ---------- place refined mask back --------------------
            refined_crop = bm_out["refined_masks"][0].cpu().numpy().astype(np.uint8)
            refined_full = np.zeros_like(lesion_np, dtype=np.uint8)
            refined_full[zsl, ysl, xsl] = refined_crop

            ref_out = dst / f"refined_{organ.replace('pancreas','pancreatic')}_lesion_mask.nii.gz"
            nib.save(nib.Nifti1Image(refined_full, affine), str(ref_out))

        # ---------- write info.txt --------------------------------
        avg_score = np.mean(organ_scores)

        with (dst / "info.txt").open("w") as f:
            f.write("Tumour organs: " + ", ".join(sorted(organs_here)) + "\n")
            f.write(f"Lesion source: {id_to_lesiondir[bid].name}\n")
            f.write("Organs with UNKNOWN tumour size: " + ", ".join(sorted(unks)) + "\n")
            f.write(f"ball_mask score (avg over organs): {avg_score:.4f}\n\n")

            
            out = estimate_tumor_volume(bid, list(organs_here))
            if not isinstance(out,str):
                vols_all, diams_all = out
                f.write("Estimated tumour volumes (mm³) [max 10]:\n")
                f.write(", ".join([f"{v:.1f}" for v in vols_all]) + "\n\n")
                f.write("Estimated diameters (mm):\n")
                for row in diams_all:
                    f.write(", ".join([f"{d:.1f}" for d in row.tolist()]) + "\n")
                f.write("\nReport:\n")
                f.write(str(meta.loc[bid].get("report", "(report column missing)")))

        # ---------- copy predictions_raw (or fallback lesion masks) ----------
        src_pred_raw = root_dir / "predictions_raw"
        dst_pred     = dst / "predictions_raw"

        if not dst_pred.exists():
            if src_pred_raw.exists() and any(src_pred_raw.iterdir()):
                shutil.copytree(src_pred_raw, dst_pred)
            else:
                # raw missing/empty: only allowed if user opted in
                if not args.accept_binary:
                    raise RuntimeError(
                        f"predictions_raw missing/empty for {bid}, but --accept_binary not set. "
                        "This should have been filtered earlier."
                    )

                dst_pred.mkdir(parents=True, exist_ok=True)
                src_bin = root_dir / "predictions"

                for o in organs_here:
                    base = o.replace("pancreas", "pancreatic")

                    # prefer nii.gz if present, otherwise npz
                    cand_nii = src_bin / f"{base}_lesion.nii.gz"
                    cand_npz = src_bin / f"{base}_lesion.npz"

                    if cand_nii.exists():
                        shutil.copy(cand_nii, dst_pred / cand_nii.name)
                    elif cand_npz.exists():
                        shutil.copy(cand_npz, dst_pred / cand_npz.name)
                    else:
                        raise FileNotFoundError(
                            f"Fallback requested but no lesion mask found for organ '{o}' in {src_bin} (bid={bid})"
                        )

        # ---------- copy binary masks -----------------------------
        src_bin = root_dir / "predictions"
        for o in organs_here:
            in_files = resolve_organ_mask_files(src_bin, o)
            if len(in_files) == 1:
                shutil.copy(in_files[0], dst / f"{o}.nii.gz")
            else:
                # bilateral organ: OR‑merge left/right into one <organ>.nii.gz
                merged, aff = load_organ_mask(src_bin, o)
                nib.save(nib.Nifti1Image((merged > 0).astype(np.uint8), aff),
                         str(dst / f"{o}.nii.gz"))

        # ---------- copy CT ---------------------------------------
        src_ct = args.source_ct / bid / "ct.nii.gz"
        if src_ct.exists():
            shutil.copy(src_ct, dst / "ct.nii.gz")
        else:
            print(f"CT not found for {bid}")

        # ---------- append summary row (atomic) ---------------------
        row = {
            "BDMAP_ID":                    bid,
            "pseudo-mask score":           f"{avg_score:.4f}",
            "organs with tumors":          "; ".join(sorted(organs_here)),
            "number of tumors per organ":  "; ".join(f"{k} {v}" for k, v in tumors_per_org.items()),
            "report":                      str(meta.loc[bid].get("report", ""))
        }
        append_row(csv_path, row)          # uses the file‑lock helper

    # ── summary printout ────────────────────────────────────────────
    print(f"\n processed {len(ids_this)} IDs of part {args.part}/{args.parts-1}")
    print(f" kept list → {args.out}")
    print(f" bundles   → {args.output_folder}")
    print(f" summary   → {csv_path}")
    print(f"  new tumour counts: {tumor_num}")

if __name__ == "__main__":
    main()