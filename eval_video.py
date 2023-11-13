from glob import glob
import argparse
import cv2
import os
import os.path as osp
import numpy as np
import torch
from tqdm import tqdm
from scipy.io import loadmat
from collections import defaultdict
from piq import psnr, ssim
import subprocess
import pickle


def read_mp4(input_fn, to_rgb=False, to_gray=False, to_nchw=False):
    frames = []
    cap = cv2.VideoCapture(input_fn)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if to_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    cap.release()
    frames = np.stack(frames)
    if to_nchw:
        frames = np.transpose(frames, (0, 3, 1, 2))
    return frames

def load_feats(fn):
    mat = loadmat(fn)
    coeff_3dmm = mat['coeff']
    _id = coeff_3dmm[:, :80]
    _exp = coeff_3dmm[:, 80:144]
    _tex = coeff_3dmm[:, 144:224]
    _angle = coeff_3dmm[:, 224:227]
    _gamma = coeff_3dmm[:, 227:254]
    _trans = coeff_3dmm[:, 254:257]
    _crop = mat['transform_params'][:, -3:]
    result = {
        'id': _id,
        'exp': _exp,
        'tex': _tex,
        'angle': _angle,
        'gamma': _gamma,
        'trans': _trans,
        'crop': _crop,
    }
    return result

def align_with_gt(pd_feats, gt_feats):
    result = {}
    for key in pd_feats.keys():
        pd_value = pd_feats[key]
        gt_value = gt_feats[key]
        if len(pd_value) > len(gt_value):
            pd_value = pd_value[:len(gt_value)]
        elif len(pd_value) < len(gt_value):
            pd_value = np.concatenate((pd_value, pd_value[-1][None, ...].repeat(len(gt_value) - len(pd_value), axis=0)), axis=0)
        else:
            pass
        result[key] = pd_value
    return result

def run_cmd(cmd):
    subprocess.call(cmd, shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_video_folder', type=str, required=True)
    parser.add_argument('--pd_video_folder', type=str, required=True)
    parser.add_argument('--teamname', type=str, required=True)
    args = parser.parse_args()

    psnr_values = []
    ssim_values = []
    gt_feats = []
    pd_feats = []
    avoffset = []
    avconf = []
    result_l1 = defaultdict(list)
    tmp_path = os.path.join(args.pd_video_folder,'syncnet')
    os.makedirs(tmp_path, exist_ok=True)
    for file_name in os.listdir(args.gt_video_folder):
        if file_name.endswith(".mp4"):
            target = os.path.splitext(file_name)[0]
            gt_video_fn = f'{args.gt_video_folder}/{target}.mp4'
            pd_video_fn = f'{args.pd_video_folder}/{target}_pred.mp4'
            
            assert osp.exists(gt_video_fn), f"'{gt_video_fn}' is not exist"
            assert osp.exists(pd_video_fn), f"'{pd_video_fn}' is not exist"

            run_cmd(f'python ./syncnet_python/run_pipeline.py --videofile {pd_video_fn} --reference {target} --data_dir {tmp_path}')
            run_cmd(f'python ./syncnet_python/run_syncnet.py --videofile {pd_video_fn} --reference {target} --data_dir {tmp_path}')
            sync_path = f'{tmp_path}/pywork/{target}/metric.txt'
            with open(sync_path, 'r') as txt_file:
                lines = txt_file.readlines()
            offset = lines[0].strip()
            conf = lines[1].strip()
            offset = np.float64(float(offset))
            conf = np.float64(float(conf))
            avoffset.append(offset)
            avconf.append(conf)
            run_cmd(f'rm -rf {tmp_path}')
                
            gt_frames = read_mp4(gt_video_fn, True, False, True)
            pd_frames = read_mp4(pd_video_fn, True, False, True)

            gt_frames = torch.from_numpy(gt_frames).float() / 255.
            pd_frames = torch.from_numpy(pd_frames).float() / 255.
            
            if len(gt_frames) > len(pd_frames):
                gt_frames = gt_frames[:len(pd_frames)]
            elif len(gt_frames) < len(pd_frames):
                pd_frames = pd_frames[:len(gt_frames)]
                
            # PSNR
            psnr_value = psnr(pd_frames, gt_frames, reduction='none')
            psnr_values.extend([e.item() for e in psnr_value])
            # SSIM
            ssim_value = ssim(pd_frames, gt_frames, data_range=1., reduction='none')
            ssim_values.extend([e.item() for e in ssim_value])
    
            # 3DMM L1 Distance
            gt_feats_fn_l1 = f'./recons/gt/{target}.mat'
            pd_feats_fn_l1 = f'./recons/{args.teamname}/{target}_pred.mat'

            gt_feats_l1 = load_feats(gt_feats_fn_l1)
            pd_feats_l1 = load_feats(pd_feats_fn_l1)
            pd_feats_l1 = align_with_gt(pd_feats_l1, gt_feats_l1)
            
            for key in pd_feats_l1.keys():
                distance_l1 = np.abs(gt_feats_l1[key] - pd_feats_l1[key])
                result_l1[key].append(distance_l1)
    
    print('psnr:', np.array(psnr_values).mean())
    print('ssim:', np.array(ssim_values).mean())
    for key, value in result_l1.items():
        if key in ['angle','trans']:
            print(key, np.mean(np.concatenate(value, axis=0)))
    print('AV Offset:', np.mean(np.array(avoffset)))
    print('AV Confidence:', np.mean(np.array(avconf)))
    
    # print(avoffset)
    # print(avconf)
    