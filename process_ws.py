import json
import numpy as np
import os.path
import argparse 
from utils.test import bone_rotations_weighted_average_log_ref
from utils.smplx_constants import *
from utils.io_utils import sort_output_pid, \
    merge_folder_into_file_smplx, merge_folder_into_file_meta
import torch
# for debug
# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         # import pdb;pdb.set_trace()
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         elif isinstance(obj, np.float32):
#             return float(obj)
#         return json.JSONEncoder.default(self, obj)

def parse_args():
    parser = argparse.ArgumentParser()
    
    # io
    parser.add_argument('--save_root', type=str)
    
    # video and cam info
    parser.add_argument('--n_cam', type=int, default=4)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    reference_rotations = get_reference_rotations()
    sample_weights = [1/args.n_cam] * args.n_cam
    # sample_weights = [0.8, 0.2]
    meta_path = []
    smplx_path = []
    for i in range(args.n_cam):
        meta_path.append(os.path.join(args.save_root, str(i), 'meta'))
        smplx_path.append(os.path.join(args.save_root, str(i), 'smplx'))


    # smplx related constants
    hands_meanl, hands_meanr = get_hands_mean()
    output_dict = sort_output_pid(smplx_path[0])
    for pid, frame_idxs in output_dict.items():
        print(f'>>>Processing pid = {pid} ...')
        smplxs = merge_folder_into_file_smplx(smplx_path, pid = pid)
        metas = merge_folder_into_file_meta(meta_path, pid = pid)
        for frame_id in frame_idxs:
            frame_index = frame_id - 1
            joint_rots = smplxs[frame_index]['body_pose'].reshape(args.n_cam, -1, 3)
            # flat_hand_mean False(smplerx) -> True(Saas)
            # joint_rots[:, 24:39, :] += hands_meanl
            # joint_rots[:, 40:, :] += hands_meanr
            smoothed_smplx = bone_rotations_weighted_average_log_ref(reference_rotations, joint_rots, sample_weights)
            smplx_smoothed = smplxs[frame_index].copy()
            # smplx_smoothed['global_orient'] = smoothed_smplx[0].numpy()
            smplx_smoothed['body_pose'] = smoothed_smplx[0:21].reshape(-1,3).numpy()
            smplx_smoothed['jaw_pose'] = smoothed_smplx[21].reshape(-1,3).numpy()
            smplx_smoothed['leye_pose'] = smoothed_smplx[22].reshape(-1,3).numpy()
            smplx_smoothed['reye_pose'] = smoothed_smplx[23].reshape(-1,3).numpy()
            smplx_smoothed['left_hand_pose'] = smoothed_smplx[24:39].reshape(-1,3).numpy()
            smplx_smoothed['right_hand_pose'] = smoothed_smplx[39:].reshape(-1,3).numpy()
            for i in range(args.n_cam):
                processed_path = os.path.join(args.save_root, str(i), 'processed_smplx')
                os.makedirs(processed_path, exist_ok=True)
                smplx_saved = smplx_smoothed.copy()
                smplx_saved['global_orient'] = smplx_saved['global_orient'][i].reshape(-1,3)
                smplx_saved['betas'] = smplx_saved['betas'][i].reshape(-1,10)
                smplx_saved['expression'] = smplx_saved['expression'][i].reshape(-1,10)
                smplx_saved['transl'] = smplx_saved['transl'][i].reshape(-1,3)
                save_fn = os.path.join(processed_path, f'{frame_id:05}_{pid:01}')
                np.savez(save_fn, **smplx_saved)
                print(f'Smoothed smplx saved to {save_fn}.')

if __name__ == '__main__':
    main()
