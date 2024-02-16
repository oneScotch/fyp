import os
import os.path as osp
import tqdm
import numpy as np
import glob
import json

def merge_folder_into_file_smplx(folders, pid = 0):
    pid_str = f"_{pid}.npz"
    fns = [f for f in os.listdir(folders[0]) if f.endswith(pid_str)]
    fns.sort()

    smplxs = []
    for fn in tqdm.tqdm(fns):
        smplx={}
        for folder in folders:
            data = np.load(osp.join(folder,fn), allow_pickle=True)
            for key, val in data.items():
                if key not in smplx:
                    smplx[key] = []
                smplx[key].append(val.reshape(1, -1))

        for key in smplx:
            smplx[key] = np.concatenate(smplx[key], axis=0)

        # global_orient = smplx['global_orient']
        body_pose = smplx.pop('body_pose')
        left_hand_pose = smplx.pop('left_hand_pose')
        right_hand_pose = smplx.pop('right_hand_pose')
        jaw_pose = smplx.pop('jaw_pose')
        leye_pose = smplx.pop('leye_pose')
        reye_pose = smplx.pop('reye_pose')
        smplx['body_pose'] = np.concatenate(
            (body_pose, jaw_pose, leye_pose, reye_pose,
            left_hand_pose, right_hand_pose),
            axis=-1)

        smplx['meta'] = {'gender': 'neutral'}
        smplxs.append(smplx)
    return smplxs

def split_file_smplx_into_folder(smplx, folder, frame_mapping, pid = 0):
    for fidx in frame_mapping.keys():
        fid = frame_mapping[fidx]
        # import pdb; pdb.set_trace()
        fidx = int(fidx)

        fullpose = smplx['body_pose'][fidx].reshape(-1, 55, 3)
        params = {}
        params['global_orient'] = fullpose[:, 0].reshape(-1, 3)
        params['body_pose'] = fullpose[:, 1:22].reshape(-1, 63)
        params['jaw_pose'] = fullpose[:, 22].reshape(-1, 3)
        params['leye_pose'] = fullpose[:, 23].reshape(-1, 3)
        params['reye_pose'] = fullpose[:, 24].reshape(-1, 3)
        params['left_hand_pose'] = fullpose[:, 25:40].reshape(-1, 45)
        params['right_hand_pose'] = fullpose[:, 40:55].reshape(-1, 45)
        params['transl'] = smplx['transl'][fidx].reshape(-1, 3)
        params['betas'] = smplx['betas'][fidx].reshape(-1, 10)
        params['expression'] = smplx['expression'][fidx].reshape(-1, 10)

        fname = osp.join(folder, f'{fid:05}_{pid}.npz')
        np.savez(fname, **params)

def merge_folder_into_file_meta(folders, pid = 0):
    pid_str = f"_{pid}.json"
    fns = [f for f in os.listdir(folders[0]) if f.endswith(pid_str)]
    fns.sort()

    metas = []
    for fn in tqdm.tqdm(fns):
        for folder in folders:
            meta = {}
            data = json.load(open(os.path.join(folder, fn)))
            for key, val in data.items():
                if key not in meta:
                    meta[key] = []
                val = np.array(val)
                meta[key].append(val.reshape(1, -1))

        for key in meta:
            meta[key] = np.concatenate(meta[key], axis=0)
        metas.append(meta)
    return metas

def sort_output_pid(smplx_path):
    pid_frameid_dict = {}

    for filename in os.listdir(smplx_path):
        # Extract frame_id and pid from the filename
        frame_id_str, pid_str = filename.split('_')
        frame_id = int(frame_id_str)
        pid = int(pid_str.split('.')[0])

        # Add to the dictionary
        if pid in pid_frameid_dict:
            pid_frameid_dict[pid].append(frame_id)
        else:
            pid_frameid_dict[pid] = [frame_id]

    # Sort the frame_ids for each pid
    for pid, frame_ids in pid_frameid_dict.items():
        pid_frameid_dict[pid] = sorted(frame_ids)

    return pid_frameid_dict
