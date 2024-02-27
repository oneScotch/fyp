import os
import sys
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
import cv2
from tqdm import tqdm
import json
from typing import Literal, Union
from mmdet.apis import init_detector, inference_detector
from mmtrack.apis import init_model, inference_sot
from utils.inference_utils import process_mmdet_results, non_max_suppression

import pdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, dest='num_gpus')
    parser.add_argument('--exp_name', type=str, default='output/test')
    parser.add_argument('--pretrained_model', type=str, default=0)
    parser.add_argument('--testset', type=str, default='EHF')
    parser.add_argument('--agora_benchmark', type=str, default='na')
    parser.add_argument('--img_path', type=str, default='input.png')
    parser.add_argument('--start', type=str, default=1)
    parser.add_argument('--end', type=str, default=1)
    parser.add_argument('--output_folder', type=str, default='output')
    parser.add_argument('--demo_dataset', type=str, default='na')
    parser.add_argument('--demo_scene', type=str, default='all')
    parser.add_argument('--show_verts', action="store_true")
    parser.add_argument('--show_bbox', action="store_true")
    parser.add_argument('--save_mesh', action="store_true")
    parser.add_argument('--multi_person', action="store_true")
    parser.add_argument('--iou_thr', type=float, default=0.5)
    parser.add_argument('--bbox_thr', type=int, default=50)
    parser.add_argument('--detection_method', type=str, default='mmtrack',
                        choices=['mmdet', 'mmtrack'])

    parser.add_argument('--use_manual_bbox', action="store_true")

    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    config_path = osp.join('./config', f'config_{args.pretrained_model}.py')
    ckpt_path = osp.join('../pretrained_models', f'{args.pretrained_model}.pth.tar')

    cfg.get_config_fromfile(config_path)
    cfg.update_test_config(args.testset, args.agora_benchmark, shapy_eval_split=None, 
                            pretrained_model_path=ckpt_path, use_cache=False)
    cfg.update_config(args.num_gpus, args.exp_name)
    cudnn.benchmark = True

    # load model
    from base import Demoer
    from utils.preprocessing import load_img, process_bbox, generate_patch_image
    from utils.vis import render_mesh, save_obj
    from utils.human_models import smpl_x
    demoer = Demoer()
    demoer._make_model()
    demoer.model.eval()
    
    start = int(args.start)
    end = start + int(args.end)
    multi_person = args.multi_person
            
    if not args.use_manual_bbox:
        ### mmdet init
        checkpoint_file = '../pretrained_models/mmdet/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        config_file= '../pretrained_models/mmdet/mmdet_faster_rcnn_r50_fpn_coco.py'
        det_model = init_detector(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'
        ### mmtrack init
        checkpoint_file = f'../pretrained_models/mmtrack/mixformer_cvt_500e_lasot.pth'
        config_file= f'../pretrained_models/mmtrack/mixformer_cvt_500e_got10k.py'
        track_model = init_model(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'
        track_started, tracked_frame = False, 0
    else:
        # check mask folder number
        mask_folders = [f for f in os.listdir(os.path.dirname(args.img_path))
                         if 'mask_' in f]
        mask_folders = [f for f in mask_folders if 'zip' not in f]
        mask_folders = sorted(mask_folders)

    for frame in tqdm(range(start, end)):
        img_path = os.path.join(args.img_path, f'{int(frame):06d}.jpg')

        # prepare input image
        transform = transforms.ToTensor()
        original_img = load_img(img_path)
        vis_img = original_img.copy()
        original_img_height, original_img_width = original_img.shape[:2]
        os.makedirs(args.output_folder, exist_ok=True)

        if args.use_manual_bbox:

            mmdet_box = []
            for m_folder in mask_folders:    
                # load mask with index - 1 
                mask_path = img_path.replace('orig_img/0', f'{m_folder}/')
                fstamp = mask_path.split('/')[-1].split('.')[0]
                mask_path = mask_path.replace(fstamp, f'{(int(fstamp)-1):05d}')

                # convert mask to bbox
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = mask.astype(np.uint8)
                else:
                    mmdet_box.append([0, 0, 0, 0, 0])
                    continue    

                mask_true = np.where(mask>0)

                # check if bbox exist
                if len(mask_true[0]) * len(mask_true[1]) < 1:
                    mmdet_box.append([0, 0, 0, 0, 0])
                    continue
                else:
                    x_min = int(np.min(mask_true[1]))
                    x_max = int(np.max(mask_true[1]))
                    y_min = int(np.min(mask_true[0]))
                    y_max = int(np.max(mask_true[0]))

                    # in xyxy format
                    mmdet_box.append([x_min, y_min, x_max, y_max, 1])
            num_bbox = len(mmdet_box)
            det_bbox = mmdet_box

        else:
            if not track_started:
                ## mmdet inference
                mmdet_results = inference_detector(det_model, img_path)
                det_bbox = process_mmdet_results(mmdet_results, cat_id=0, multi_person=True)
                det_bbox = det_bbox[0]

            if args.detection_method == 'mmtrack':

                if len(det_bbox) >= 1 and not track_started:
                    track_gt_bbox = det_bbox[0]
                    track_started = True
                if track_started:
                    img = cv2.imread(img_path)
                    # img = np.array([img, img])
                    mmtrack_results = inference_sot(track_model, img, init_bbox=track_gt_bbox, frame_id=tracked_frame)
                    tracked_frame += 1
                    det_bbox = [mmtrack_results['track_bboxes'][:5]]
            
        # save original image if no bbox
        if len(det_bbox)<1:
            # save rendered image
            frame_name = img_path.split('/')[-1]
            save_path_img = os.path.join(args.output_folder, 'img')
            os.makedirs(save_path_img, exist_ok= True)
            cv2.imwrite(os.path.join(save_path_img, f'{frame_name}'), vis_img[:, :, ::-1])
            continue

        if not args.use_manual_bbox:
            if not multi_person:
                # only select the largest bbox
                num_bbox = 1
            else:
                # keep bbox by NMS with iou_thr
                # print(det_bbox)
                det_bbox = non_max_suppression(det_bbox, args.iou_thr)
                num_bbox = len(det_bbox)
        
        ## loop all detected bboxes
        for bbox_id in range(num_bbox):
            bbox_xywh = np.zeros((4))
            bbox_xywh[0] = det_bbox[bbox_id][0]
            bbox_xywh[1] = det_bbox[bbox_id][1]
            bbox_xywh[2] =  abs(det_bbox[bbox_id][2]-det_bbox[bbox_id][0])
            bbox_xywh[3] =  abs(det_bbox[bbox_id][3]-det_bbox[bbox_id][1]) 
            # print(bbox_id, bbox_xywh)

            # skip small bboxes by bbox_thr in pixel
            if bbox_xywh[2] < args.bbox_thr or bbox_xywh[3] < args.bbox_thr * 3:
                continue

            # for bbox visualization 
            start_point = (int(det_bbox[bbox_id][0]), int(det_bbox[bbox_id][1]))
            end_point = (int(det_bbox[bbox_id][2]), int(det_bbox[bbox_id][3]))   

            bbox = process_bbox(bbox_xywh, original_img_width, original_img_height)
            img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
            img = transform(img.astype(np.float32))/255
            img = img.cuda()[None,:,:,:]
            inputs = {'img': img}
            targets = {}
            meta_info = {}

            # mesh recovery
            with torch.no_grad():
                out = demoer.model(inputs, targets, meta_info, 'test')
            mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

            ## save mesh
            if args.save_mesh:
                save_path_mesh = os.path.join(args.output_folder, 'mesh')
                os.makedirs(save_path_mesh, exist_ok= True)
                save_obj(mesh, smpl_x.face, os.path.join(save_path_mesh, f'{frame:05}_{bbox_id}.obj'))

            ## save single person param
            smplx_pred = {}
            smplx_pred['global_orient'] = out['smplx_root_pose'].reshape(-1,3).cpu().numpy()
            smplx_pred['body_pose'] = out['smplx_body_pose'].reshape(-1,3).cpu().numpy()
            smplx_pred['left_hand_pose'] = out['smplx_lhand_pose'].reshape(-1,3).cpu().numpy()
            smplx_pred['right_hand_pose'] = out['smplx_rhand_pose'].reshape(-1,3).cpu().numpy()
            smplx_pred['jaw_pose'] = out['smplx_jaw_pose'].reshape(-1,3).cpu().numpy()
            smplx_pred['leye_pose'] = np.zeros((1, 3))
            smplx_pred['reye_pose'] = np.zeros((1, 3))
            smplx_pred['betas'] = out['smplx_shape'].reshape(-1,10).cpu().numpy()
            smplx_pred['expression'] = out['smplx_expr'].reshape(-1,10).cpu().numpy()
            smplx_pred['transl'] =  out['cam_trans'].reshape(-1,3).cpu().numpy()
            save_path_smplx = os.path.join(args.output_folder, 'smplx')
            os.makedirs(save_path_smplx, exist_ok= True)

            npz_path = os.path.join(save_path_smplx, f'{frame:05}_{bbox_id}.npz')
            np.savez(npz_path, **smplx_pred)

            ## render single person mesh
            focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
            princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
            # vis_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt}, 
            #                       mesh_as_vertices=args.show_verts)
            
            if args.show_bbox:
            # if True:
                # print(start_point, end_point)
                vis_img = cv2.rectangle(vis_img, start_point, end_point, (255, 0, 0), 2)

            ## save single person meta
            meta = {'focal': focal, 
                    'princpt': princpt, 
                    'bbox': bbox.tolist(), 
                    'bbox_mmdet': bbox_xywh.tolist(), 
                    'bbox_id': bbox_id,
                    'img_path': img_path}
            json_object = json.dumps(meta, indent=4)

            save_path_meta = os.path.join(args.output_folder, 'meta')
            os.makedirs(save_path_meta, exist_ok= True)
            with open(os.path.join(save_path_meta, f'{frame:05}_{bbox_id}.json'), "w") as outfile:
                outfile.write(json_object)

        # save rendered image with all person
        frame_name = img_path.split('/')[-1]
        save_path_img = os.path.join(args.output_folder, 'img')
        os.makedirs(save_path_img, exist_ok= True)
        cv2.imwrite(os.path.join(save_path_img, f'{frame_name}'), vis_img[:, :, ::-1])


if __name__ == "__main__":
    main()
