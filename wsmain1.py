import os
import shutil
import gc
import cv2
import glob
import math

import argparse
import pdb
# import mmcv

def call_inference(args):

    # parse videos
    if args.vid == 'all':
        video_list = [vn for vn in os.listdir(args.img_path)
                    if vn.endswith(args.format)]
    elif args.vid == 'zhege':
        video_list = glob.glob(os.path.join(args.save_dir, '*', 'input.mp4'))
        for vid in video_list:
            vid_name = vid.split('/')[-2]
            shutil.copy(vid, os.path.join(args.img_path, vid.split('/')[-2] + '.mp4'))
        video_list = [vn for vn in os.listdir(args.img_path)
                    if vn.endswith(args.format)]
    else:
        video_list = [f'{args.vid}.{args.format}']

    # loop over videos
    for vid in video_list:
        print('processing', vid)

        vid_name = vid.split('.')[0]

        # # get frame rate
        # video = cv2.VideoCapture(os.path.join(args.img_path, vid))
        # # fps=30
        # fps = math.ceil(video.get(5))
        # width = int(video.get(3))
        # height = int(video.get(4))
        # video_len = int(video.get(7))
        # if args.fps != fps:
        #     args.fps = fps
        # print('fps', fps)

        # frame_path = os.path.join(args.save_dir, vid_name, 'orig_img')
        # os.makedirs(frame_path, exist_ok=True)
        # if args.format not in ['jpg', 'png', 'jpeg']:
        #     # extract frames from video
        #     video_path = os.path.join(args.img_path, f'{vid}')
        #     # clear frame folder
        #     if args.clear_folder:
        #         files = glob.glob(os.path.join(frame_path, '*'))
        #         for file in files:
        #             os.remove(file)
        #     os.system(f'ffmpeg -i {video_path} -f image2 '
        #             f'-vf fps={args.fps} -qscale 0 {frame_path}/%06d.jpg ' \
        #             f'-hide_banner  -loglevel error')
        #     assert len(os.listdir(frame_path)) == video_len
        # else:
        #     # copy frames from folder
        #     frame_source_dir = os.path.join(args.img_path,  vid_name)
        #     os.makedirs(frame_source_dir, exist_ok=True)
        #     for file in os.listdir(frame_path):
        #         if file.endswith(args.format):
        #             frame_stamp = int(file.split('.')[0])
        #             filem = f'{frame_stamp:06d}.{file.split(".")[-1]}'

        #         shutil.copy(os.path.join(frame_path, file),
        #                     os.path.join(frame_source_dir, filem))
            # # concat to videos
            # os.system(f'ffmpeg -r {args.fps} -i ' \
            #     f'{frame_path}/%06d.{args.format} ' \
            #     f'-vcodec libx264 -crf 25 -pix_fmt yuv420p -r {args.fps} ' \
            #     f'{args.img_path}/{vid_name}.mp4 ' \
            #     f'-hide_banner  -loglevel error')


        # prepare cmd for smplx post-processing
        cmd_smplx_post_processing = f'python process_ws.py ' \
            f'--save_root {os.path.join(args.save_dir, vid_name)} ' \
            f'--n_cam 2 '
        os.system(cmd_smplx_post_processing)

        # concat video with frame rate
        cmd_concat_video = f'ffmpeg -r {args.fps} -i ' \
            f'{args.save_dir}/{vid_name}/overlay_img/%05d.jpg ' \
            f'-vcodec libx264 -crf 25 -pix_fmt yuv420p -r {args.fps} ' \
            f'{args.save_dir}/{vid_name}_propainter.mp4 ' \
            f'-hide_banner  -loglevel error'
        # os.system(cmd_concat_video)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--vid', type=str,
                        default='all')
    parser.add_argument('--format', type=str,
                        default='mp4')
    parser.add_argument('--fps', type=int,
                        default=0)
    parser.add_argument('--ckpt', type=str,
                        default='smpler_x_h32')
    parser.add_argument('--img_path', type=str,
                        default='vid_input')
    parser.add_argument('--save_dir', type=str,
                        default='vid_output')
    parser.add_argument('--clear_folder',
                        action='store_true')

    args = parser.parse_args()
    call_inference(args)
