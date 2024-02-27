import os
import shutil
import gc
import cv2
import glob
import math

import argparse
import pdb
import mmcv

def call_inference(args):

    # parse videos
    video_list = [vn for vn in os.listdir(args.img_path)]
    # loop over videos
    for i, vid in enumerate(video_list):
        print('processing', vid)
        save_path = os.path.join(args.img_path, vid)
        # prepare cmd for rendering
        cmd_visualize_overlay = f'cd main && python render.py ' \
            f'--data_path ../{save_path} --seq "" ' \
            f'--image_path ../{save_path} ' \
            f'--smplx_folder_name "processed_smplx" ' \
            f'--render_biggest_person False'
            # f'--load_mode propainter'
        if args.clear_folder:
            # clear overlay folder
            files = glob.glob(f'../{save_path}/smplerx_processed_smplx_overlay/*')
            for file in files:
                os.remove(file)
        os.system(cmd_visualize_overlay)

        # concat overlay video with frame rate
        cmd_concat_overlay_video = f'ffmpeg -r {args.fps} -i ' \
            f'{save_path}/smplerx_processed_smplx_overlay/%06d.jpg ' \
            f'-vcodec mjpeg -qscale 0 -pix_fmt yuv420p -r {args.fps} ' \
            f'{save_path}/processed_smplerx.mp4 ' \
            f'-hide_banner  -loglevel error -y'
        os.system(cmd_concat_overlay_video)
        cmd_concat_overlay_video = f'ffmpeg -r {args.fps} -i ' \
            f'{save_path}/smplerx_smplx_overlay/%06d.jpg ' \
            f'-vcodec mjpeg -qscale 0  -pix_fmt yuv420p -r {args.fps} ' \
            f'{save_path}/smplerx.mp4 ' \
            f'-hide_banner  -loglevel error -y'
        os.system(cmd_concat_overlay_video)
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--format', type=str,
                        default='mp4')
    parser.add_argument('--fps', type=int,
                        default=30)
    parser.add_argument('--ckpt', type=str,
                        default='smpler_x_h32')
    parser.add_argument('--img_path', type=str,
                        default='vid_input')
    parser.add_argument('--clear_folder',
                        action='store_true')

    args = parser.parse_args()
    call_inference(args)
