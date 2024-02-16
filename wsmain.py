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

        # get frame rate
        video = cv2.VideoCapture(os.path.join(args.img_path, vid))
        # fps=30
        fps = math.ceil(video.get(5))
        width = int(video.get(3))
        height = int(video.get(4))
        video_len = int(video.get(7))
        if args.fps != fps:
            args.fps = fps
        print('fps', fps)

        frame_path = os.path.join(args.save_dir, vid_name, 'orig_img')
        os.makedirs(frame_path, exist_ok=True)
        if args.format not in ['jpg', 'png', 'jpeg']:
            # extract frames from video
            video_path = os.path.join(args.img_path, f'{vid}')
            # clear frame folder
            if args.clear_folder:
                files = glob.glob(os.path.join(frame_path, '*'))
                for file in files:
                    os.remove(file)
            # os.system(f'ffmpeg -i {video_path} -f image2 '
            #         f'-vf fps={args.fps} -qscale 0 {frame_path}/%06d.jpg ' \
            #         f'-hide_banner  -loglevel error')
            assert len(os.listdir(frame_path)) == video_len
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

        start_count = int(sorted(os.listdir(frame_path))[0].split('.')[0])
        end_count = len(os.listdir(frame_path)) + start_count - 1
        # import pdb; pdb.set_trace()

        # prepare cmd for inference
        cmd_smplerx_inference = f'cd main && python inference.py ' \
            f'--num_gpus 1 --pretrained_model {args.ckpt} ' \
            f'--agora_benchmark agora_model ' \
            f'--img_path ../{frame_path} --start {start_count} --end {end_count} ' \
            f'--output_folder ../{args.save_dir}/{vid_name} ' \
            f'--show_verts --show_bbox --save_mesh ' \
            f'--multi_person ' 
            # f'--use_manual_bbox ' \
            # f'--show_verts --show_bbox --save_mesh --multi_person'
        if args.clear_folder:
            # clear inference folder
            files = glob.glob(f'../{args.save_dir}/{vid_name}/meta/*')
            for file in files:
                os.remove(file)
            files = glob.glob(f'../{args.save_dir}/{vid_name}/smplx/*')
            for file in files:
                os.remove(file)
        # os.system(cmd_smplerx_inference)

        # prepare cmd for rendering
        cmd_visualize_overlay = f'cd main && python render.py ' \
            f'--data_path ../{args.save_dir} --seq {vid_name} ' \
            f'--image_path ../{args.save_dir}/{vid_name} ' \
            f'--render_biggest_person False'
            # f'--load_mode propainter'
        if args.clear_folder:
            # clear overlay folder
            files = glob.glob(f'../{args.save_dir}/{vid_name}/smplerx_overlay_img/*')
            for file in files:
                os.remove(file)
        os.system(cmd_visualize_overlay)

        # copy orig img if img not exist in smplx overlay
        # if os.path.exists(os.path.join(args.save_dir, vid_name, 'smplerx_smplx_overlay')):
        #     for file in os.listdir(frame_path):
        #         if not os.path.exists(os.path.join(args.save_dir, vid_name, 'smplerx_smplx_overlay', file)):
        #             shutil.copy(os.path.join(frame_path, file),
        #                         os.path.join(args.save_dir, vid_name, 'smplerx_smplx_overlay', file))

        # # concat overlay video with frame rate
        # cmd_concat_overlay_video = f'ffmpeg -r {args.fps} -i ' \
        #     f'{args.save_dir}/{vid_name}/smplerx_smplx_overlay/%06d.jpg ' \
        #     f'-vcodec libx264 -crf 25 -pix_fmt yuv420p -r {args.fps} ' \
        #     f'{args.save_dir}/{vid_name}_smplerx.mp4 ' \
        #     f'-hide_banner  -loglevel error -y'
        # os.system(cmd_concat_overlay_video)

        # # prepare cmd for smplx post-processing
        # cmd_smplx_post_processing = f'python process_ws.py ' \
        #     f'--save_root {os.path.join(args.save_dir, vid_name)} ' \
        #     f'--width {width} --height {height} ' \
        #     f'--n_cam 2 '
        # os.system(cmd_smplx_post_processing)

        # # concat video with frame rate
        # cmd_concat_video = f'ffmpeg -r {args.fps} -i ' \
        #     f'{args.save_dir}/{vid_name}/overlay_img/%05d.jpg ' \
        #     f'-vcodec libx264 -crf 25 -pix_fmt yuv420p -r {args.fps} ' \
        #     f'{args.save_dir}/{vid_name}_propainter.mp4 ' \
        #     f'-hide_banner  -loglevel error'
        # # os.system(cmd_concat_video)

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
