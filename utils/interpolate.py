import os
import cv2
import time
import torch
import warnings
import argparse
import numpy as np
from tqdm import tqdm
import _thread
import psutil
import skvideo.io
import random
from queue import Queue
from torch.nn import functional as F
from utils.utils import remaining_time
from model.pytorch_msssim import ssim_matlab
from utils.utils import process_ffmpeg_info, check_file_size, get_gpu_memory
warnings.filterwarnings("ignore")


def transferAudio(sourceVideo, targetVideo):
    rand_seed = int(random.randint(0, 4800))
    tempAudioFileName = f"./temp/{rand_seed}_audio.mkv"

    # split audio from original video file and store in "temp" directory
    if True:
        # create new "temp" directory
        if not os.path.exists("temp"):
            os.makedirs("temp")
        # extract audio from video
        os.system('ffmpeg -y -i "{}" -c:a copy -vn {}'.format(sourceVideo, tempAudioFileName))

    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    os.rename(targetVideo, targetNoAudio)
    
    # combine audio file and new video file
    os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))

    if os.path.getsize(targetVideo) == 0: 
        # if ffmpeg failed to merge the video and audio together try converting the audio to aac
        tempAudioFileName = f"./temp/{rand_seed}_audio.m4a"
        os.system('ffmpeg -y -i "{}" -c:a aac -b:a 160k -vn {}'.format(sourceVideo, tempAudioFileName))
        os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))
        if (os.path.getsize(targetVideo) == 0): # if aac is not supported by selected format
            os.rename(targetNoAudio, targetVideo)
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)
    os.remove(tempAudioFileName)


def interpolate_rife_inference(**kwargs):
    """
        视频插帧模块推理功能
    """
    root_path = kwargs.get("root_path")
    print("root_path: ", root_path)

    video_full_name = kwargs.get("video_full_name")
    print("video_full_name: ", video_full_name)
    cache_video_path = os.path.join(root_path, "cache", video_full_name)

    parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
    parser.add_argument('--video', dest='video', type=str, default=None)
    parser.add_argument('--output', dest='output', type=str, default=None)
    parser.add_argument('--img', dest='img', type=str, default=None)
    parser.add_argument('--montage', dest='montage', action='store_true', help='montage origin video')
    parser.add_argument('--model', dest='modelDir', type=str, default='interpolate/train_log', help='directory with trained model files')
    parser.add_argument('--fp16', dest='fp16', action='store_true', help='fp16 mode for faster and more lightweight inference on cards with Tensor Cores')
    parser.add_argument('--UHD', dest='UHD', action='store_true', help='support 4k video')
    parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='Try scale=0.5 for 4k video')
    parser.add_argument('--fps', dest='fps', type=int, default=None)
    parser.add_argument('--png', dest='png', action='store_true', help='whether to vid_out png format vid_outs')
    parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='vid_out video extension')

    parser.add_argument('--exp', dest='exp', type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    
    if args.UHD and args.scale==1.0:
        args.scale = 0.5

    assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
    if not args.img is None:
        args.png = True

    # 加载模型权重
    model_dir = os.path.join(root_path, "train_log")
    try:
        try:
            try:
                from model.RIFE_HDv2 import Model
                model = Model()
                model.load_model(model_dir, -1)
                print("Loaded v2.x HD model.")
            except:
                from train_log.RIFE_HDv3 import Model
                model = Model()
                model.load_model(model_dir, -1)
                print("Loaded v3.x HD model.")
        except:
            from model.RIFE_HD import Model
            model = Model()
            model.load_model(model_dir, -1)
            print("Loaded v1.x HD model")
    except:
        from model.RIFE import Model
        model = Model()
        model.load_model(model_dir, -1)
        print("Loaded ArXiv-RIFE model")

    model.eval()
    model.device()
    videoCapture = kwargs.get("videoCapture")

    if not videoCapture is None:
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        videoCapture.release()
        
        if args.fps is None:
            fpsNotAssigned = True
            args.fps = fps * (2 ** args.exp)
        else:
            fpsNotAssigned = False
        
        videogen = skvideo.io.vreader(cache_video_path)
        
        lastframe = next(videogen)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video_path_wo_ext, ext = os.path.splitext(cache_video_path)
        print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(video_path_wo_ext, args.ext, tot_frame, fps, args.fps))
        
        if args.png == False and fpsNotAssigned == True:
            print("The audio will be merged after interpolation process")
        else:
            print("Will not merge audio because using png or fps flag!")
    else:
        videogen = []
        for f in os.listdir(args.img):
            if 'png' in f:
                videogen.append(f)

        tot_frame = len(videogen)
        videogen.sort(key= lambda x:int(x[:-4]))
        lastframe = cv2.imread(os.path.join(args.img, videogen[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
        videogen = videogen[1:]
  
    h, w, _ = lastframe.shape
    vid_out_name = None
    vid_out = None

    if args.png:
        if not os.path.exists('vid_out'):
            os.mkdir('vid_out')
    else:
        if args.output is not None:
            vid_out_name = args.output
        else:
            vid_out_name = '{}_{}X_{}fps.{}'.format(video_path_wo_ext, (2 ** args.exp), int(np.round(args.fps)), args.ext)
        vid_out = cv2.VideoWriter(vid_out_name, fourcc, args.fps, (w, h))
    
    def clear_write_buffer(user_args, write_buffer):
        cnt = 0
        while True:
            item = write_buffer.get()
            if item is None:
                break
            if user_args.png:
                cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt), item[:, :, ::-1])
                cnt += 1
            else:
                vid_out.write(item[:, :, ::-1])
    
    def build_read_buffer(user_args, read_buffer, videogen):
        try:
            for frame in videogen:
                if not user_args.img is None:
                    frame = cv2.imread(os.path.join(user_args.img, frame), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
                if user_args.montage:
                    frame = frame[:, left: left + w]
                read_buffer.put(frame)
        except:
            pass
        read_buffer.put(None)
    
    def make_inference(I0, I1, n):
        middle = model.inference(I0, I1, args.scale)
        if n == 1:
            return [middle]
        first_half = make_inference(I0, middle, n=n//2)
        second_half = make_inference(middle, I1, n=n//2)
        if n % 2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]

    def pad_image(img):
        if args.fp16:
            return F.pad(img, padding).half()
        else:
            return F.pad(img, padding)
    
    if args.montage:
        left = w // 4
        w = w // 2
        
    tmp = max(32, int(32 / args.scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)

    pbar = tqdm(total = tot_frame)

    if args.montage:
        lastframe = lastframe[:, left: left + w]

    write_buffer = Queue(maxsize=500)
    read_buffer = Queue(maxsize=500)
    _thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))
    _thread.start_new_thread(clear_write_buffer, (args, write_buffer))

    I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = pad_image(I1)
    temp = None # save lastframe when processing static frame

    # 获取参数面板
    st_sr_frame = kwargs.get("st_sr_frame")
    progress_bar = kwargs.get("progress_bar")
    
    # 输入视频名称
    kpi1_text = kwargs.get("kpi1_text")
    kpi1_text.write(str(video_full_name))

    video_ffprobe_info = kwargs.get("video_ffprobe_info", None)  # video_ffprobe_info记录完整的ffprobe信息
    video_info_dict = process_ffmpeg_info(video_ffprobe_info)  # video_info_dict记录处理过的信息字典

    video_stream_dict = dict()  # 视频流参数
    audio_stream_dict = dict()  # 音频流参数

    video_stream_dict['视频时长'] = video_info_dict.get('duration', 'Unkown')
    video_stream_dict['视频码率'] = video_info_dict.get('bitRate', 'Unkown')
    video_stream_dict['视频格式'] = video_full_name.split(".")[1].upper()
    video_stream_dict['文件大小'] = check_file_size(os.path.join(root_path, 'cache', video_full_name))
    video_stream_dict['视频帧数'] = video_info_dict.get('Size', 'Unkown')
    video_stream_dict['输入分辨率'] = video_info_dict.get('resolution', 'Unkown')
    video_stream_dict['视频帧率'] = video_info_dict.get('Framesrate', 'Unkown')
    video_stream_dict['编码格式'] = video_info_dict.get('videoCodec', 'Unkown')
    video_stream_dict['扫描格式'] = video_info_dict.get('pixelFormat', 'Unkown')
    video_stream_dict['色深'] = video_info_dict.get('BitDepth', 'Unkown')
    video_stream_dict['SDR/HDR'] = video_info_dict.get('transfer', 'Unkown')
    video_stream_dict['色彩空间'] = video_info_dict.get('Gamut', 'Unkown')
    video_stream_dict['视频的横幅比'] = video_info_dict.get('ratio', 'Unkown')
    audio_stream_dict['音频编码格式'] = video_info_dict.get('audioCodec', 'Unkown')
    audio_stream_dict['声道数'] = video_info_dict.get("channels", 'Unkown')

    inf_ov_1_text = kwargs.get("inf_ov_1_text")
    inf_ov_2_text = kwargs.get("inf_ov_2_text")
    inf_ov_3_text = kwargs.get("inf_ov_3_text")
    
    output_info_dict = dict()  # 输出参数配置
    output_info_dict['输出分辨率'] = f"{w}x{h}"
    output_info_dict['插帧倍数'] = "x2"
    output_info_dict['FPS'] = f"{int(fps * 2)}"
    inf_ov_3_text.write(output_info_dict)  # 输出参数配置


    kpi2_text = kwargs.get("kpi2_text")
    try:
        kpi2_text.write(video_stream_dict)
    except:
        kpi2_text.write(str('NA'))

    kpi3_text = kwargs.get("kpi3_text")
    try:
        kpi3_text.write(audio_stream_dict)
    except:
        kpi3_text.write(str('NA'))
    
    js1_text = kwargs.get("js1_text")
    js2_text = kwargs.get("js2_text")
    js3_text = kwargs.get("js3_text")

    # 插帧处理
    step = 0
    while True:
        start_time = time.perf_counter()

        if temp is not None:
            frame = temp
            temp = None
        else:
            frame = read_buffer.get()

        if frame is None:
            break

        I0 = I1
        I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = pad_image(I1)
        I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
        I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

        break_flag = False
        if ssim > 0.996:        
            frame = read_buffer.get() # read a new frame
            if frame is None:
                break_flag = True
                frame = lastframe
            else:
                temp = frame
            I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
            I1 = pad_image(I1)
            I1 = model.inference(I0, I1, args.scale)
            I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
            frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
        
        if ssim < 0.2:
            output = []
            for i in range((2 ** args.exp) - 1):
                output.append(I0)
        else:
            output = make_inference(I0, I1, 2**args.exp-1) if args.exp else []

        if args.montage:
            write_buffer.put(np.concatenate((lastframe, lastframe), 1))
            for mid in output:
                mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
                write_buffer.put(np.concatenate((lastframe, mid[:h, :w]), 1))
        else:
            write_buffer.put(lastframe)
            for mid in output:
                mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
                write_buffer.put(mid[:h, :w])

        pbar.update(1)
        lastframe = frame
        if break_flag:
            break
        
        # 显示当前处理视频帧
        st_sr_frame.image(frame, use_column_width=True)

        # 更新推理概述信息
        inf_ov_1_text.write("当前帧：{}  | 总帧数：{}".format(step + 1, int(tot_frame)))

        # CPU, GPU占用使用情况
        if step % 20 == 0:
            js1_text.write(str(psutil.virtual_memory()[2])+"%")
            js2_text.write(str(psutil.cpu_percent())+'%')
            
            try:
                js3_text.write(str(get_gpu_memory())+' MB')
            except:
                js3_text.write(str('NA'))
        
        # 写入剩余时间参数
        end_time = time.perf_counter()
        time_cost = end_time - start_time

        if step % 10 == 0:
            extra_time = remaining_time(int(tot_frame) - step, time_cost)
            inf_ov_2_text.write(str(extra_time))
        
        progress_bar.progress((step + 1) / tot_frame)
        step += 1
    
    ## 视频处理结束
    ## 所有面板归零
    progress_bar.progress(tot_frame / tot_frame)
    js1_text.write("0.0%")
    js2_text.write("0.0%")
    js3_text.write("0 MB")
    inf_ov_1_text.write("当前帧：{} | 总帧数： {}".format(int(tot_frame), int(tot_frame)))
    inf_ov_2_text.write("0 s")

    if args.montage:
        write_buffer.put(np.concatenate((lastframe, lastframe), 1))
    else:
        write_buffer.put(lastframe)

    while not write_buffer.empty():
        time.sleep(0.1)

    pbar.close()
    if not vid_out is None:
        vid_out.release()

    # move audio to new video file if appropriate
    if args.png == False and fpsNotAssigned == True and not cache_video_path is None:
        try:
            transferAudio(cache_video_path, vid_out_name)
        except:
            print("Audio transfer failed. Interpolated video will have no audio")
            targetNoAudio = os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
            os.rename(targetNoAudio, vid_out_name)

    return vid_out_name
