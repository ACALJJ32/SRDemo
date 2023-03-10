import os
import cv2
import mmcv
import time
import random
import torch
import shutil
import psutil
import subprocess
import numpy as np
import streamlit as st
from copy import deepcopy
from utils.color import plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931
from utils.arch_utils import chop_forward_dim4, chop_forward_dim5
from model.archs.edvr import EDVRNet
from model.archs.basicvsr import BasicVSRNet
from model.archs.basicvsr_pp import BasicVSRPlusPlus
from model.archs.basicvsr_pp_gauss import BasicVSRPlusPlus_Gauss


def check_workspace(root_path):
    work_space_path = os.path.join(root_path, "work_space")
    
    if not os.path.exists(work_space_path):
        os.makedirs(work_space_path, exist_ok=True)
    

    sr_work_space = os.path.join(root_path, "work_space", "super_resolution")
    if not os.path.exists(sr_work_space):
        os.makedirs(sr_work_space, exist_ok=True)
    

    interpolate_work_space = os.path.join(root_path, "work_space", "interpolate")
    if not os.path.exists(interpolate_work_space):
        os.makedirs(interpolate_work_space, exist_ok=True)
    

    hdr_work_space = os.path.join(root_path, "work_space", "hdr")
    if not os.path.exists(hdr_work_space):
        os.makedirs(hdr_work_space, exist_ok=True)


    result_dir = os.path.join(root_path, 'result')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    cache_dir = os.path.join(root_path, 'cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)


def get_gpu_memory():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    return gpu_memory[0]


def load_model(model, device, model_path=None):
    if not model_path:
        return model

    load_net = torch.load(model_path, map_location=lambda storage, loc:storage)
    load_net = load_net['state_dict']

    choose_key = 'generator'
    for key, value in deepcopy(load_net).items():
        key_list = key.split('.')

        if choose_key in key_list:
            tmp_key = ".".join(key_list[1:])
            load_net[tmp_key] = value
    
        load_net.pop(key)

    model.load_state_dict(load_net, strict=True)
    return model


def compute_flow_map(model, lrs_ndarray, device=None):
    h, w, c = lrs_ndarray[0].shape

    lrs_zero_to_one = [v.astype(np.float32) / 255. for v in lrs_ndarray]
    lrs_tensor = [torch.from_numpy(v).permute(2,0,1) for v in lrs_zero_to_one]
    
    lrs = torch.cat(lrs_tensor).view(-1, c, h, w).unsqueeze(0)
    
    if device is not None:
        lrs = lrs.to(device)

    _, flow = model(lrs)
    flow = flow.permute(0, 2, 3, 1).detach().cpu().numpy()
    flow = flow[0]
    flow_map = np.uint8(mmcv.flow2rgb(flow) * 255.)
    return flow_map


def save_uploaded_file(uploadedfile):
    """
        Args ??????:
            uploadedfile (streamlit ?????????????????????): ??????streamlit???????????????cache?????????
    """
    with open(os.path.join("cache", uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())


def save_img_with_ratio(image_path, image, alignratio_path):
    align_ratio = (2 ** 16 - 1) / image.max()
    # np.save(alignratio_path, align_ratio)
    uint16_image_gt = np.round(image * align_ratio).astype(np.uint16)
    cv2.imwrite(image_path, uint16_image_gt)
    return uint16_image_gt


def generate_paths(folder, name):
    # id = name[:4]
    id = name.split('.')[0]

    image_path = os.path.join(folder, id+'.png')
    alignratio_path = os.path.join(folder, id+'_alignratio.npy')
    return image_path, alignratio_path


def remaining_time(steps, time_cost):
    """
        ??????????????????
        Args ?????????
            steps (int): ?????????????????????
            time_cost (float): ???????????????????????????
    """
    seconds = steps * time_cost
    extra = "Time"

    if seconds < 60:
        extra = "{:02d} s".format(int(seconds))
    elif seconds >= 60 and seconds < 3600:
        minutes = int(seconds) // 60
        seconds = int(seconds) % 60
        extra = "{:02d} m {:02d} s".format(minutes, seconds)
    elif seconds >= 3600 and seconds < 3600 * 24:
        hours = int(seconds) // (3600)
        minutes = (int(seconds) % 3600) // 60
        seconds = int(seconds) % 60
        extra = "{:02d} h {:02d} m {:02d} s".format(hours, minutes, seconds)

    return extra


def bicubic(frame, scale):
    """
        frame: numpy.ndarray
        scale: int
    """
    if frame is not None:
        h, w, _ = frame.shape
        sr_frame = cv2.resize(frame, [w * scale, h * scale], interpolation=cv2.INTER_CUBIC)
        return sr_frame


def read_imgdata(path, ratio=255.0):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED) / ratio


def tensor2numpy(input_tensor):
    output_numpy = torch.squeeze(input_tensor, dim=0).data.cpu().permute(1,2,0).numpy()
    print("Tensor ==> Numpy  Feature shape: {}".format(output_numpy.shape))
    return output_numpy.astype(np.float32)


def transfer_audio(
                    sourceVideo, 
                    targetVideo, 
                    targetNoAudio):
    """
        Args:
            sourceVideo (str): A video complete path.
            targetVideo (str): A target output file-name.
            targetNoAudio (str): A video without audio.
    """

    random_seed = int(random.randint(0, 4800))
    tempAudioFileName = f"./temp/{random_seed}_audio.mkv"

    # split audio from original video file and store in "temp" directory
    if True:  
        # create new "temp" directory
        if not os.path.exists("temp"):
            os.makedirs("temp", exist_ok=True)
        
        # extract audio from video
        os.system('ffmpeg -y -i "{}" -c:a copy -vn {}'.format(sourceVideo, tempAudioFileName))
    
    # combine audio file and new video file
    combine = os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))

    # if ffmpeg failed to merge the video and audio together try converting the audio to aac
    if os.path.exists(targetVideo):
        if os.path.getsize(targetVideo) == 0: 
            tempAudioFileName = f"./temp/{random_seed}_audio.m4a"
            os.system('ffmpeg -y -i "{}" -c:a aac -b:a 160k -vn {}'.format(sourceVideo, tempAudioFileName))
            os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))
            
            # if aac is not supported by selected format
            if os.path.getsize(targetVideo) == 0: 
                # os.rename(targetNoAudio, targetVideo)
                print("Audio transfer failed. Interpolated video will have no audio")
            else:
                print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")

    if os.path.exists(tempAudioFileName):
        os.remove(tempAudioFileName)

    return combine


def super_resolution_process(cap, 
                            st_lr_frame = None, 
                            st_sr_frame = None, 
                            scale = 2, 
                            progress_bar = None,
                            js1_text = None,
                            js2_text = None,
                            js3_text = None,
                            inf_ov_4_text = None,
                            if_plot_chromaticity = 1,
                            if_save_video_frame = False,
                            **args):
    """ 
        Bicubic ???????????????, ???????????????????????????????????????
        Args ??????:
            cap (mmcv video cap): mmcv?????????cap??????
            st_lr_frame (streamlit ????????????): ?????????????????????????????? 
    
    """
    assert cap is not None, 'Cap is None!'

    current_path = args.get("current_path", None)
    assert current_path is not None, 'current_path is None!'

    root_path = args.get("root_path")  # ???????????????

    if_save_video = args.get("if_save_video", 'No')  # ??????????????????

    video_fps = cap.fps  # ??????????????????
    video_full_name = args.get('video_full_name', 'None')
    
    kpi1_text = args.get('kpi1_text', None)
    kpi1_text.write(str(video_full_name))

    kpi2_text = args.get("kpi2_text", None)
    kpi3_text = args.get("kpi3_text", None)

    video_ffprobe_info = args.get("video_ffprobe_info", None)  # video_ffprobe_info???????????????ffprobe??????
    video_info_dict = process_ffmpeg_info(video_ffprobe_info)  # video_info_dict??????????????????????????????

    video_stream_dict = dict()  # ???????????????
    audio_stream_dict = dict()  # ???????????????

    video_stream_dict['????????????'] = video_info_dict.get('duration', 'Unkown')
    video_stream_dict['????????????'] = video_info_dict.get('bitRate', 'Unkown')
    video_stream_dict['????????????'] = video_full_name.split(".")[1].upper()
    video_stream_dict['????????????'] = check_file_size(os.path.join(root_path, 'cache', video_full_name))
    video_stream_dict['????????????'] = video_info_dict.get('Size', 'Unkown')
    video_stream_dict['???????????????'] = video_info_dict.get('resolution', 'Unkown')
    video_stream_dict['????????????'] = video_info_dict.get('Framesrate', 'Unkown')
    video_stream_dict['????????????'] = video_info_dict.get('videoCodec', 'Unkown')
    video_stream_dict['????????????'] = video_info_dict.get('pixelFormat', 'Unkown')
    video_stream_dict['??????'] = video_info_dict.get('BitDepth', 'Unkown')
    video_stream_dict['SDR/HDR'] = video_info_dict.get('transfer', 'Unkown')
    video_stream_dict['????????????'] = video_info_dict.get('Gamut', 'Unkown')
    video_stream_dict['??????????????????'] = video_info_dict.get('ratio', 'Unkown')
    audio_stream_dict['??????????????????'] = video_info_dict.get('audioCodec', 'Unkown')
    audio_stream_dict['?????????'] = video_info_dict.get("channels", 'Unkown')

    inf_ov_1_text = args.get("inf_ov_1_text")
    inf_ov_2_text = args.get("inf_ov_2_text")
    inf_ov_3_text = args.get("inf_ov_3_text")
    output_info_dict = dict()  # ??????????????????

    if cap is not None:
        for i in range(len(cap)):
            start_time = time.perf_counter()

            lr_frame = cap[i]
            try:
                lr_frame = cv2.cvtColor(lr_frame, cv2.COLOR_BGR2RGB)
                height, width, _ = lr_frame.shape
                output_height, output_width = height * scale, width * scale
                output_info_dict['???????????????'] = f"{output_width}x{output_height}"
                output_info_dict['???????????????'] = scale
            except: continue
            
            # ???????????????
            sr_frame = bicubic(lr_frame, scale=scale)
            if i % 10 == 0:
                st_lr_frame.image(lr_frame, use_column_width=True)
                st_sr_frame.image(sr_frame, use_column_width=True)

            # CPU, GPU??????????????????
            js1_text.write(str(psutil.virtual_memory()[2])+"%")
            js2_text.write(str(psutil.cpu_percent())+'%')
            
            try:
                js3_text.write(str(get_gpu_memory())+' MB')
            except:
                js3_text.write(str('NA'))

            # ???????????????????????????
            try:
                kpi2_text.write(video_stream_dict)
            except:
                kpi2_text.write(str('NA'))
            
            try:
                kpi3_text.write(audio_stream_dict)
            except:
                kpi3_text.write(str('NA'))

            # ????????????????????????
            inf_ov_1_text.write("????????????{}  | ????????????{}".format(i + 1, len(cap)))
            
            # ?????????????????????
            if i % 30 == 0 and if_plot_chromaticity == "Yes":
                h_lr, w_lr, _ = lr_frame.shape
                color_lr = cv2.resize(lr_frame, [w_lr // 4, h_lr // 4])
                color_diagram = plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931(color_lr, filename=None)
                inf_ov_4_text.image(color_diagram, width=200)

            # ???????????????????????????
            sr_frame = cv2.cvtColor(sr_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(current_path, '{:08d}.png'.format(i)), sr_frame)

            end_time = time.perf_counter()
            time_cost = end_time - start_time

            if i % 20 == 0:
                extra_time = remaining_time(len(cap) - i, time_cost)
            
            inf_ov_2_text.write(str(extra_time))
            inf_ov_3_text.write(output_info_dict)  # ??????????????????
            progress_bar.progress(i / (len(cap) + 1))
    
    # ?????????????????????
    progress_bar.progress((len(cap) + 1) / (len(cap) + 1))
    js1_text.write("0.0%")
    js2_text.write("0.0%")
    js3_text.write("0 MB")
    inf_ov_1_text.write("????????????{} | ???????????? {}".format(len(cap), len(cap)))
    inf_ov_2_text.write("0 s")

    st.info("????????????, ?????????????????????...")
    frame2video = 1

    if if_save_video == 'Yes':
        result_save_path = args.get("result_save_path")
        temp_result_save_path = os.path.join(root_path, "temp")
        if not os.path.exists(temp_result_save_path):
            os.makedirs(temp_result_save_path)

        target_no_audio_video = f"{temp_result_save_path}/{video_full_name}"

        frame2video = os.system(f"ffmpeg -y -r {video_fps} -i {current_path}/%08d.png -c:v h264 -pix_fmt yuv420p {target_no_audio_video}")
        source_video = os.path.join(root_path, 'cache', video_full_name)
        
        target_video = f"{result_save_path}/{video_full_name}"
        combine = transfer_audio(sourceVideo=source_video, targetVideo=target_video, targetNoAudio=target_no_audio_video)

        print("combine: ", combine)
        if combine != 0:
            # if combie video and audio fail
            shutil.copy(target_no_audio_video, f"{result_save_path}/{video_full_name}")

    # if os.path.exists(temp_result_save_path):
    #     shutil.rmtree(temp_result_save_path)

    # ??????????????????????????????????????????????????????
    if if_save_video_frame == 'No':
        if os.path.exists(current_path):
            shutil.rmtree(current_path)

    return frame2video


def super_resolution_process_deeplearning_sliding_window(video, 
                                                        st_lr_frame = None, 
                                                        st_sr_frame = None, 
                                                        scale = 4, 
                                                        progress_bar = None,
                                                        js1_text = None,
                                                        js2_text = None,
                                                        input_frames = 5,
                                                        **args):
    """ ???????????????????????????????????????: ???????????? EDVR """
    root_path = args.get("root_path")

    frame_count = len(video)
    h, w, c = video[0].shape

    seq = [x for x in range(-input_frames // 2 + 1, input_frames // 2 + 1)] # [-2, -1, 0, 1, 2]

    # ???????????????
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(root_path, "model", "weight", "edvr", "edvrm_tencent_iter_150000.pth")
    model = EDVRNet(in_channels=3, out_channels=3).to(device)
    model = load_model(model, device, model_path=model_path)

    video_fps = video.fps  # ??????????????????p

    # ????????????????????????
    video_full_name = args.get('video_full_name', 'None')
    kpi1_text = args.get('kpi1_text', None)
    kpi1_text.write(str(video_full_name))

    kpi1_text = args.get('kpi1_text', None)
    kpi1_text.write(str(video_full_name))

    kpi2_text = args.get("kpi2_text", None)
    kpi3_text = args.get("kpi3_text", None)

    video_ffprobe_info = args.get("video_ffprobe_info", None)  # video_ffprobe_info???????????????ffprobe??????
    video_info_dict = process_ffmpeg_info(video_ffprobe_info)  # video_info_dict??????????????????????????????

    video_stream_dict = dict()  # ???????????????
    audio_stream_dict = dict()  # ???????????????

    video_stream_dict['????????????'] = video_info_dict.get('duration', 'Unkown')
    video_stream_dict['????????????'] = video_info_dict.get('bitRate', 'Unkown')
    video_stream_dict['????????????'] = video_full_name.split(".")[1].upper()
    video_stream_dict['????????????'] = check_file_size(os.path.join(root_path, 'cache', video_full_name))
    video_stream_dict['????????????'] = video_info_dict.get('Size', 'Unkown')
    video_stream_dict['???????????????'] = video_info_dict.get('resolution', 'Unkown')
    video_stream_dict['????????????'] = video_info_dict.get('Framesrate', 'Unkown')
    video_stream_dict['????????????'] = video_info_dict.get('videoCodec', 'Unkown')
    video_stream_dict['????????????'] = video_info_dict.get('pixelFormat', 'Unkown')
    video_stream_dict['??????'] = video_info_dict.get('BitDepth', 'Unkown')
    video_stream_dict['SDR/HDR'] = video_info_dict.get('transfer', 'Unkown')
    video_stream_dict['????????????'] = video_info_dict.get('Gamut', 'Unkown')
    video_stream_dict['??????????????????'] = video_info_dict.get('ratio', 'Unkown')
    audio_stream_dict['??????????????????'] = video_info_dict.get('audioCodec', 'Unkown')
    audio_stream_dict['?????????'] = video_info_dict.get("channels", 'Unkown')

    inf_ov_1_text = args.get("inf_ov_1_text")
    inf_ov_2_text = args.get("inf_ov_2_text")
    inf_ov_3_text = args.get("inf_ov_3_text")

    output_info_dict = dict()  # ??????????????????
    tmp_frame = video[0]
    height, width, _ = tmp_frame.shape
    output_height, output_width = height * scale, width * scale
    output_info_dict['???????????????'] = f"{output_width}x{output_height}"
    output_info_dict['???????????????'] = scale

    current_path = args.get("current_path")
    js3_text = args.get("js3_text")
    inf_ov_1_text = args.get("inf_ov_1_text")
    inf_ov_2_text = args.get("inf_ov_2_text")

    # ?????????????????????
    if_plot_chromaticity = args.get("if_plot_chromaticity")
    inf_ov_4_text = args.get("inf_ov_4_text")

    # ???????????????????????????
    try:
        kpi2_text.write(video_stream_dict)
    except:
        kpi2_text.write(str('NA'))
    
    # ???????????????????????????
    try:
        kpi3_text.write(audio_stream_dict)
    except:
        kpi3_text.write(str('NA'))

    for idx in range(frame_count - 1):
        start_time = time.perf_counter()

        lrs_ndarray = []
        check_list = []

        for index in seq:
            if index + idx < 0:
                check_list.append(0)
                lrs_ndarray.append(video[0].copy())
            elif index + idx >= frame_count - 1:
                check_list.append(frame_count - 2)
                lrs_ndarray.append(video[frame_count - 2].copy())
            else:
                check_list.append(index + idx)
                lrs_ndarray.append(video[index + idx].copy())

        lrs_zero_to_one = [v.astype(np.float32) / 255. for v in lrs_ndarray]
        lrs_tensor = [torch.from_numpy(v).permute(2,0,1) for v in lrs_zero_to_one]
        lrs = torch.cat(lrs_tensor).view(-1, c, h, w).unsqueeze(0)

        lrs = lrs.to(device)
        output = chop_forward_dim4(lrs, model, scale=4)
        output_ndarray = output.squeeze(0).detach().cpu()
        sr_frame = output_ndarray.permute(1,2,0).numpy()

        lr_frame = video[idx]
        lr_show = cv2.cvtColor(lr_frame, cv2.COLOR_BGR2RGB)

        if idx % 20 == 0:
            st_lr_frame.image(lr_show, use_column_width=True)
            st_sr_frame.image(lr_show, use_column_width=True)

        # CPU, GPU??????????????????
        js1_text.write(str(psutil.virtual_memory()[2])+"%")
        js2_text.write(str(psutil.cpu_percent())+'%')
        try:
            js3_text.write(str(get_gpu_memory())+' MB')
        except:
            js3_text.write(str('NA'))
        
        # ????????????????????????
        end_time = time.perf_counter()
        time_cost = end_time - start_time
        
        if idx % 5 == 0:
            extra_time = remaining_time(len(video) - idx, time_cost)
        
        inf_ov_2_text.write(str(extra_time))
        inf_ov_3_text.write(output_info_dict)  # ??????????????????

        # ????????????????????????
        inf_ov_1_text.write("????????????{}  | ????????????{}".format(idx + 1, len(video)))

        # ?????????????????????
        if idx % 30 == 0 and if_plot_chromaticity == "Yes":
            h_lr, w_lr, _ = lr_frame.shape
            color_lr = cv2.resize(lr_frame, [w_lr // 4, h_lr // 4])
            color_diagram = plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931(color_lr, filename=None)
            inf_ov_4_text.image(color_diagram, width=200)

        # ???????????????
        progress_bar.progress((idx + 1) / frame_count)
        
        frame_name = os.path.join(current_path, "{:08d}.png".format(idx))
        if scale == 4:
            cv2.imwrite(frame_name, sr_frame * 255.)
        elif scale == 2:
            sr_height, sr_width, _ = sr_frame.shape
            sr_frame = cv2.resize(sr_frame, [sr_width // 2, sr_height // 2], interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(frame_name, sr_frame * 255.)

    # ?????????????????????
    progress_bar.progress((len(video) + 1) / (len(video) + 1))
    js1_text.write("0.0%")
    js2_text.write("0.0%")
    js3_text.write("0 MB")
    inf_ov_1_text.write("????????????{} | ???????????? {}".format(len(video), len(video)))
    inf_ov_2_text.write("0 s")

    st.info("????????????, ?????????????????????...")
    if_save_video = args.get("if_save_video")

    frame2video = 1
    if if_save_video == 'Yes':
        result_save_path = args.get("result_save_path")
        temp_result_save_path = os.path.join(root_path, "temp")
        if not os.path.exists(temp_result_save_path):
            os.makedirs(temp_result_save_path)

        target_no_audio_video = f"{temp_result_save_path}/{video_full_name}"

        frame2video = os.system(f"ffmpeg -y -r {video_fps} -i {current_path}/%08d.png -c:v h264 -pix_fmt yuv420p {target_no_audio_video}")
        source_video = os.path.join(root_path, 'cache', video_full_name)
        
        target_video = f"{result_save_path}/{video_full_name}"
        combine = transfer_audio(sourceVideo=source_video, targetVideo=target_video, targetNoAudio=target_no_audio_video)

        print("combine: ", combine)
        if combine != 0:
            # if combie video and audio fail
            shutil.copy(target_no_audio_video, f"{result_save_path}/{video_full_name}")

    # if os.path.exists(temp_result_save_path):
    #     shutil.rmtree(temp_result_save_path)

    # ??????????????????????????????????????????????????????
    if_save_video_frame = args.get("if_save_video_frame")
    if if_save_video_frame == 'No':
        if os.path.exists(current_path):
            shutil.rmtree(current_path)

    return frame2video


def super_resolution_process_deeplearning_recurrent(*args, **kwargs):
    """ 
        ?????????BasicVSR, BasicVSR++???????????????, ?????????????????????EDVR???????????????
    """
    video = kwargs.get("video")
    assert video is not None
    frame_count = len(video)  # ????????????
    root_path = kwargs.get("root_path")  # ?????????????????????

    # ???????????????
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    your_method = kwargs.get("your_method")

    if your_method == "BasicVSR":
        model = BasicVSRNet().to(device)
    elif your_method == "BasicVSR++":
        model = BasicVSRPlusPlus().to(device)
        model_path = os.path.join(root_path, "model", "weight", "basicvsr_pp", "basicvsr_pp.pth")
        model = load_model(model, device, model_path=model_path)
        model.eval()
    else:
        model = BasicVSRPlusPlus_Gauss().to(device)
        model_path = os.path.join(root_path, "model", "weight", "basicvsr_pp_gauss_focal", "iter_250000.pth")
        model = load_model(model, device, model_path=model_path)
        model.eval()
    
    # ????????????????????????
    video_full_name = kwargs.get('video_full_name', 'None')
    kpi1_text = kwargs.get('kpi1_text', None)
    kpi1_text.write(str(video_full_name))

    # ???????????????????????????????????????
    video_ffprobe_info = kwargs.get("video_ffprobe_info", None)  # video_ffprobe_info???????????????ffprobe??????
    video_info_dict = process_ffmpeg_info(video_ffprobe_info)  # video_info_dict??????????????????????????????

    video_stream_dict = dict()  # ???????????????
    audio_stream_dict = dict()  # ???????????????

    video_stream_dict['????????????'] = video_info_dict.get('duration', 'Unkown')
    video_stream_dict['????????????'] = video_info_dict.get('bitRate', 'Unkown')
    video_stream_dict['????????????'] = video_full_name.split(".")[1].upper()
    video_stream_dict['????????????'] = check_file_size(os.path.join(root_path, 'cache', video_full_name))
    video_stream_dict['????????????'] = video_info_dict.get('Size', 'Unkown')
    video_stream_dict['???????????????'] = video_info_dict.get('resolution', 'Unkown')
    video_stream_dict['????????????'] = video_info_dict.get('Framesrate', 'Unkown')
    video_stream_dict['????????????'] = video_info_dict.get('videoCodec', 'Unkown')
    video_stream_dict['????????????'] = video_info_dict.get('pixelFormat', 'Unkown')
    video_stream_dict['??????'] = video_info_dict.get('BitDepth', 'Unkown')
    video_stream_dict['SDR/HDR'] = video_info_dict.get('transfer', 'Unkown')
    video_stream_dict['????????????'] = video_info_dict.get('Gamut', 'Unkown')
    video_stream_dict['??????????????????'] = video_info_dict.get('ratio', 'Unkown')
    audio_stream_dict['??????????????????'] = video_info_dict.get('audioCodec', 'Unkown')
    audio_stream_dict['?????????'] = video_info_dict.get("channels", 'Unkown')

    # ???????????????????????????
    kpi2_text = kwargs.get("kpi2_text")
    try:
        kpi2_text.write(video_stream_dict)
    except:
        kpi2_text.write(str('NA'))
    
    # ???????????????????????????
    kpi3_text = kwargs.get("kpi3_text")
    try:
        kpi3_text.write(audio_stream_dict)
    except:
        kpi3_text.write(str('NA'))
    
    # ??????????????????
    inf_ov_3_text = kwargs.get("inf_ov_3_text")

    scale = kwargs.get("scale")
    output_info_dict = dict()  
    tmp_frame = video[0]
    height, width, _ = tmp_frame.shape
    output_height, output_width = height * scale, width * scale
    output_info_dict['???????????????'] = f"{output_width}x{output_height}"
    output_info_dict['???????????????'] = scale
    inf_ov_3_text.write(output_info_dict)  # ??????????????????

    js1_text = kwargs.get("js1_text")
    js2_text = kwargs.get("js2_text")
    js3_text = kwargs.get("js3_text")
    inf_ov_1_text = kwargs.get("inf_ov_1_text")
    inf_ov_2_text = kwargs.get("inf_ov_2_text")
    inf_ov_4_text = kwargs.get("inf_ov_4_text")
    if_plot_chromaticity = kwargs.get("if_plot_chromaticity")

    # ?????????????????????????????????
    step = 0
    input_frames = kwargs.get("input_frames")
    tmp_frame = video[0]
    h, w, c = tmp_frame.shape

    current_path = kwargs.get("current_path")
    progress_bar = kwargs.get("progress_bar") # ?????????

    st_lr_frame = kwargs.get("st_lr_frame")
    st_sr_frame = kwargs.get("st_sr_frame")

    
    last_stop = False
    while step < frame_count:
        start_time = time.perf_counter()

        lr_frame = video[step]
        lr_show = cv2.cvtColor(lr_frame, cv2.COLOR_BGR2RGB)
        if step % 2 == 0:
            st_lr_frame.image(lr_show, use_column_width=True)
            st_sr_frame.image(lr_show, use_column_width=True)

        if step + input_frames >= frame_count:
            lrs_ndarray = video[frame_count - 1 - input_frames : frame_count - 1]
            step = frame_count - input_frames - 1
            last_stop = True  # ???????????????????????????????????????????????????
        else:
            sliding_window = input_frames
            lrs_ndarray = video[step : step + sliding_window]

        lrs_zero_to_one = [v.astype(np.float32) / 255. for v in lrs_ndarray]
        lrs_tensor = [torch.from_numpy(v).permute(2,0,1) for v in lrs_zero_to_one]
        lrs = torch.cat(lrs_tensor).view(-1, c, h, w).unsqueeze(0)

        # ???????????????GPU?????????
        lrs = lrs.to(device)
        output = chop_forward_dim5(lrs, model)
        output_ndarray = output.squeeze(0).detach().cpu()

        # CPU, GPU??????????????????
        js1_text.write(str(psutil.virtual_memory()[2])+"%")
        js2_text.write(str(psutil.cpu_percent())+'%')
        try:
            js3_text.write(str(get_gpu_memory())+' MB')
        except:
            js3_text.write(str('NA'))
        
        # ????????????????????????
        inf_ov_1_text.write("????????????{}  | ????????????{}".format(step + 1, len(video)))

        # ?????????????????????
        if step % 2 == 0 and if_plot_chromaticity == "Yes":
            h_lr, w_lr, _ = lr_frame.shape
            color_lr = cv2.resize(lr_frame, [w_lr // 4, h_lr // 4])
            color_diagram = plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931(color_lr, filename=None)
            inf_ov_4_text.image(color_diagram, width=200)

        frame_index = step
        for i in range(sliding_window):
            sr_frame = output_ndarray[i].permute(1,2,0).numpy()
            frame_name = os.path.join(current_path, "{:08d}.png".format(frame_index))
            # ???????????????
            try:
                if scale == 4:
                    cv2.imwrite(frame_name, sr_frame * 255.)
                elif scale == 2:
                    sr_height, sr_width, _ = sr_frame.shape
                    sr_frame = cv2.resize(sr_frame, [sr_width // 2, sr_height // 2], interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(frame_name, sr_frame * 255.)
                frame_index += 1
            except:
                pass
        step += input_frames

        # ????????????????????????
        end_time = time.perf_counter()
        time_cost = end_time - start_time
        extra_time = remaining_time(len(video) - step, time_cost)
        inf_ov_2_text.write(str(extra_time))

        # ???????????????????????????????????????
        if last_stop:
            break

        progress_bar.progress((step + 1) / frame_count)

    # ?????????????????????
    progress_bar.progress((step + 1) / frame_count)
    js1_text.write("0.0%")
    js2_text.write("0.0%")
    js3_text.write("0 MB")
    inf_ov_1_text.write("????????????{} | ???????????? {}".format(len(video), len(video)))
    inf_ov_2_text.write("0 s")

    st.info("????????????, ?????????????????????...")
    if_save_video = kwargs.get("if_save_video")
    video_fps = video.fps
    
    frame2video = 1
    if if_save_video == 'Yes':
        result_save_path = kwargs.get("result_save_path")
        temp_result_save_path = os.path.join(root_path, "temp")
        if not os.path.exists(temp_result_save_path):
            os.makedirs(temp_result_save_path)

        target_no_audio_video = f"{temp_result_save_path}/{video_full_name}"

        frame2video = os.system(f"ffmpeg -y -r {video_fps} -i {current_path}/%08d.png -c:v h264 -pix_fmt yuv420p {target_no_audio_video}")
        source_video = os.path.join(root_path, 'cache', video_full_name)
        
        target_video = f"{result_save_path}/{video_full_name}"
        combine = transfer_audio(sourceVideo=source_video, targetVideo=target_video, targetNoAudio=target_no_audio_video)

        print("combine: ", combine)
        if combine != 0:
            # if combie video and audio fail
            shutil.copy(target_no_audio_video, f"{result_save_path}/{video_full_name}")


    # ??????????????????????????????????????????????????????
    if_save_video_frame = kwargs.get("if_save_video_frame")
    if if_save_video_frame == 'No':
        if os.path.exists(current_path):
            shutil.rmtree(current_path)

    return frame2video


def convert_bytes(num):
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


def check_file_size(file_path):
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return convert_bytes(file_info.st_size) 


def process_ffmpeg_info(info):
    """
        Args ?????????
            info (dict): ffmpeg.probe ???????????????????????????
    """

    for i in range(len(info['streams'])):
        if info['streams'][i]['codec_type'] == 'video':
            video_resolution = str(info['streams'][i]['width']) + 'x' + str(info['streams'][i]['height'])
            l1, l2 = int(info['streams'][i]['r_frame_rate' ].split('/')[0]), int(info['streams'][i]['r_frame_rate'].split('/')[1])
            video_frame_rate = str(round(l1 / l2)) + ' fps'

            try:
                video_pix_fmt = info['streams'][i]['pix_fmt'].upper()
                if video_pix_fmt == 'YUV420P':
                    video_color_depth = '12bits'
                elif video_pix_fmt == 'YUV422P':
                    video_color_depth = '16bits'
                elif video_pix_fmt == 'YUV444P':
                    video_color_depth = '32bits'
                else:
                    video_color_depth = '8bits'
            except:
                video_pix_fmt = 'Unkown'
                video_color_depth = 'Unkown'

            try:
                video_codec_name = info['streams'][i]['codec_name'].upper()
            except:
                video_codec_name = 'Unkown'

            try:
                video_colour_space = info['streams'][i]['color_space'].upper()
            except:
                video_colour_space = 'BT709'
            
            try: 
                bits_per_raw_sample = info['streams'][i]['bits_per_raw_sample'].upper()
            except:
                bits_per_raw_sample = 'Unkown'

            try:
                video_transfer = info['streams'][i]['ransfer']
                video_transfer = 'HDR'
            except:
                video_transfer = 'SDR'
            
            try:
                display_aspect_ratio = info['streams'][i]['display_aspect_ratio']
            except:
                display_aspect_ratio = 'Unkown'
            
            try:
                duration = float(info['streams'][i]['duration'])
                seconds = "{:.2f} s".format(duration)
                hours = int(duration / 3600)
                minute = int((duration - hours * 3600) / 60)
                second = duration - hours*3600 - minute*60
                duration = '{:02d}:{:02d}:{:.2f}'.format(hours, minute, second)
            except:
                duration = 'Unkown'
                seconds = 'Unkown'
            
            try:
                nb_frames = info['streams'][i]['nb_frames']
            except:
                nb_frames = 'Unkown'
            
            try:
                bit_rate = int(info['streams'][i]['bit_rate']) // 1000
                bit_rate = str(bit_rate) + ' kb/s' 
            except:
                bit_rate = 'Unkown'
        
        audio_channels = 'Unkown'
        audio_coding = 'Unkown'
        if info['streams'][i]['codec_type'] == 'audio':
            audio_coding = info['streams'][i]['codec_name'].upper()
            try: 
                audio_channels = info['streams'][i]['channels']
            except: 
                audio_channels = 'Unkown'

    info_dict = dict(
        duration = duration,  # ??????
        bitRate = bit_rate,  # ????????????
        frames = nb_frames,   # ????????????
        resolution = video_resolution,  # ?????????
        Framesrate = video_frame_rate,  # ???????????? 
        videoCodec = video_codec_name,   # ?????????????????? h264 ?????? hevc
        pixelFormat = video_pix_fmt,     # ????????? yuvj420p
        BitDepth = bits_per_raw_sample,  # ???????????? ?????????????????????
        transfer = video_transfer,  # SDR ?????? HDR
        seconds = seconds,  # ???????????????
        Gamut = video_colour_space,  # ?????? bt709 ??????bt2020
        ratio = display_aspect_ratio, # ??????????????????
        audioCodec = audio_coding,   # ?????????????????????
        channels = audio_channels, # ??????????????????
    )

    return info_dict


def optical_flow_visualization():
    pass