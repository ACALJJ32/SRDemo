import os
import mmcv
import ffmpeg
import tempfile
import streamlit as st
import torch
import pynvml
from utils.utils import super_resolution_process, super_resolution_process_deeplearning_sliding_window, super_resolution_process_deeplearning_recurrent
from utils.utils import save_uploaded_file

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
percent_cost = meminfo.used / meminfo.total

st.set_page_config(page_title="视频超分辨率")

#  Slidebar Part
root_path = os.getcwd()
st.sidebar.title("参数设置")

your_method =  st.sidebar.selectbox('请选择视频超分模型',
    ('Bicubic', 'EDVR', 'BasicVSR_PlusPlus'))

scale = st.sidebar.radio("放大倍数",
                   ("X2","X4"), index=0)

if scale == "X2": scale = 2
elif scale == 'X4': scale = 4

if_save_video = st.sidebar.radio("是否保存视频?",
                   ("Yes","No"), index=0)

if_save_video_frame = st.sidebar.radio("是否保存视频帧?",
                   ("Yes","No"), index=1)

if_plot_chromaticity = st.sidebar.radio("是否绘制色品图?",
                   ("Yes","No"), index=1)

input_video = st.sidebar.file_uploader("请上传视频文件")
start_process =  st.sidebar.button("开始处理!")

st.title("视频超分辨率处理指示板")   # 设定标题
col1, col2 = st.columns(2)

st.subheader("输入视频参数")
kpi1, kpi2, kpi3 = st.columns(3)

st.subheader("系统统计数据")
js1, js2, js3 = st.columns(3)


# Show LR frame and SR frame
with col1:
    st.markdown("**低分辨率视频**")
    st_lr_frame = st.empty()

with col2:
    st.markdown("**高分辨率视频**")
    st_sr_frame = st.empty()

# Updating Inference results
with kpi1:
    st.markdown("**视频输入名**")
    kpi1_text = st.markdown("0")
    fps_warn = st.empty()

with kpi2:
    st.markdown("**视频流参数**")
    kpi2_text = st.markdown("0")

with kpi3:
    st.markdown("**音频流参数**")
    kpi3_text = st.markdown("0")

# Updating System stats
with js1:
    st.markdown("**内存使用**")
    js1_text = st.markdown("0")

with js2:
    st.markdown("**CPU 利用率**")
    js2_text = st.markdown("0")

with js3:
    st.markdown("**GPU 显存占用**")
    js3_text = st.markdown("0")

st.subheader("超分辨率推理详情")
inf_ov_1, inf_ov_2, inf_ov_3 = st.columns(3)

with inf_ov_1:
    st.markdown("**当前视频帧位置**".format(2))
    inf_ov_1_text = st.markdown("0")

with inf_ov_2:
    st.markdown("**预计剩余时间**")
    inf_ov_2_text = st.markdown("0")

with inf_ov_3:
    st.markdown("**输出参数**")
    inf_ov_3_text = st.markdown("0")


cap = None
video_name = None
if input_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(input_video.read())

    # cap的文件名
    video_full_name = input_video.name
    video_name = input_video.name
    video_name = video_name.split('.')[0].strip('')
    
    save_uploaded_file(input_video)  # 将临时文件保存在cache下面
    cache_video_dir = os.path.join(root_path, 'cache', video_full_name)
    video_ffprobe_info = ffmpeg.probe(cache_video_dir)  # 使用ffmpeg-python读取cahce下面的临时文件
    cap = mmcv.VideoReader(tfile.name)   # 使用mmcv控件读取视频流，本质上和opencv-cap一致
    
    if cap:
        height, width, _ = cap[0].shape


# 开始处理视频
if start_process:
    if cap is None:
        st.warning("请上传视频!")
    else:
        if percent_cost >= 20:
            st.warning("GPU out of Memory, please wait!")
        elif min(height, width) >= 540:
            st.warning("Input video should smaller than 540p!")
        elif len(cap) // int(cap.fps) > 900:
            st.warning("Video Duration is too long! Less than 15 minutes is recommended!")
        else:
            current_path = os.path.join(root_path, "work_space", "super_resolution", your_method.lower(), video_name)  # 保存视频帧的路径
            current_full_path = os.path.join(root_path, "result", your_method.lower(), video_full_name)
            result_save_path = os.path.join(root_path, "result", your_method.lower())                                  # 保存超分结果的路径

            os.makedirs(current_path, exist_ok=True)
            os.makedirs(result_save_path, exist_ok=True)

            if if_plot_chromaticity == "Yes":
                st.sidebar.write("----")
                st.markdown("**色品图**")
                inf_ov_4_text = st.empty()
            else: inf_ov_4_text = None

            st.markdown("**处理进度**")
            progress_bar = st.progress(0.)

            # 选择对应超分的方法
            if your_method == 'Bicubic':
                video2frame = super_resolution_process(cap=cap, 
                                                        st_lr_frame=st_lr_frame, 
                                                        st_sr_frame=st_sr_frame, 
                                                        scale=scale, 
                                                        progress_bar=progress_bar,
                                                        js1_text = js1_text,
                                                        js2_text = js2_text,
                                                        js3_text = js3_text,
                                                        if_plot_chromaticity = if_plot_chromaticity,
                                                        inf_ov_4_text = inf_ov_4_text,
                                                        if_save_video_frame = if_save_video_frame,
                                                        if_save_video = if_save_video,
                                                        current_path = current_path,
                                                        video_full_name = video_full_name,
                                                        result_save_path = result_save_path,
                                                        your_method = your_method,
                                                        kpi1_text = kpi1_text,
                                                        kpi2_text = kpi2_text,
                                                        kpi3_text = kpi3_text,
                                                        root_path = root_path,
                                                        video_ffprobe_info=video_ffprobe_info,
                                                        inf_ov_1_text = inf_ov_1_text,
                                                        inf_ov_2_text = inf_ov_2_text, 
                                                        inf_ov_3_text = inf_ov_3_text)
            elif your_method == 'EDVR':
                video2frame = super_resolution_process_deeplearning_sliding_window(video = cap, 
                                                                        st_lr_frame = st_lr_frame, 
                                                                        st_sr_frame = st_sr_frame, 
                                                                        scale = scale, 
                                                                        progress_bar=progress_bar,
                                                                        js1_text = js1_text,
                                                                        js2_text = js2_text,
                                                                        js3_text = js3_text,
                                                                        if_plot_chromaticity = if_plot_chromaticity,
                                                                        inf_ov_4_text = inf_ov_4_text,
                                                                        if_save_video_frame = if_save_video_frame,
                                                                        if_save_video = if_save_video,
                                                                        current_path = current_path,
                                                                        video_full_name = video_full_name,
                                                                        result_save_path = result_save_path,
                                                                        your_method = your_method,
                                                                        kpi1_text = kpi1_text,
                                                                        kpi2_text = kpi2_text,
                                                                        kpi3_text = kpi3_text,
                                                                        root_path = root_path,
                                                                        video_ffprobe_info = video_ffprobe_info,
                                                                        inf_ov_1_text = inf_ov_1_text,
                                                                        inf_ov_2_text = inf_ov_2_text,
                                                                        inf_ov_3_text = inf_ov_3_text)
            else:
                video2frame = super_resolution_process_deeplearning_recurrent(video = cap, 
                                                                        st_lr_frame = st_lr_frame, 
                                                                        st_sr_frame = st_sr_frame, 
                                                                        scale = scale, 
                                                                        progress_bar=progress_bar,
                                                                        js1_text = js1_text,
                                                                        js2_text = js2_text,
                                                                        js3_text = js3_text,
                                                                        if_plot_chromaticity = if_plot_chromaticity,
                                                                        inf_ov_4_text = inf_ov_4_text,
                                                                        if_save_video_frame = if_save_video_frame,
                                                                        if_save_video = if_save_video,
                                                                        current_path = current_path,
                                                                        video_full_name = video_full_name,
                                                                        result_save_path = result_save_path,
                                                                        your_method = your_method,
                                                                        kpi1_text = kpi1_text,
                                                                        kpi2_text = kpi2_text,
                                                                        kpi3_text = kpi3_text,
                                                                        root_path = root_path,
                                                                        video_ffprobe_info = video_ffprobe_info,
                                                                        inf_ov_1_text = inf_ov_1_text,
                                                                        inf_ov_2_text = inf_ov_2_text,
                                                                        inf_ov_3_text = inf_ov_3_text,
                                                                        input_frames = 6)
            
            if video2frame == 0:
                st.info("推理结束!")
                with open(current_full_path, "rb") as video_file:
                    st.download_button("下载到本地", data=video_file, file_name=os.path.basename(current_full_path))
            else:
                st.warning("视频合成失败!")

torch.cuda.empty_cache()