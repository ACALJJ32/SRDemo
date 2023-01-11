import os
import cv2
import mmcv
import torch
import tempfile
from streamlit_image_comparison import image_comparison
import streamlit as st


def init_posters(video_path=None):
    if not video_path:
        raise 'Value Error!'
    else:
        posters = list()
        videos = os.listdir(video_path)
        videos = [v for v in videos if v.endswith(".mp4")]

        for v in videos:
            cap = cv2.VideoCapture(os.path.join(video_path, v))
            _, frame = cap.read()
            frame = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, [240,160])
            posters.append(frame)
            cap.release()
        return posters


st.write("# 视频对比展示")
root_path = os.getcwd()

col1, col2 = st.columns(2)

with col1:
    video1 = st.file_uploader("请选择低分辨率视频") # 选择上传一个文件
    image_placeholder = st.empty()  # 创建空白块使得图片展示在同一位置

with col2:
    video2 = st.file_uploader("请选择高分辨率视频") # 选择上传一个文件
    image_placeholder2 = st.empty()  # 创建空白块使得图片展示在同一位置

cap_lr, cap_sr = None, None
if video1 is not None:
    tfile_lr = tempfile.NamedTemporaryFile(delete=False)
    tfile_lr.write(video1.read())
    cap_lr = mmcv.VideoReader(tfile_lr.name)  # opencv打开文件

if video2 is not None:
    tfile_sr = tempfile.NamedTemporaryFile(delete=False)
    tfile_sr.write(video2.read())
    cap_sr = mmcv.VideoReader(tfile_sr.name)  # opencv打开文件

if cap_lr is not None and cap_sr is not None:
    show_frame_index = len(cap_lr) // 2
    
    if len(cap_lr) == len(cap_sr):
        st.warning("两个视频的帧数不一致！")
    else:
        "----"
        "#### 逐帧对比"
        my_slider = st.slider("请选择视频帧", 0, len(cap_lr)-1, 0, 1)

        lr_frame = cap_lr[my_slider]
        sr_frame = cap_sr[my_slider]

        lr_frame = cv2.cvtColor(lr_frame, cv2.COLOR_BGR2RGB)
        sr_frame = cv2.cvtColor(sr_frame, cv2.COLOR_BGR2RGB)
        image_comparison(img1=lr_frame, img2=sr_frame, label1="低分辨率视频", label2="高分辨率视频")


torch.cuda.empty_cache()