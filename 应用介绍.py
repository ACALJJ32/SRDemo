import os
import streamlit as st
from utils.utils import check_workspace


st.set_page_config(page_title="视频增强处理应用")
st.sidebar.success("请选择功能")


root_path = os.getcwd()
check_workspace(root_path='./')

# 设定标题
st.title("视频增强处理应用")   
st.image('src/CUC.png', width=700)
st.image('src/NRTA.png', width=700)

# Header 标题
r1c1, r1c2 = st.columns([9,6])
with r1c1:
    f"#### 作者: 中国传媒大学 李金京 (2022.12.9)"
    st.warning("本应用需要使用NVIDIA显卡")
    
with r1c2:
    st.image("src/vx.png".format(52), width=100, caption="交流微信")

"""
    本应用中集成了视频超分辨率、视频插帧、HDR成像等主要功能，同时可以实现超分辨率网络光流可视化对比、超分素材对比等功能。\n
    视频超分辨率、视频插帧主要面向视频，HDR成像面向单帧成像。
"""
    
"---"

# Streamlit 介绍
"## Streamlit介绍"

"""
    [✨Streamlit](https://docs.streamlit.io/)是一个可以用于快速搭建Web应用的Python库。\n
    Streamlit官方的定位是服务于机器学习和数据科学的Web应用框架。\n
    当然，您也可以将其用于给自己的Python程序快速创建美观的GUI。\n
    Streamlit与Markdown有异曲同工之妙。它让创作者专注于后端业务的实现，无需为前端设计分心。\n
"""

"---"

# 超分辨率介绍
"## 视频超分辨率"
"#### 应用界面"

st.image("src/super_resolution.png", width=800)
"#### 应用介绍"
"""
    超分辨率推理中使用到的方法有插值法(Bicubic)、[EDVR](https://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Wang_EDVR_Video_Restoration_With_Enhanced_Deformable_Convolutional_Networks_CVPRW_2019_paper.html)、[BasicVSR](https://arxiv.org/abs/2012.02181)、[BasicVSR++](https://arxiv.org/abs/2104.13371)、GMBasicVSR++(Ours)。\n
    在实现上主要参考了视频超分开源库[mmediting](https://github.com/open-mmlab/mmediting.git)。\n
    绘图过程中色品图的绘制参考了代码库[color](https://blog.csdn.net/weixin_44238733/article/details/118916242)。\n
"""

"---"

# 视频插帧
"## 视频插帧"
"#### 应用界面"

st.image("src/interpolate.png", width=800)
"#### 应用介绍"
"""
    该模块实现主要参考了代码库和算法[RIFE](https://github.com/megvii-research/ECCV2022-RIFE.git) 可以实现快速插帧。 \n
"""

"---"
# HDR成像
"## HDR成像"
"#### 应用界面"

st.image("src/hdr_imaging.png", width=800)
"#### 应用介绍"
"""
    该模块主要集成了两种方法[HDRUNet](https://arxiv.org/pdf/2105.13084.pdf)、与MA-UNet(Ours)。上传单帧图像后，可以进行HDR成像处理，同时具备对比功能。 \n
"""


"---"
# 超分光流可视化面板
"## 超分光流可视化面板"
"#### 应用界面"

st.image("src/flow_vir.png", width=800)
"#### 应用介绍"

"""
    该模块主要对BasicVSR++方法以及作者本人改进方法进行光流可视化对比，光流可视化的原理与实现可以参考应用[mmflow](https://github.com/open-mmlab/mmflow)。 \n
"""

"---"
# 素材对比
"## 超分素材对比"
"#### 应用介绍"
"""
    该模块主要对已经处理好的超分素材进行对比展示。 \n
"""

"---"
"""
    ##### 本项目要感谢我的室友[皮皮](https://blog.csdn.net/Be_racle?type=blog)的指导. \n
    ##### 参考的代码有[mmediting](https://github.com/open-mmlab/mmediting.git)、[santiago911](https://santiago911-streamlit-experience-streamlit-experience-f92buw.streamlit.app/)、[mmflow](https://github.com/open-mmlab/mmflow)、[BaiscVSR-inference](http://t.csdn.cn/LHGlo)。 \n
"""