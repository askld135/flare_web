import os
import cv2
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image 
from streamlit import config
from streamlit_option_menu import option_menu
import streamlit as st
from tkinter.tix import COLUMN
from pyparsing import empty

from inference import get_detected_img
from mmdet.apis import init_detector
from mmdet.registry import VISUALIZERS

from dpt.models import DPTDepthModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
from networks import *
import remove_flare
import utils

st.set_page_config(layout="wide")

empty1,con1,empty2 = st.columns([0.2,1.0,0.2])
empty1,con2,con3,empty2 = st.columns([0.2,0.5,0.5,0.2])




with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'Task'], 
        icons=['house', 'gear'], menu_icon="cast", default_index=1)


if selected =="Home":
    with empty1:
        empty()
    with con1:
        tab1, tab2, tab3,tab4 = st.tabs(["Introduction" ,"flare remove", "segmentation", "depth estimation"])
        
        with tab1:
            empty1,con1,empty2 = st.columns([0.1,1.0,0.1])
            empty3, con2, empty4 = st.columns([0.2, 1.0, 0.2])
            
            with empty1:
                empty()
            with con1:
                
                st.title("Introduction")
                st.markdown(
            """
            #### 플레어란?
            강렬한 빛이 산란하거나 반사되는 광학 현상으로 이러한 플레어는 이미지의 일부를 가림으로써, 영상이나 이미지의 품질과 이를 이용하는 알고리즘의 성능을 저하시키는 등의 문제가 발생된다.
            """
            )
                st.image("./image/img1.png")
                st.markdown(
            """
            
            #### 플레어제거 프로그램의 필요성
            기존의 카메라 빞 번짐 제거 기술은 **물리적인 방법**을 사용하는 것이 대부분이지만, 이는 많은 비용이 들며, 렌즈의 흠집이난 지문등으로 발생한 빛번짐을 제거할 수 없다.  
            #### &rarr; :red[**소프트 웨어적인 방법이 필요하다.**]
            """
            )
                
                with empty3:
                    empty()
                with con2:
                    st.image("./image/icon2_1.png")
                with empty4:
                    empty()
            with empty2:
                empty()
                 
                
            
        with tab2:
            empty1,con1,empty2 = st.columns([0.1,1.0,0.1])
            
            with empty1:
                empty()
            with con1:
                st.title("Flare remove")
                st.markdown(
            """
            #### 설명
            
            """
            )
                st.image("./image/flare_removal.png")    
            
            with empty2:
                empty()
        with tab3:
            empty1,con1,empty2 = st.columns([0.1,1.0,0.1])
            
            with empty1:
                empty()
            with con1:
                st.title("Instance Segmentation")
                st.markdown(
            """
            #### 설명
            
            """
            )
                st.image("./image/segmentation.png")    
            
            with empty2:
                empty()
              
        with tab4:
            empty1,con1,empty2 = st.columns([0.1,1.0,0.1])
            
            with empty1:
                empty()
            with con1:
                st.title("Depth Estimation")
                st.markdown(
            """
            #### 설명
            
            """
            )
                st.image("./image/depth.png")    
            
            with empty2:
                empty()
    with empty2:
        empty()
        

        
if selected == "Task":
   
    with empty1:
       empty() 
    with con1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = NAFNet().to(device)
         
        
        selected2 = option_menu(None, ["flare removal", "segmentation","depth estimation"], 
            icons=['bi bi-sun', 'bi bi-search', "bi bi-rulers"], 
            menu_icon="cast", default_index=0, orientation="horizontal")
    
    
    
        if selected2 == "flare removal":
            
            st.title("flare removal")
            file = st.file_uploader( "이미지를 올려주세요",type = ['jpg', 'png'])  # 파일을 첨부하는 영역

            to_tensor = transforms.ToTensor()

            if file is None:
                st.text('이미지를 올려주세요')
            else:
            # 원본 이미지와 빛번짐 제거 결과를 나란히 배치
                columns = st.columns(2)  # 이미지를 2개의 칼럼에 배치하도록 설정

            # 원본 이미지 표시
                columns[0].image(Image.open(file), caption="Original Image", use_column_width=True)

            # 빛번짐 제거 결과 이미지 표시
                img = to_tensor(Image.open(file))
                results = remove_flare.remove_flare(model, img)
                utils.save_outputs(results, 'test')
                columns[1].image(Image.open('./result/pred_blend/test.jpg'), caption="Removed Flare", use_column_width=True)
    
    
        elif selected2 == "segmentation":
            
            st.title("segmentation")
            file = st.file_uploader( "이미지를 올려주세요",type = ['jpg', 'png'])  # 파일을 첨부하는 영역

            to_tensor = transforms.ToTensor()

            if file is None:
                st.text('이미지를 올려주세요')
            else:
            # 원본 이미지와 빛번짐 제거 결과를 나란히 배치
                columns = st.columns(2)  # 이미지를 2개의 칼럼에 배치하도록 설정

            # 원본 이미지 표시
                columns[0].image(Image.open(file), caption="Original Image", use_column_width=True)
                img = to_tensor(Image.open(file))
                results = remove_flare.remove_flare(model, img)
                utils.save_outputs(results, 'test')

                config_path = './pretrained/rtmdet-ins_x_8xb16-300e_coco.py'
                weight_path = './pretrained/rtmdet-ins_x_8xb16-300e_coco_20221124_111313-33d4595b.pth'
    
                removed_image_path = os.path.join('./result/pred_blend', 'test.jpg')
                removed_image = cv2.imread(removed_image_path)
    
                with torch.no_grad():
                    ins_model = init_detector(config=config_path, checkpoint=weight_path, device='cuda')
            
                    pred = get_detected_img(ins_model, removed_image)
        
                    ins_dir = './ins_dir'
                    os.makedirs(ins_dir, exist_ok=True)
                    ins_path = os.path.join(ins_dir, f'ins.png')
                    plt.imsave(ins_path, pred)
                    columns[1].image(Image.open('./ins_dir/ins.png'), caption="Instance Segmentation", use_column_width=True)
            
            
                
        elif selected2 =="depth estimation":
            st.title("depth estmation")
            file = st.file_uploader( "이미지를 올려주세요",type = ['jpg', 'png'])  # 파일을 첨부하는 영역

            to_tensor = transforms.ToTensor()

            if file is None:
                st.text('이미지를 올려주세요')
            else:
             # 원본 이미지와 빛번짐 제거 결과를 나란히 배치
                 columns = st.columns(2)
                 columns[0].image(Image.open(file), caption="Original Image", use_column_width=True)
                 img = to_tensor(Image.open(file))
                 results = remove_flare.remove_flare(model, img)
                 utils.save_outputs(results, 'test')

                 with torch.no_grad():
                    depth_model = DPTDepthModel(
                    path="./pretrained/dpt_hybrid-midas-501f0c75.pt",
                    backbone="vitb_rn50_384",
                    non_negative=True,
                    enable_attention_hooks=False,
                    ).to(device)
                    depth_model.eval()
                    depth_normalize = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
                    net_w, net_h = 384, 384

                    depth_transform = transforms.Compose(
                        [
                            Resize(
                                net_w,
                                net_h,
                                resize_target=None,
                                keep_aspect_ratio=True,
                                ensure_multiple_of=32,
                                resize_method="minimal",
                                image_interpolation_method=cv2.INTER_CUBIC,
                            ),
                            depth_normalize,
                            PrepareForNet(),
                        ]
                    )
                    results = np.uint8(results['pred_blend'].permute(0, 2, 3, 1).numpy()*255)
                    img_input = np.zeros((1, 3, 384, 384), dtype=np.float32)
                    img_input[0] = depth_transform({"image": results[0]})["image"]

                    sample = torch.from_numpy(img_input).to(device)
            
                    prediction = depth_model.forward(sample)
                    prediction = (
                    torch.nn.functional.interpolate(
                            prediction.unsqueeze(1),
                            size=512,
                            mode="bicubic",
                            align_corners=False,
                        )
                        .squeeze()
                        .cpu()
                        .numpy()
                    )
                    normalized_prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min())

                    disparity_map_dir = './disparity_map_dir'
                    os.makedirs(disparity_map_dir, exist_ok=True)
            
                    disparity_map_path = os.path.join(disparity_map_dir, f'disparity_map.png')
                    plt.imsave(disparity_map_path, normalized_prediction, cmap='gray')
                    columns[1].image(Image.open('./disparity_map_dir/disparity_map.png'), caption="Depth Estimation", use_column_width=True)
            
                   
                
    with empty2:
        empty()
        