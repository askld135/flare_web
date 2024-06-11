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
        tab1, tab2, tab3,tab4 = st.tabs(["Introduction" ,"Flare Remove", "Instance Segmentation", "Depth Estimation"])
        
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
            #### &rarr; :red[**소프트 웨어적인 방법이 필요**]
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
                st.title("Flare Remove")
                st.image("./image/flare_removal.png")
                st.markdown(
            """
            #### 기존의 연구
            ##### How to Train Neural Networks for Flare Removal
            """
            )  
                st.image("./image/flare_removal_1.png") 
                st.markdown(
            """
            - 빛 번짐 제거 학습을 위한 데이터셋을 증강시키기 위해 Clean image에 Flare를 합성시키는 방식을 사용  
            합성한 이미지를 U-Net구조의 플레어 제거 네트워크에 넣어 플레어를 제거한 이미지를 얻음.
            - 장면 내에 광원이 포함되었을 때 발생하는 어려움을 masking 기법을 이용해 해결  
            """
            )
                st.markdown(
            """ 
            #### 수행 방법
            ##### 1. Network 변경 : **U-Net구조 &rarr; NaFNet**
            """        
            )
                st.image("./image/flare_removal_2.png")
                st.markdown(
            """
            -  Image Restoration 분야에서 좋은 성능을 보이는 Deep Learning 네트워크로 비선형 활성화 함수를 사용하지 않음으로 인해 **연산량 감소** 및 **구조 단순화**
            ##### 2. 데이터 전처리 : NAFNet 구조의 모델이 여러 개의 광원을 인식하지 못하는 문제를 해결하기 위한 기법
            """        
            )
                st.image("./image/flare_removal_3.png")
                st.markdown(
            """
            How to Train Neural Networks for Flare Removal에서의 플레어 합성 이미지는 이미지 하나당 플레어가 하나가 합성된다.  
            NAFNet 구조의 모델이 여러 개의 광원을 인식하지 못하는 문제를 해결하기 위하여 무작위로 여러 개의 빛 번짐을 합성하여 학습한다.  
            광원이 많은 경우 보통 작은 광원이 여럿 찍히는 경우가 일반적이므로 광원 수에 따라 광원의 크기를 제한한다.  
            
            ##### 3. Loss 추가
            """
            )
                st.image("./image/flare_removal_4.png")
                st.markdown(
            """
            - 모델을 통해 빛 번짐을 제거한 사진과 annotation(광원)을 배경 이미지에 합성한 사진을 비교하는 손실함수  

            마스킹되어 비교되지 않은 부분에 대한 손실을 계산하여 더 풍부한 정보를 제공한다.
            - **SSIM**과 **L1**을 결합한 형태의 새로운 손실함수 도입, 기존 손실함수와 **1:2**의 비율로 사용
            
            ##### 4. 실험 결과
            """        
            )
                st.image("./image/flare_removal_5.png")
                st.image("./image/flare_removal_6.png")
                
            with empty2:
                empty()
        with tab3:
            empty1,con1,empty2 = st.columns([0.1,1.0,0.1])
            
            with empty1:
                empty()
            with con1:
                st.title("Instance Segmentation")
                st.image("./image/segmentation_1.png")
                st.markdown(
            """
            #### 개별 객체마다 클래스 분류 / 객체 분할을 수행하는 연구 분야
            ##### Object Detection 기술과 Semantic Segmentation 주요 특징을 합친 형태  
            - Semantic Segmentation의 경우, 각 픽셀마다 어떤 클래스에 해당하는지 판별한다. 
            이때, 객체마다 수행하는 것이 아니기 때문에 사진에서는 양이 세 마리가 존재하지만, 결과를 보시면 하나의 덩어리로 양이라고 판별합니다.
            - 하지만 Instance Segmentation의 경우에 개별 객체에 대한 분할이 이루어지며, 양 세마리를 구분한 것을 확인할 수 있습니다.
            """
            )
                st.image("./image/instance_segmentation.png") 
                st.markdown(
            """
            #### 수행방법
            ##### Spatial Attention
            """        
            )
                st.image('./image/Spatial_Attention.png')
                st.markdown(
            """
            이미지의 **공간적 위치**에 대한 중요도를 학습하여, **모델이 더 중요한 부분에 집중하도록 유도한다.**
            - 빛 번짐이 존재하는 이미지에서 좋은 특징을 추출할 구 있도록 하여 빛번짐에 강건한 모델을 설계
            
            ##### Spatial Attention ResNet
            """        
            )
                st.image('./image/Spatial_Attention_ResNet.png')
                st.markdown(
            """
            - ResNet-50 4개의 레이어 중 마지막 레이어 만을 제외하고 적용
            - 사전 학습된 ResNet-50 가중치 사용
            ##### 실험 결과
            - Flare-corrupted COCO 데이터세트 학습 (train: 약 118,000장, val: 5,000장)
            - ResNet-50 백본, Light SOLOv2 모델 사용
            - Heatmap을 통해 공간 어텐션을 적용한 모델이 더욱 특징을 잘 추출하는 것 확인
            """        
            )
                st.image('./image/segmentation_result.png')
                st.image('./image/segmentation_result2.png')
            
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
            #### 주어진 영상에서 3차원 공간에서의 깊이 정보를 측정하는 task
            - 자율주행, 로봇 비전, 환경 인식 등 다양한 분야에서 활용
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
         
        
        selected2 = option_menu(None, ["Flare Removal", "Instance Segmentation","Depth Estimation"], 
            icons=['bi bi-sun', 'bi bi-search', "bi bi-rulers"], 
            menu_icon="cast", default_index=0, orientation="horizontal")
    
    
    
        if selected2 == "Flare Removal":
            
            st.title("Flare Removal")
            file = st.file_uploader( "이미지를 올려주세요",type = ['jpg', 'png'])  # 파일을 첨부하는 영역

            to_tensor = transforms.ToTensor()

            if file is None:
                st.text('')
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
    
    
        elif selected2 == "Instance Segmentation":
            
            st.title("Instance Segmentation")
            file = st.file_uploader( "이미지를 올려주세요",type = ['jpg', 'png'])  # 파일을 첨부하는 영역

            to_tensor = transforms.ToTensor()

            if file is None:
                st.text('')
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
            
            
                
        elif selected2 =="Depth Estimation":
            st.title("Depth Estmation")
            file = st.file_uploader( "이미지를 올려주세요",type = ['jpg', 'png'])  # 파일을 첨부하는 영역

            to_tensor = transforms.ToTensor()

            if file is None:
                st.text('')
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
        