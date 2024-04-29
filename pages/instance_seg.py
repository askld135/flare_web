import streamlit as st
import os
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
from networks import *
import remove_flare
import utils
import matplotlib.pyplot as plt
from inference import get_detected_img
from mmdet.apis import init_detector
from mmdet.registry import VISUALIZERS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NAFNet().to(device)

st.title("빛번짐 제거를 이용한 향상된 객체 분할")
file = st.file_uploader("이미지를 올려주세요.", type = ['jpg', 'png'])  # 파일을 첨부하는 영역

to_tensor = transforms.ToTensor()

if file is None:
    st.text('이미지를 먼저 올려주세요')
else:
    columns = st.columns(2)
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
        breakpoint()
        ins_dir = './ins_dir'
        os.makedirs(ins_dir, exist_ok=True)
        ins_path = os.path.join(ins_dir, f'ins.png')
        plt.imsave(ins_dir, pred)
        columns[1].image(Image.open('./ins_dir/ins.png'), caption="Instance Segmentation", use_column_width=True)