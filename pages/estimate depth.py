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
from dpt.models import DPTDepthModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NAFNet().to(device)

st.title("빛번짐 제거를 이용한 향상된 깊이 추정")
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