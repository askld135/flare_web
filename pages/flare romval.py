from pyngrok import ngrok

ngrok.set_auth_token('2dXcWkKBPZbO8GgIQOuK5r1eBem_7tyXZXJ6rr2FapTYeWbZA')

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
from networks import *
import remove_flare
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NAFNet().to(device)

st.title("빛번짐 제거 웹페이지")
file = st.file_uploader("이미지를 올려주세요.", type = ['jpg', 'png'])  # 파일을 첨부하는 영역

to_tensor = transforms.ToTensor()

if file is None:
    st.text('이미지를 먼저 올려주세요')
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