import streamlit as st
from PIL import Image 

st.title("SSSG 캡스톤 디자인")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    ## 플레어란?
    강렬한 빛이 산란하거나 반사되는 광학 현상으로 이러한 플레어는 이미지의 일부를 가림으로써, 영상이나 이미지의 품질과 이를 이용하는 알고리즘의 성능을 저하시키는 등의 문제가 발생된다.
    ### 플레어제거 프로그램의 필요성"""
)
# 이미지 파일 경로 또는 이미지 객체 리스트
image_paths = ["./image/need1.png", "./image/need2.png"]

# 이미지 객체 리스트 생성 (PIL Image 또는 file buffer 등)
images = [Image.open(image_path) for image_path in image_paths]

columns = st.columns(len(images))

# 각 칼럼에 이미지 배치
for column, image in zip(columns, images):
    column.image(image, caption="Image", use_column_width=True)
    
st.markdown(
    """
    기존의 카메라 빞 번짐 제거 기술은 **물리적인 방법**을 사용하는 것이 대부분이지만, 이는 많은 비용이 들며, 렌즈의 흠집이난 지문등으로 발생한 빛번짐을 제거할 수 없다.  
    #### &rarr; :red[**소프트 웨어적인 방법이 필요하다.**]
    """
)    