from mmdet.apis import init_detector, inference_detector
import os
import cv2
import numpy as np
from mmdet.registry import VISUALIZERS
import time

total_time = 0

def get_detected_img(model,
                     img_array,
                     score_threshold=0.4,
                     is_print=False) -> np.ndarray:
    global total_time
    draw_img = img_array.copy()
    a = time.time()
    results = inference_detector(model, draw_img)
    b = time.time()
    total_time += (b-a)

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    visualizer.add_datasample(
        'result',
        draw_img,
        data_sample=results,
        draw_gt=None,
        wait_time=0,
        pred_score_thr=score_threshold
    )
    drawn_img = visualizer.get_image()

    if is_print:
        cv2.imshow("sample", drawn_img)

    return drawn_img

data_root = './result/input'
config_path = './pretrained/rtmdet-ins_x_8xb16-300e_coco.py'
weight_path = './pretrained/rtmdet-ins_x_8xb16-300e_coco_20221124_111313-33d4595b.pth'
images_list = os.listdir(data_root)

model = init_detector(config=config_path, checkpoint=weight_path, device='cuda')

for idx in range(len(images_list)):
    image_path = os.path.join(data_root, images_list[idx])
    image = cv2.imread(image_path)
    drawn_img = get_detected_img(model, image)
    cv2.imwrite(f'./data/flare_removal_m/{images_list[idx]}', drawn_img)
    
print("FPS = ", (100/total_time))

cv2.imshow("sample", drawn_img)
cv2.waitKey()