from mmdet.apis import init_detector, inference_detector
import os
import cv2
import numpy as np
from mmdet.registry import VISUALIZERS

def get_detected_img(model,
                     img_array,
                     score_threshold=0.4,
                     is_print=False) -> np.ndarray:
    global total_time
    draw_img = img_array.copy()
    results = inference_detector(model, draw_img)

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