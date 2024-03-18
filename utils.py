import os
import torchvision
import torchvision.transforms as T

def save_outputs(result, idx, resolution=512):
    for k in ["input", "pred_blend", "pred_scene", "pred_flare"]:
        image = result[k]
        if max(image.shape[-1], image.shape[-2]) > resolution:
            image = T.functional.resize(image, [resolution, resolution], antialias=True)

        result_path = './result'
        
        seperate_path = os.path.join(result_path, k)
        if not os.path.exists(seperate_path):
             os.makedirs(seperate_path)
        torchvision.utils.save_image(image, seperate_path+ '/' + idx + '.jpg')