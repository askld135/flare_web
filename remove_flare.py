import os
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import synthesis
from networks import *
import utils

def remove_flare(model, image):

    # model 불러오기
    ckp_path = './pretrained/mul_loss_NAF_020.pt'

    if os.path.isfile(ckp_path):
        print("Loading model from", ckp_path)
        if model == 'NAFNet':
            model = NAFNet().cuda()
        
        ckp = torch.load(ckp_path, map_location=torch.device("cpu"))
        model.load_state_dict(ckp["g"])
        model.eval()
    else: raise Exception("Can't find args.ckp_path: {}".format(ckp_path))
    
    inputs = image
    _, w, h = inputs.shape

    inputs = inputs.cuda().unsqueeze(0)     # (1,3,h,w)


    if min(h, w) >= 2048:
        inputs = T.functional.resize(inputs, [2048,2048])  # resize
        inputs_low = F.interpolate(inputs, (512,512), mode='area')
        pred_scene_low = model(inputs_low).clamp(0.0, 1.0)
        pred_flare_low = synthesis.remove_flare(inputs_low, pred_scene_low)
        pred_flare = T.functional.resize(pred_flare_low, [2048,2048], antialias=True)
        pred_scene = synthesis.remove_flare(inputs, pred_flare)
    else:
        inputs = T.functional.resize(inputs, [512,512])    # resize
        pred_scene = model(inputs).clamp(0.0, 1.0)
        pred_flare = synthesis.remove_flare(inputs, pred_scene)

    pred_blend = synthesis.blend_light_source(inputs.cpu(), pred_scene.cpu())

    return dict(
        input=inputs.cpu(),
        pred_blend=pred_blend.cpu(),
        pred_scene=pred_scene.cpu(),
        pred_flare=pred_flare.cpu()
    )