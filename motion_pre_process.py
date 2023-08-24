import sys
import torch
import numpy as np
import os.path as osp
from omegaconf import OmegaConf
import torch.nn.functional as F
import importlib
from torchvision.transforms import ToTensor, ToPILImage, Resize
from torchvision.utils import save_image
from PIL import Image

import cv2
import numpy
import torch

sys.path.append('.')

def motion(w, flow):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    flow_x = flow[0,:,:]
    flow_y = flow[1,:,:]
    flow_s_x=flow_x**2
    flow_s_y=flow_y**2
    flow_m=(flow_s_y+flow_s_x)**0.5
    
    resize = Resize((w,w))
    flow_m = torch.tensor(flow_m).to(device).unsqueeze(0)
    flow_m = resize(flow_m).unsqueeze(3)
    return flow_m


def motion_y(w, flow):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    if flow.shape[0] == 2:
        flow = torch.permute(flow,(1,2,0))

    flow_y_mask = flow[:,:,1:] < 0

    flow = flow*flow_y_mask
    flow_m = l_2_norm(flow)

    resize = Resize((w,w))
    flow_m = torch.tensor(flow_m).to(device).unsqueeze(0)
    flow_m = resize(flow_m).unsqueeze(3)

    return flow_m

def motion_x(w, flow):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if flow.shape[0] == 2:
        flow = torch.permute(flow,(1,2,0))
        
    flow_x_mask = flow[:,:,:1] < 0
    
    flow = flow*flow_x_mask

    flow_m = l_2_norm(flow)

    resize = Resize((w,w))
    flow_m = torch.tensor(flow_m).to(device).unsqueeze(0)
    flow_m = resize(flow_m).unsqueeze(3)

    return flow_m



def l_1_norm(flow):
    flow_x = flow[:,:,0]
    flow_y = flow[:,:,1]
    abs_x = np.abs(flow_x)
    abs_y = np.abs(flow_y)
    l_1_norm = abs_x + abs_y

    return l_1_norm

def l_2_norm(flow):
    flow_x = flow[:,:,0]
    flow_y = flow[:,:,1]
    flow_s_x=flow_x**2
    flow_s_y=flow_y**2
    l_2_norm=(flow_s_y+flow_s_x)**0.5

    return l_2_norm


def l_infinity_norm(flow):
    flow_x = flow[:,:,0]
    flow_y = flow[:,:,1]
    diff_matrix = torch.abs(flow_x - flow_y)
    max_norm = torch.max(diff_matrix.sum(dim=1))
    l_infinity_norm = np.full((flow.shape[0], flow.shape[1]), max_norm)

    return l_infinity_norm


def magnitude(w, flow):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    flow_x = flow[0,:,:]
    flow_y = flow[1,:,:]
    flow_s_x=flow_x**2
    flow_s_y=flow_y**2
    flow_m=(flow_s_y+flow_s_x)**0.5
    resize = Resize((w,w))
    flow_m = torch.tensor(flow_m).to(device).unsqueeze(0)
    flow_m = resize(flow_m).unsqueeze(3)
    print(flow_m.shape)
    return flow_m



def template_matching_ncc(src, temp):
    h, w = src.shape[1:3]
    ht, wt = temp.shape[1:3]

    score = np.empty((h-ht+1, w-wt+1))

    src.cpu()

    src = np.array(src.cpu(), dtype="float")
    temp = np.array(temp.cpu(), dtype="float")

    for dy in range(0, h - ht+1):
        for dx in range(0, w - wt+1):
            roi = src[dy:dy + ht, dx:dx + wt]
            num = np.sum(roi * temp)
            den = np.sqrt( (np.sum(roi ** 2))) * np.sqrt(np.sum(temp ** 2)) 
            if den == 0: score[dy, dx] = 0
            score[dy, dx] = num / den

    return score


def calculate_correlation_score(prompt, motion_prompt, attn_map, mag, x, cur_step):
    split_prompt = prompt.split(" ")

    for idx, word in enumerate(split_prompt):
        if word == motion_prompt:
            start = idx

    frame_per_one_attention = torch.mean(attn_map[:8], dim=0)
    frame_per_one_attention_np = np.array(frame_per_one_attention.cpu())

    for i in range(1, len(split_prompt)+1):
        image = frame_per_one_attention[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.cpu().numpy().astype(np.uint8)
        image = Image.fromarray(image)
        image.save('/path/save/images/name'+'.png')

    mag_ori_np = mag
    mag = mag.squeeze(-1).squeeze(0)
    mag_np = np.array(mag.cpu())

    score_list = []
    for p_idx in range(1, len(split_prompt)+1):
        correlation_score = cv2.matchTemplate(frame_per_one_attention_np[:,:,p_idx], mag_np, cv2.TM_CCOEFF_NORMED)
        correlation_score_norm = (correlation_score + 1)/2
        score_list.append(correlation_score_norm)

    for i in range(len(score_list)):
        if cur_step > 0:
            attn_map[:8,:,:,i+1:i+2] = (score_list[i].item() * mag_ori_np) * x /cur_step + attn_map[:8,:,:,i+1:i+2]

    return attn_map, start


def calculate_correlation_score_many_method(prompt, motion_prompt, attn_map, mag, x, cur_step):
    split_prompt = prompt.split(" ")

    for idx, word in enumerate(split_prompt):
        if word == motion_prompt:
            start = idx

    frame_per_one_attention = torch.mean(attn_map[:8], dim=0)
    frame_per_one_attention_np = np.array(frame_per_one_attention.cpu())

    # for i in range(1, len(split_prompt)+1):
    #     image = frame_per_one_attention[:, :, i]
    #     image = 255 * image / image.max()
    #     image = image.unsqueeze(-1).expand(*image.shape, 3)
    #     image = image.cpu().numpy().astype(np.uint8)
    #     image = Image.fromarray(image)
    #     image.save('/path/save/images/name'+'.png')

    mag_ori_np = mag
    mag = mag.squeeze(-1).squeeze(0)
    mag_np = np.array(mag.cpu())

    score_list = []
    for p_idx in range(1, len(split_prompt)+1):

        '''
        Choose one of the 6 NCC methods.

        AttentionFlow used 'cv2.TM_CCOEFF_NORMED'.
        '''

        # cv2.TM_SQDIFF
        # correlation_score = cv2.matchTemplate(frame_per_one_attention_np[:,:,p_idx], mag_np, cv2.TM_SQDIFF)
        # correlation_score_norm = 1 - (correlation_score/255)

        # # cv2.TM_SQDIFF_NORMED
        # correlation_score = cv2.matchTemplate(frame_per_one_attention_np[:,:,p_idx], mag_np, cv2.TM_SQDIFF_NORMED)
        # correlation_score_norm = 1 - (correlation_score)

        # # cv2.TM_CCORR
        # correlation_score = cv2.matchTemplate(frame_per_one_attention_np[:,:,p_idx], mag_np, cv2.TM_CCORR)
        # correlation_score_norm = correlation_score/255

        # # cv2.TM_CCORR_NORMED
        # correlation_score = cv2.matchTemplate(frame_per_one_attention_np[:,:,p_idx], mag_np, cv2.TM_CCORR_NORMED)
        # correlation_score_norm = correlation_score
        
        # # cv2.TM_CCOEFF
        # correlation_score = cv2.matchTemplate(frame_per_one_attention_np[:,:,p_idx], mag_np, cv2.TM_CCOEFF)
        # correlation_score_norm = correlation_score

        #cv2.TM_CCOEFF_NORMED
        correlation_score = cv2.matchTemplate(frame_per_one_attention_np[:,:,p_idx], mag_np, cv2.TM_CCOEFF_NORMED)
        correlation_score_norm = (correlation_score + 1)/2

        score_list.append(correlation_score_norm)


    for i in range(len(score_list)):
        if cur_step > 0:
            attn_map[:8,:,:,i+1:i+2] = (score_list[i].item() * mag_ori_np) * x /cur_step + attn_map[:8,:,:,i+1:i+2]

    return attn_map, start









    