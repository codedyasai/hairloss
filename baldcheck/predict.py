import time
import datetime
import os
import copy
import cv2
import random
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from django.conf.global_settings import STATICFILES_DIRS
from torch.optim import lr_scheduler
from torchvision import transforms, datasets
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from PIL import Image
from efficientnet_pytorch import EfficientNet

s_time = time.time()
model_path = './model/'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# # 예비 모델 불러오기
# model1 = EfficientNet.from_pretrained('efficientnet-b3', num_classes=4)
# checkpoint_path = model_path + 'checkpoint.pth'
# checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
# # 모델에 저장된 상태를 불러오기
# model1.load_state_dict(checkpoint['model_state_dict'])

model1 = torch.load(model_path + 'aram_model_gakzil.pt', map_location=torch.device('cpu'))
model2 = torch.load(model_path + 'aram_model_pizi.pt', map_location=torch.device('cpu'))
model3 = torch.load(model_path + 'aram_model_hongban.pt', map_location=torch.device('cpu'))
model4 = torch.load(model_path + 'aram_model_nongpo.pt', map_location=torch.device('cpu'))
model5 = torch.load(model_path + 'aram_model_videm.pt', map_location=torch.device('cpu'))
model6 = torch.load(model_path + 'aram_model_talmo.pt', map_location=torch.device('cpu'))

model1.eval()
model2.eval()
model3.eval()
model4.eval()
model5.eval()
model6.eval()

def preprocess_image_(image_path):
    transform = transforms.Compose([
        transforms.Resize([300, 300]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # 배치 차원 추가
    return image


def predict_image(model, imgpath):
    # 이미지 전처리
    input_image = preprocess_image_(imgpath)

    # 모델에 이미지 전달하여 예측 수행
    with torch.no_grad():
        output = model(input_image)

    # 확률값을 클래스로 변환
    _, predicted_class = torch.max(output, 1)

    return predicted_class.item()

# image_path = '../static/test/01_2.jpg'
    imgpath = './static/images/checkimg'


    # p1 = predict_image(model1, imgpath)
    # p2 = predict_image(model2, imgpath)
    # p3 = predict_image(model3, imgpath)
    # p4 = predict_image(model4, imgpath)
    # p5 = predict_image(model5, imgpath)
    # p6 = predict_image(model6, imgpath)

    print(f'The model1(gakzil) predicted class is: {p1}')
    print(f'The model2(pizi) predicted class is: {p2}')
    print(f'The model3(hongban) predicted class is: {p3}')
    print(f'The model4(nongpo) predicted class is: {p4}')
    print(f'The model5(videm) predicted class is: {p5}')
    print(f'The model6(talmo) predicted class is: {p6}')
    # 1. 미세각질 2. 피지과다 3. 모낭사이홍반 4. 모낭홍반농포 5. 비듬 6. 탈모
    f_class = {1:'양호', 2:'건성', 3:'지성', 4:'지루성', 5:'민감성', 6:'염증성', 7:'비듬성', 8:'탈모성', 9:'복합성'}
    # 1) 양호:
    # 2) 건성: model1미세각질(+)
    # 3) 지성: model2피지과다(+)
    # 5) 민감성: 미세각질(+-), model3모낭사이홍반(+)
    # 4) 지루성: 미세각질(+-), model2피지과다(+), model3모낭사이홍반(+)
    # 6) 염증성: 미세각질(+-), 피지과다(+-), model4모낭홍반/농포(+), 비듬(+-)
    # 7) 비듬성: 미세각질(+-), 피지과다(+-), moedel5비듬(+)
    # 8) 탈모성: model6탈모(+)
    # p_values = [p1, p2, p3, p4, p5, p6]
    # m_c = {0:'정상', 1:'경증', 2:'중등도', 3:'중증'}
    # v, p = 1, 0
    # if not any(p_values):
    #     v, p = 1, 0
    # elif max(p_values) == p1 and all(value < 2 for value in [p2, p3, p4, p5, p6]):
    #     v, p = 2, p1
    # elif max(p_values) == p2 and all(value < 2 for value in [p1, p3, p4, p5, p6]):
    #     v, p = 3, p2
    # elif max(p_values) == p3 and all(value < 2 for value in [p2, p4, p5, p6]) and p1 < p3:
    #     v, p = 5, p3
    # elif (max(p_values) == p2 or max(p_values) == p3) and all(value < 2 for value in [p1, p4, p5, p6]) and not(p2 == p3 == 3):
    #     v, p = 4, max(p2, p3)
    # elif max(p_values) == p4 and all(value < 2 for value in [p3, p6]) and p1 < p4 and p2 < p4 and p5 < p4:
    #     v, p = 6, p4
    # elif max(p_values) == p5 and all(value < 2 for value in [p3, p4, p6]) and p1 < p5 and p2 < p5:
    #     v, p = 7, p5
    # elif max(p_values) == p6 and all(value < 2 for value in [p1, p2, p3, p4, p5]):
    #     v, p = 8, p6
    # else:
    #     v, p = 9, max(p_values)
    #
    # # 결과 출력
    # print(f'당신의 두피는 {f_class[v]}(으)로 {m_c[p]}입니다.')
    # e_time = time.time()
    # print(f'실행시간: {int(round((e_time - s_time) * 1000))}ms')