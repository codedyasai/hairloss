import json
from urllib import request

import pandas as pd
import torch
from django.conf.global_settings import STATICFILES_DIRS
from django.http import HttpResponse
from django.shortcuts import render
from efficientnet_pytorch import EfficientNet

from baldcheck.predicttest import predict_image, preprocess_image
from result.bs import table
from bald.models import Human


# Create your views here.
def index(request):
    return render(request, 'index.html')




def mapmarker(request):
    list = pd.read_excel('static/list.xlsx')
    positions = [] # 좌표 데이터 초기화
    for i in range(len(list)):
        lat = list['좌표(Y)'].iloc[i]
        lng = list['좌표(X)'].iloc[i]
        name = list['요양기관명'].iloc[i]
        add = list['주소'].iloc[i]
        positions.append({"lat": lat, "lng": lng, "name": name, "add": add})
    # result = {"positions": positions[:100]}
    result = {"positions": positions}
    return HttpResponse(json.dumps(result), content_type='application/json')


def upform(request):
    gender = request.POST['gender']
    age = request.POST['age']
    imgname = ''

    model_path = './model/'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

    if 'img' in request.FILES:
        uploadimg = request.FILES['img']
        # imgname = uploadimg._name
        fp = open('%s%s' % ('./static/images/', 'checkimg'), 'wb')
        for temp in uploadimg.chunks():
            fp.write(temp)
        fp.close()

    imgpath = './static/images/checkimg'


    p1 = predict_image(model1, imgpath)
    p2 = predict_image(model2, imgpath)
    p3 = predict_image(model3, imgpath)
    p4 = predict_image(model4, imgpath)
    p5 = predict_image(model5, imgpath)
    p6 = predict_image(model6, imgpath)

    print(f'The model1(gakzil) predicted class is: {p1}')
    print(f'The model2(pizi) predicted class is: {p2}')
    print(f'The model3(hongban) predicted class is: {p3}')
    print(f'The model4(nongpo) predicted class is: {p4}')
    print(f'The model5(videm) predicted class is: {p5}')
    print(f'The model6(talmo) predicted class is: {p6}')
    # 1. 미세각질 2. 피지과다 3. 모낭사이홍반 4. 모낭홍반농포 5. 비듬 6. 탈모
    f_class = {1:'정상', 2:'건성', 3:'지성', 4:'지루성', 5:'민감성', 6:'염증성', 7:'비듬성', 8:'탈모성', 9:'복합성'}

    search = {1:'탈모예방', 2:'건성두피', 3:'지성두피', 4:'지루성두피', 5:'민감성두피', 6:'두피 염증', 7:'두피 비듬', 8:'탈모 두피', 9:'두피케어'}
    # 1) 양호:
    # 2) 건성: model1미세각질(+)
    # 3) 지성: model2피지과다(+)
    # 5) 민감성: 미세각질(+-), model3모낭사이홍반(+)
    # 4) 지루성: 미세각질(+-), model2피지과다(+), model3모낭사이홍반(+)
    # 6) 염증성: 미세각질(+-), 피지과다(+-), model4모낭홍반/농포(+), 비듬(+-)
    # 7) 비듬성: 미세각질(+-), 피지과다(+-), moedel5비듬(+)
    # 8) 탈모성: model6탈모(+)
    p_values = [p1, p2, p3, p4, p5, p6]
    m_c = {0:'정상', 1:'경증', 2:'중등도', 3:'중증'}

    v, p = 1, 0
    if not any(p_values):
        v, p = 1, 0
    elif max(p_values) == p1 and all(value < 2 and value < p1 for value in [p2, p3, p4, p5, p6]):
        v, p = 2, p1
    elif max(p_values) == p2 and all(value < 2 and value < p2 for value in [p1, p3, p4, p5, p6]):
        v, p = 3, p2
    elif max(p_values) == p3 and all(value < 2 and value < p3 for value in [p2, p4, p5, p6]) and p1 < p3:
        v, p = 5, p3
    elif (max(p_values) == p2 or max(p_values) == p3) and all(
            value < 2 and value < max(p2, p3) for value in [p1, p4, p5, p6]):
        v, p = 4, max(p2, p3)
    elif max(p_values) == p4 and all(value < 2 for value in [p3, p6]) and all(
            value < p4 for value in [p1, p2, p3, p5, p6]):
        v, p = 6, p4
    elif max(p_values) == p5 and all(value < 2 for value in [p3, p4, p6]) and all(
            value < p5 for value in [p1, p2, p3, p4, p6]):
        v, p = 7, p5
    elif max(p_values) == p6 and all(value < 2 for value in [p1, p2, p3, p4, p5]) and all(
            value < p6 for value in [p1, p2, p3, p4, p5]):
        v, p = 8, p6
    else:
        v, p = 9, max(p_values)

    # 결과 출력
    print(f'당신의 두피는 {f_class[v]}(으)로 {m_c[p]}입니다.')

    r1 = m_c[p1]
    r2 = m_c[p2]
    r3 = m_c[p3]
    r4 = m_c[p4]
    r5 = m_c[p5]
    r6 = m_c[p6]

    human = Human(sex=gender, age=age, pred=m_c[p], skin=f_class[v])
    human.save()

    result = table()

    value1 = p1  # 이 값을 1, 2, 3, 4, 5, 6 중 하나로 설정
    value2 = p2
    value3 = p3
    value4 = p4
    value5 = p5
    value6 = p6

    # 퍼센티지 계산
    percentage = value1 * 25
    percentage2 = value2 * 25
    percentage3 = value3 * 25
    percentage4 = value4 * 25
    percentage5 = value5 * 25
    percentage6 = value6 * 25

    percentageset = [percentage , percentage2 , percentage3 , percentage4, percentage5, percentage6]

    predtype = ['미세각질', '피지과다', '모낭사이홍반', '모낭홍반농포', '비듬', '탈모']

    colorset = ['#F94144', '#F3722C', '#F9C74F', '#90BE6D', '#43AA8B', '#577590']

    resultzip = zip(percentageset, predtype, colorset)

    ctx = {
        'predresult': m_c[p],
        'headskin': f_class[v],
        'skintype': search[v],
        'imgpath': imgpath,
        'result': result,
        'gender': gender,
        'age': age,
        'resultzip': resultzip,
        'p1': p1,
        'p2': p2,
        'p3': p3,
        'p4': p4,
        'p5': p5,
        'p6': p6,
        'r1': r1,
        'r2': r2,
        'r3': r3,
        'r4': r4,
        'r5': r5,
        'r6': r6,
    }
    return render(request, 'result.html', ctx)

