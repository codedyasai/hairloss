import torch
from PIL import Image
from efficientnet_pytorch import EfficientNet
from torchvision import transforms

model_path = './model/checkpoint(비듬).pth'

model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=4)

checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

model.load_state_dict(checkpoint['model_state_dict'])

model.eval()


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize([300, 300]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # 배치 차원 추가
    return image


def predict_image(model, image_path):
    # 이미지 전처리
    input_image = preprocess_image(image_path)

    # 모델에 이미지 전달하여 예측 수행
    with torch.no_grad():
        output = model(input_image)

    # 확률값을 클래스로 변환
    _, predicted_class = torch.max(output, 1)

    return predicted_class.item()



if __name__ == '__main__':
    image_path = '../static/test/hb_3.jpg'
    predicted_class = predict_image(model, image_path)

    print(f'The predicted class is: {predicted_class}')

# 0 양호
# 1 경증
# 2 중증도
# 3 중증