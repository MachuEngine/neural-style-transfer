import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import copy

# GPU가 사용 가능하면 GPU를 사용하고, 아니면 CPU 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 이미지 로드 및 전처리 함수 (이미지 경로와 원하는 크기를 입력)====================
def load_image(image_path, imsize=512):
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")  # RGB로 변환하여 3채널 유지
    image = loader(image).unsqueeze(0)  # 배치 차원 추가
    return image.to(device, torch.float)
"""
torch.float는 torch.float32와 동일하며, 텐서를 float32 타입으로 변환합니다.
ToTensor() 변환을 거친 이미지 텐서는 기본적으로 torch.float32 타입을 갖지만, 일부 연산 과정에서 uint8, float16, double 등의 타입으로 변형될 수도 있습니다.
PyTorch 모델은 보통 float32(32비트 부동소수점) 연산을 수행하므로, 이를 맞춰주기 위해 to(torch.float)을 명시적으로 적용하는 것이 일반적입니다.
"""

# 예시: 콘텐츠와 스타일 이미지 로드
content_img = load_image("./data/city-water-sky-Germany.com.jpg")
style_img = load_image("./data/Danny-Ivan-abstract-pattern-texture.jpg")

# 이미지 크기 확인 - 보통 RGB 3채널로 이루어져있지만, 투명도가 추가된 4채널인 경우도 있음 
print(f"Content Image Size: {content_img.size()}")
print(f"Style Image Size: {style_img.size()}")


# 두 이미지의 크기가 같아야 합니다.
# assert 문은 조건이 True가 아니면 AssertionError를 발생시킵니다.
# assert 문을 사용하여 두 이미지의 크기가 동일한지 확인합니다.
# error가 발생하면 "콘텐츠와 스타일 이미지의 크기가 동일해야 합니다."라는 메시지를 출력합니다.
assert content_img.size() == style_img.size(), "콘텐츠와 스타일 이미지의 크기가 동일해야 합니다."

# ======================================================================
"""
그람 행렬은 같은 계층 내 피처맵들 간의 상관관계를 내적 계산한 헹렬
피처들 간의 유사도를 보는 것은 스타일 정보를 확인하는 것임
스타일은 피처맵의 상관관계를 보는 것이므로, 그람 행렬을 사용하여 스타일 손실을 계산




스타일 손실은 "생성된 이미지의 스타일"과 "스타일 이미지의 스타일"을 비교하여 계산하는 값





"""

def gram_matrix(input_tensor):
    # input_tensor: [batch_size, feature_maps, height, width]
    batch_size, n_feature_maps, h, w = input_tensor.size()
    features = input_tensor.view(n_feature_maps, h * w)
    G = torch.mm(features, features.t())
    return G.div(n_feature_maps * h * w)

# ======================================================================
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # target은 미리 계산해두고 기울기 전파하지 않도록 detach
        self.target = target.detach()
        self.loss = 0

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = 0

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

""" 
models.vgg19(pretrained=True).features.to(device).eval()는 사전 학습된 
VGG19 모델을 가져와서 특징 추출기(features) 부분만 사용하도록 설정하는 것.
즉, fully connected layer(FCL)를 제거한 채 컨볼루션 레이어만 남겨서 이미지의 
스타일과 콘텐츠 특징을 추출하는 용도.
"""
# 사전 학습된 VGG19 모델을 불러오고, 평가 모드로 설정 ========================
cnn = models.vgg19(pretrained=True, progress=True).features.to(device).eval()

# 정규화에 사용될 평균값과 표준편차 (ImageNet 기준)
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# 정규화 모듈 정의
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # 채널별로 맞춰주기 위해 [C x 1 x 1] 형태로 변환
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std

# 사용할 층 이름 지정
content_layers = ['conv_4']  # 콘텐츠는 보통 conv4 이후의 층 사용
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# 모델에 콘텐츠와 스타일 손실 모듈 삽입
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers, style_layers=style_layers):
    cnn = copy.deepcopy(cnn)  # 원본 모델을 보존하기 위해 복제
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    
    content_losses = []
    style_losses = []
    
    model = nn.Sequential(normalization)
    i = 0  # conv 층 카운터
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            continue
        
        model.add_module(name, layer)
        
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)
        
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)
    
    # 모델 구조에서 손실 모듈 이후의 층은 사용하지 않아도 됩니다.
    for j in range(len(model) - 1, -1, -1):
        if isinstance(model[j], ContentLoss) or isinstance(model[j], StyleLoss):
            break
    model = model[:j+1]
    
    return model, style_losses, content_losses

model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                           cnn_normalization_mean,
                                           cnn_normalization_std,
                                           style_img, content_img)



# 초기 입력 이미지를 콘텐츠 이미지로 설정 (복사본) ===========================
input_img = content_img.clone()

# 이미지를 출력하는 헬퍼 함수
def imshow(tensor, title=None):
    image = tensor.cpu().clone().detach()
    image = image.squeeze(0)
    unloader = transforms.ToPILImage()
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # 화면 갱신

# 최적화를 위한 파라미터 설정
optimizer = optim.LBFGS([input_img.requires_grad_()])

num_steps = 300
style_weight = 1e6
content_weight = 1

print("최적화 시작...")
run = [0]
while run[0] <= num_steps:
    def closure():
        # 입력 이미지를 0~1 사이로 클리핑
        input_img.data.clamp_(0, 1)
        optimizer.zero_grad()
        model(input_img)
        
        style_score = 0
        content_score = 0
        
        for sl in style_losses:
            style_score += sl.loss
        for cl in content_losses:
            content_score += cl.loss
        
        loss = style_weight * style_score + content_weight * content_score
        loss.backward()
        
        run[0] += 1
        if run[0] % 50 == 0:
            print(f"스텝 {run[0]}: 스타일 손실: {style_score.item():.4f}, 콘텐츠 손실: {content_score.item():.4f}")
        return loss

    optimizer.step(closure)

# 최종 이미지 클리핑
input_img.data.clamp_(0, 1)
imshow(input_img, title='최종 생성 이미지')
plt.show()
