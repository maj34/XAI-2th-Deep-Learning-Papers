import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Generator의 Bottleneck 부분에 사용되는 REsidual Block 클래스
class ResidualBlock(nn.Module):     # nn.Module 클래스 상속
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()   # 상위 클래스인 nn.Module 초기화
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))
            # 논문에서는 Convolution, Normalizaton, ReLU만 나와있으나 실제 코드에선 Convolution과 Normalization이 추가
            # 여러 개의 layer를 순차적으로 거치기 때문에 Sequantial() 함수로 묶어준 후, self.main에 할당
            # dim_in/dim_out : 입/출력 dimension (논문에 따르면 둘 다 256)

    def forward(self, x):   # 여러 동작을 진행하면서 처음의 정보를 잃어갈 수 있기 때문에 처음 정보를 더해주는 것
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):     
        # 첫번째 convolution layer의 output dimension인 conv_dim, domain label의 수인 c_dim, Residual Block의 수인 repeat_num을 매개변수에 넘겨줌
        super(Generator, self).__init__()

        layers = []     # layers 리스트에 모든 레이어들을 append. (layer가 너무 많아 리스트에 넣어준 후 Sequential() 함수에 리스트를 넘겨줄 것임)
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False)) # 3+c_dim인 이유 forward에서 설명
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False)) # dimension 커짐
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
            # 맨 처음 정의한 ResidualBlock 인스턴스를 repeat_num 수만큼 만들어 layers 리스트에 추가
            # 논문에 따르면 6개의 Residual Block 추가

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False)) # dimension 작아짐
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers) # layers 리스트를 Sequential() 함수에 전달

    def forward(self, x, c):       # x : real image, c : target domain  -> solve.py에서 들어옴
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1)      # c의 size = [16, 7, 1, 1]
        c = c.repeat(1, 1, x.size(2), x.size(3))    # c의 size = [16, 7, 128, 128]
        x = torch.cat([x, c], dim=1)    # x의 size가 [16, 3, 128, 128], c의 size가 [16, 7, 128, 128]이므로 붙이면 [16, 10, 128, 128]이 됨
        return self.main(x)     # Generator의 레이어들의 집합에 x를 입력으로 주는 것
        # foward() 함수에 들어온 초기 x의 dimension은 3이지만, dimension이 7인 c와 torch.cat() 함수로 이어붙였기 때문에 3+c_dim이 된것
        # self.main(x)에서 반환된 값(합성 이미지)은 초기 x(원본이미지)와 동일한 size를 가짐 ([16, 3, 128, 128])
        # 논문에 나와있는 Generator의 마지막 layer의 shape과 동일

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        # Input Layer
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)) # 입력으로 RGB 이미지가 들어오기 때문에 input dimension=3
        layers.append(nn.LeakyReLU(0.01))

        # Hidden layer
        curr_dim = conv_dim
        for i in range(1, repeat_num): # repeat_num = 6이므로 5번(1~5) 반복
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)) 
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2 # dimension이 점점 2배가 됨

        # Output Layer
        kernel_size = int(image_size / np.power(2, repeat_num))     # kernel_size를 (이미지 한 변의 길이/2^repeat_num)로 할당, 이 값은 두번째 Conv layer의 kernel size로 들어감 (논문에 h/64로 표기)
        self.main = nn.Sequential(*layers)      # Discriminator에서 나왔던 layer들을 Sequential() 함수에 넣고 self.main에 할당
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)     # 논문에서 Output Layer(D_src) 부분 : real인지 fake인지 여부만 출력해야 하므로 output dimension = 1
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)       # 논문에서 Output Layer(D_cls) 부분 : domain label을 출력해야 하므로 output dimension = c_dim
        
    def forward(self, x):   # x에 진짜인지 가짜인지 판별할 이미지가 전달
        h = self.main(x)    # Hidden layer까지 모두 거치고 나면 그 값이 h에 할당
        out_src = self.conv1(h)     # D_src값 반환하여 out_src에 저장
        out_cls = self.conv2(h)     # D_cls값 반환하여 out_cls에 저장
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))      # out_cls의 크기를 조정하여 out_src와 함께 반환
