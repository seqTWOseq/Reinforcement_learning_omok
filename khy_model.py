import torch.nn as nn
import torch.nn.functional as F

class OmokCNN(nn.Module):
    def __init__(self):
        super(OmokCNN, self).__init__()
        
        # Conv 층마다 BatchNorm을 추가하여 학습 속도와 안정성 극대화
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 128채널 * 15 * 15 = 28,800
        self.fc1 = nn.Linear(128 * 15 * 15, 512)
        self.fc2 = nn.Linear(512, 225) 

    def forward(self, x):
        # Conv -> BatchNorm -> ReLU 순서로 통과
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 2차원 데이터를 1차원으로 넓게 펴기
        x = x.view(x.size(0), -1) 
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # Q-value 출력 (활성화 함수 없음)
        
        return x