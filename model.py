import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from transformers import AutoConfig, AutoModel, ViTForImageClassification, ViTFeatureExtractor


class BaseModel(nn.Module):
    """
    기본적인 컨볼루션 신경망 모델
    """

    def __init__(self, num_classes):
        """
        모델의 레이어 초기화

        Args:
            num_classes (int): 출력 레이어의 뉴런 수
        """
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 입력 이미지 텐서

        Returns:
            x (torch.Tensor): num_classes 크기의 출력 텐서
        """
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x
    
# Custom Model Template
class Resnet34CategoryModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet_model = models.resnet34(pretrained=True)
        num_ftrs = resnet_model.fc.in_features
        self.softmax = nn.Softmax()
        self.resnet = nn.Sequential(*list(resnet_model.children())[:-1])
        self.mask_linear = nn.Linear(num_ftrs, 3)
        self.gender_linear = nn.Linear(num_ftrs, 2)
        self.age_linear = nn.Linear(num_ftrs, 3)
    def forward(self, x):
        x = self.resnet(x)
        x = x.squeeze(3).squeeze(2)
        mask_prediction = self.mask_linear(x)
        gender_prediction = self.gender_linear(x)
        age_prediction = self.age_linear(x)
        return mask_prediction, gender_prediction, age_prediction

class custom_resnet34(nn.Module):
    #input size: 224,224
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = models.resnet34(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
    
    def forward(self,x):
        x = self.resnet(x)
        return x

class custom_resnet50(nn.Module):
    #input size: 224,224
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
    
    def forward(self,x):
        x = self.resnet(x)
        return x

class sh_vit_model(nn.Module):
    #input size: 
    def __init__(self, num_classes):
        super().__init__()
        self.vitmodel = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        x = self.vitmodel(x)
        return x
    
class EfficientNet_b0(nn.Module): # input size 224 224
    def __init__(self, num_classes):
        super(EfficientNet_b0, self).__init__()

        # Load EfficientNet model and its configuration
        config = AutoConfig.from_pretrained('google/EfficientNet-b0')
        self.eff_net = AutoModel.from_pretrained('google/EfficientNet-b0', config=config)

        # Custom layers after EfficientNet
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive pooling to get a fixed size output
        self.dropout = nn.Dropout(0.1)  # Adding dropout for regularization
        self.fc = nn.Linear(config.hidden_dim, num_classes)  # Custom fully connected layer

    def forward(self, x):
        # Pass input through EfficientNet
        eff_net_output = self.eff_net(x)
        
        # Extract the output of EfficientNet
        x = eff_net_output.last_hidden_state
        x = F.relu(x)  # Adding ReLU activation
        
        # Additional custom layers
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = self.dropout(x)  # Applying dropout
        x = self.fc(x)

        return x
    
class EfficientNet_b1(nn.Module): # input size 240 240
    def __init__(self, num_classes):
        super(EfficientNet_b1, self).__init__()

        # Load EfficientNet model and its configuration
        config = AutoConfig.from_pretrained('google/EfficientNet-b1')
        self.eff_net = AutoModel.from_pretrained('google/EfficientNet-b1', config=config)

        # Custom layers after EfficientNet
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive pooling to get a fixed size output
        self.dropout = nn.Dropout(0.1)  # Adding dropout for regularization
        self.fc = nn.Linear(config.hidden_dim, num_classes)  # Custom fully connected layer

    def forward(self, x):
        # Pass input through EfficientNet
        eff_net_output = self.eff_net(x)
        
        # Extract the output of EfficientNet
        x = eff_net_output.last_hidden_state
        x = F.relu(x)  # Adding ReLU activation
        
        # Additional custom layers
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = self.dropout(x)  # Applying dropout
        x = self.fc(x)

        return x

class EfficientNet_b2(nn.Module): # input size 260 260
    def __init__(self, num_classes):
        super(EfficientNet_b2, self).__init__()

        # Load EfficientNet model and its configuration
        config = AutoConfig.from_pretrained('google/EfficientNet-b2')
        self.eff_net = AutoModel.from_pretrained('google/EfficientNet-b2', config=config)

        # Custom layers after EfficientNet
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive pooling to get a fixed size output
        self.dropout = nn.Dropout(0.1)  # Adding dropout for regularization
        self.fc = nn.Linear(self.eff_net.config.hidden_dim, num_classes)  # Custom fully connected layer

    def forward(self, x):
        # Pass input through EfficientNet
        eff_net_output = self.eff_net(x)
        
        # Extract the output of EfficientNet
        x = eff_net_output.last_hidden_state
        x = F.relu(x)  # Adding ReLU activation
        
        # Additional custom layers
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = self.dropout(x)  # Applying dropout
        x = self.fc(x)

        return x

class torchvision_VIT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.vit_b_32()
        in_features = 768
        self.model.heads[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class age_VIT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit_model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
        #self.transforms = ViTFeatureExtractor.from_pretrained('nateraw/vit-age-classifier')

        # Modify the last layer of the pre-trained model to match the number of classes
        self.vit_model.classifier = nn.Linear(self.vit_model.config.hidden_size, num_classes)

    def forward(self, x):
        # Forward pass through the ViT model
        #inputs = self.transforms(x, return_tensors='pt')
        outputs = self.vit_model(x)

        # Extract logits and apply linear layer
        logits = outputs.logits
        return logits
    
class pytorch_VIT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = ViT('B_16_imagenet1k', pretrained=True, num_classes=num_classes, image_size=384)

    def forward(self, x):
        return self.model(x)
