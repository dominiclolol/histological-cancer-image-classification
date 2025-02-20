import torch.nn as nn
from torchvision import transforms
import torchvision.models as models

transform_TumorClassifierb4 = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize([0.5], [0.5])
])

class TumorClassifierb4(nn.Module):
    def __init__(self, num_classes=2):
        super(TumorClassifierb4, self).__init__()

        self.model = models.efficientnet_b3(pretrained=True)

        for param in self.model.features[:6].parameters():
            param.requires_grad = False

        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)