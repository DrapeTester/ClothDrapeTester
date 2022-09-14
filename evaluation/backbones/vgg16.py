import torch.nn as nn
import torch
torch.manual_seed(0)
from .base import BaseModel


class VGG(BaseModel):
    def __init__(self,
                 features,
                 input_mean,
                 input_std,
                 output_mean,
                 output_std,
                 num_of_input,
                 num_of_output,
                 dropout,
                 init_weights=True):
        super(VGG, self).__init__("VGG16", input_mean, input_std, output_mean,
                                  output_std, num_of_input, num_of_output)

        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 5, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_of_output),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, inputs):
        # print(f"[input] {inputs[0, 0, 0, 0]}")
        x = self.normalize_input(inputs)
        # print(f"[normalized] {x[0, 0, 0, 0]}")
        x = self.features(x)
        # print(f"[features] {x[0, 0]}")
        x = x.view(x.size(0), -1)
        # print(f"[view] {x[0]}")
        x = self.classifier(x)
        # print(f"[classifier] {x[0]}")
        x = self.unnormalize_output(x)
        # print(f"[unnormalize] {x[0]}")
        x = self.clamp(x)
        # print(f"[clamp] {x[0]}")
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, in_channels=3):
    layers = []

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B':
    [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
        512, 512, 512, 'M'
    ],
    'E': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512,
        512, 'M', 512, 512, 512, 512, 'M'
    ],
}


def make_vgg16(input_mean, input_std, output_mean, output_std, num_of_input,
               num_of_output, dropout, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # if pretrained:
    #     kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], in_channels=num_of_input),
                num_of_input=num_of_input,
                num_of_output=num_of_output,
                input_mean=input_mean,
                input_std=input_std,
                output_mean=output_mean,
                output_std=output_std,
                dropout = dropout,
                **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model
