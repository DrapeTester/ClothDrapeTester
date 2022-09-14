import torch.nn as nn
import torchvision
import torch
torch.manual_seed(0)
from backbones.base import BaseModel


def Conv1p(in_planes, places, stride=2):
    '''
    conv2d
    bn2d
    relu
    maxpool2d
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,
                  out_channels=places,
                  kernel_size=7,
                  stride=stride,
                  padding=3,
                  bias=False), nn.BatchNorm2d(places), nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


def Conv1(in_planes, places, stride=2):
    '''
    conv2d,
    bn2d
    relu
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,
                  out_channels=places,
                  kernel_size=7,
                  stride=stride,
                  padding=3,
                  bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
    )


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    #inplanes其实就是channel,叫法不同
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #把shortcut那的channel的维度统一
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    '''
    bottleneck module in resnet
    conv2d - bn2d - relu
    conv2d - bn2d - relu
    conv2d - bn2d
    '''
    def __init__(self,
                 in_places,
                 places,
                 stride=1,
                 downsampling=False,
                 expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,
                      out_channels=places,
                      kernel_size=1,
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places,
                      out_channels=places,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places,
                      out_channels=places * self.expansion,
                      kernel_size=1,
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )

        if self.downsampling:
            '''
            covn2d
            bn2d
            '''
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places,
                          out_channels=places * self.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(places * self.expansion))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        '''
            relu( 
                x + bottleneck(x) 
                )
        '''
        residual = x
        out = self.bottleneck(x)
        if self.downsampling:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNetBasicblock(nn.Module):
    def __init__(self, inplanes, layers, num_classes=1000, expansion=1):
        input_size = inplanes
        self.inplanes = 64
        block = BasicBlock
        super(ResNetBasicblock, self).__init__()
        self.conv1 = nn.Conv2d(
            input_size,
            64,
            kernel_size=7,
            stride=2,
            padding=3,  #因为mnist为（1，28，28）灰度图，因此输入通道数为1
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # kaiming init for all conv2ds
        # constant for bn2d
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        #downsample 主要用来处理H(x)=F(x)+x中F(x)和xchannel维度不匹配问题
        downsample = None
        #self.inplanes为上个box_block的输出channel,planes为当前box_block块的输入channel
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class ResNetBottleneck(nn.Module):
    def __init__(self, inplanes, blocks, num_classes=1000, expansion=4):
        super(ResNetBottleneck, self).__init__()
        self.expansion = expansion
        # conv bn relu, input 3, output 64
        # 1/4
        self.conv1 = Conv1(in_planes=inplanes, places=64)

        # bottleneck 1, input 64, output 64 * expansion = 156
        # image 1
        self.layer1 = self.make_layer(in_places=64,
                                      places=64,
                                      block=blocks[0],
                                      stride=1)

        # bottleneck 2, input 256, output 128 * expansion = 512
        # image 1/4 per bottlenck
        self.layer2 = self.make_layer(in_places=256,
                                      places=128,
                                      block=blocks[1],
                                      stride=2)

        # bottleneck 3, input 512, OUTPUT 256 * 4 = 1024
        # image 1/4 per bottlenck
        self.layer3 = self.make_layer(in_places=512,
                                      places=256,
                                      block=blocks[2],
                                      stride=2)

        # bottleneck 4, input 1024, output 512 * 4 = 2048
        # image 1/4 per bottlenck
        self.layer4 = self.make_layer(in_places=1024,
                                      places=512,
                                      block=blocks[3],
                                      stride=2)

        # kaiming init for all conv2ds
        # constant for bn2d
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, downsampling=True))
        for i in range(1, block):
            layers.append(Bottleneck(places * self.expansion, places))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def ResNet50(inplanes=4):
    return ResNetBottleneck(inplanes, [3, 4, 6, 3])


def ResNet18(inplanes=4):
    return ResNetBasicblock(inplanes, [2, 2, 2, 2])


class ResNet18Model(BaseModel):
    def __init__(
        self,
        input_mean,
        input_std,
        output_mean,
        output_std,
        num_of_input,
        num_of_output,
        version="18",
    ):
        super(ResNet18Model,
              self).__init__("resnet18", input_mean, input_std, output_mean,
                             output_std, num_of_input, num_of_output)

        if version == "50":
            base_class = ResNet50
            torch_base_mode = torchvision.models.resnet50(pretrained=False)
            self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        elif version == "18":
            base_class = ResNet18
            torch_base_mode = torchvision.models.resnet18(pretrained=False)
            # self.avgpool = nn.AdaptiveAvgPool2d((4, 2))
        else:
            raise ValueError(version)

        self.base = base_class(inplanes=self.num_of_input)
        self.relu = nn.ReLU(inplace=True)

        # the decoder part

        self.fc1 = nn.Linear(24576, 64)
        self.fc2 = nn.Linear(64, self.num_of_output)
        self._initialize_weights()

        # load the pretrain weight
        # if load_pretrained_weights is True:
        #     # get the pretrained weight
        #     mod = torch_base_mode
        #     self_model_keys = list(self.base.state_dict().keys())
        #     pretrained_model_values = list(mod.state_dict().values())
        #     model_dict_1 = {}
        #     for k in range(len(self_model_keys)):
        #         self_model_key = self_model_keys[k]
        #         if self_model_key == "conv1.0.weight" or self_model_key == "conv1.weight":
        #             print(f"set {self_model_key} with an expanded weight")

        #             # 1. get the value of current conv1 weight, deep copy
        #             cur_weight = self.state_dict()[f"base.{self_model_key}"]
        #             # for i in self.parameters():

        #             # 2. get the value of the desired conv1 weight
        #             desired_weight = pretrained_model_values[k]
        #             # 3. assert 3, assert 4
        #             assert cur_weight.shape[1] == 4 and desired_weight.shape[
        #                 1] == 3
        #             # print(f"old weight {cur_weight[:, 0:3, :, :]}")
        #             cur_weight[:, 0:3, :, :] = desired_weight
        #             # print(f"new weight {cur_weight[:, 0:3, :, :]}")
        #             # 4. cur_weight.segment(0,3) = desired conv1 weight
        #             # 5. put it into the dic1
        #             print(
        #                 f"set the first 3 channels for weight {self_model_key} succ"
        #             )
        #             dic1 = {self_model_key: cur_weight}
        #         else:
        #             # print(f"self_model_key is {self_model_key}, k = {k}")
        #             dic1 = {self_model_key: pretrained_model_values[k]}
        #         model_dict_1.update(dic1)
        #     self.base.load_state_dict(model_dict_1)

    def forward(self, x):
        x = self.normalize_input(x)
        x = self.base(x)
        # print(f"before avg pool {x.shape}")
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # print(f"after avg pool {x.shape}")
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.unnormalize_output(x)
        x = self.clamp(x)
        # clip in eval mode

        # if x.shape[1] == 3:
        #     # print(f"raw {x}")
        #     x = torch.clip(x, min=1, max=100)
        #     # print(f"after {x} ")
        #     # exit()
        # elif x.shape[1] == 6:
        #     # the first three channels [1, 100]
        #     # the last thrre channels [-100, 100]
        #     # set upper bound and lower bound

        #     # set x
        #     x = torch.max(torch.min(x, self.upper_bound), self.lower_bound)

        # else:
        #     raise ValueError("please change")
        return x

    def _initialize_weights(self):
        '''
        conv2d: weight std 0.01
        bn2d: weight std 0.02
        bn1d: weight std 0.02
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1, 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, 1, 0.02)
                nn.init.constant_(m.bias, 0)

    def _freeze_backbone_except_1st_conv(self):
        get_the_first_layer = False
        for name, param in self.named_parameters():
            if name[:4] == "base":
                # is backbone!
                if name == "base.conv1.0.weight" or name == "base.conv1.weight":
                    get_the_first_layer = True
                    # print(
                    #     f"{name} {param.shape}, requred_grad {param.requires_grad}, can be changed"
                    # )
                else:
                    param.requires_grad = False
                    # print(
                    #     f"{name} {param.shape}, requred_grad {param.requires_grad}, freezed"
                    # )

        assert get_the_first_layer == True, f"we didnt' find the first conv"

    def _freeze_backbone(self):
        get_the_first_layer = False
        for name, param in self.named_parameters():
            if name[:4] == "base":
                param.requires_grad = False


def make_resnet18(input_mean, input_std, output_mean, output_std, num_of_input,
                  num_of_output):

    return ResNet18Model(input_mean, input_std, output_mean, output_std,
                         num_of_input, num_of_output)
