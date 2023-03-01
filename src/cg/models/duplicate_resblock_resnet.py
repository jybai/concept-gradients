from typing import Any, Callable, List, Optional, Type, Union
from copy import deepcopy

import torch
import torch.nn as nn
from torch import Tensor

from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck, model_urls, conv1x1
from torchvision.models.utils import load_state_dict_from_url

class DuplicateResblockResNet(ResNet):
    def __init__(
        self,
        duplicate_layer: int,
        duplicate_block: int,
        duplicate_copies: int,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        
        assert (duplicate_layer > 0) and (duplicate_layer < len(layers) + 1)
        assert (duplicate_block > 0) and (duplicate_block < layers[duplicate_layer - 1])
        assert duplicate_copies > 1 # at least two copies (including the original)
        
        nn.Module.__init__(self)
        
        self.duplicate_layer = duplicate_layer
        self.duplicate_block = duplicate_block
        self.duplicate_copies = duplicate_copies
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        duplicate_kwargs = {'duplicate_block': self.duplicate_block, 'duplicate_copies': self.duplicate_copies}
        self.layer1 = self._make_layer(block, 64, layers[0], 
                                       **(duplicate_kwargs if self.duplicate_layer == 1 else {}))
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], 
                                       **(duplicate_kwargs if self.duplicate_layer == 2 else {}))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], 
                                       **(duplicate_kwargs if self.duplicate_layer == 3 else {}))
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], 
                                       **(duplicate_kwargs if self.duplicate_layer == 4 else {}))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
    
    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        duplicate_block: Optional[int] = None,
        duplicate_copies: Optional[int] = None,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if (duplicate_block is not None) and (i == duplicate_block):
                duplicate_blocks = [
                    block(
                        self.inplanes,
                        planes,
                        groups=self.groups,
                        base_width=self.base_width,
                        dilation=self.dilation,
                        norm_layer=norm_layer,
                    )
                    for _ in range(duplicate_copies)
                ]
                block_ = nn.Sequential(*duplicate_blocks)
            else:
                block_ = block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            layers.append(block_)

        return nn.Sequential(*layers)
    
    def load_state_dict_from_nonduplicate(self, weights):
        prefix = f'layer{self.duplicate_layer}.{self.duplicate_block}.'

        for k in list(weights.keys()):
            if k.startswith(prefix):
                for i in range(self.duplicate_copies):
                    new_k = f'{prefix}{i}.{k[len(prefix):]}'
                    weights[new_k] = deepcopy(weights[k])
                del weights[k]

        self.load_state_dict(weights, strict=True)

def _duplicate_resblock_resnet(
    arch: str,
    duplicate_layer: int,
    duplicate_block: int,
    duplicate_copies: int,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> DuplicateResblockResNet:
    model = DuplicateResblockResNet(duplicate_layer, duplicate_block, duplicate_copies, 
                                    block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict_from_nonduplicate(state_dict)
    return model

def duplicate_resblock_resnet50(duplicate_layer: int, duplicate_block: int, duplicate_copies: int,
                                pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DuplicateResblockResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _duplicate_resblock_resnet(
        'resnet50', duplicate_layer, duplicate_block, duplicate_copies,
        Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)
        
'''
def _duplicate_resblock_resnet(
    duplicate_layer: int,
    duplicate_block: int,
    duplicate_copies: int,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights,
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = DuplicateResblockResNet(duplicate_layer, duplicate_block, duplicate_copies, 
                                    block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict_from_nonduplicate(weights.get_state_dict(progress=progress))

    return model

def duplicate_resblock_resnet50(duplicate_layer: int, duplicate_block: int, duplicate_copies: int,
                                weights: Optional[ResNet50_Weights] = None, progress: bool = True, 
                                **kwargs: Any) -> DuplicateResblockResNet:
    weights = ResNet50_Weights.verify(weights)

    return _duplicate_resblock_resnet(duplicate_layer, duplicate_block, duplicate_copies, 
                                      Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)
'''
