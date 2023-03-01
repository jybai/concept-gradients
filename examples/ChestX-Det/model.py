from utils import *

'''Classification model based on PSPNet 
https://github.com/Deepwise-AILab/ChestX-Det-Dataset/blob/main/pre-trained_PSPNet/ptsemseg/pspnet.py
'''

class PspDethead(nn.Module):

    def __init__(
        self,
        n_classes=13,
        block_config=[3, 4, 23, 3],
        input_size=(473, 473),
    ):

        super(PspDethead, self).__init__()

        self.block_config =  block_config
        self.n_classes = n_classes
        self.input_size = input_size

        # Encoder
        self.convbnrelu1_1 = conv2DBatchNormRelu(
            in_channels=3, k_size=3, n_filters=64, padding=1, stride=2, bias=False
        )
        self.convbnrelu1_2 = conv2DBatchNormRelu(
            in_channels=64, k_size=3, n_filters=64, padding=1, stride=1, bias=False
        )
        self.convbnrelu1_3 = conv2DBatchNormRelu(
            in_channels=64, k_size=3, n_filters=128, padding=1, stride=1, bias=False
        )

        # Vanilla Residual Blocks
        self.res_block2 = residualBlockPSP(self.block_config[0], 128, 64, 256, 1, 1)
        self.res_block3 = residualBlockPSP(self.block_config[1], 256, 128, 512, 2, 1)

        # Dilated Residual Blocks
        self.res_block4 = residualBlockPSP(self.block_config[2], 512, 256, 1024, 1, 2)
        self.res_block5 = residualBlockPSP(self.block_config[3], 1024, 512, 2048, 1, 4)

        # Pyramid Pooling Module
        self.pyramid_pooling = pyramidPooling(2048, [6, 3, 2, 1])

        # Classification Head
        self.cbr_final = conv2DBatchNormRelu(4096, 512, 3, 1, 1, False)
        self.classification = torch.nn.Linear(512, self.n_classes)

    def load_from_pspnet(self, pkl_path):
        rm_modules = ['convbnrelu4_aux', 'aux_cls', 'classification']

        state_dict = torch.load(pkl_path)['model_state']

        for k in list(state_dict.keys()):
            new_k = '.'.join(k.split('.')[1:])
            if all([not new_k.startswith(m) for m in rm_modules]):
                state_dict[new_k] = state_dict[k]
            del state_dict[k]

        self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        
        inp_shape = x.shape[2:]

        # H, W -> H/2, W/2
        x = self.convbnrelu1_1(x)
        x = self.convbnrelu1_2(x)
        x = self.convbnrelu1_3(x)

        # H/2, W/2 -> H/4, W/4
        x = F.max_pool2d(x, 3, 2, 1)

        # H/4, W/4 -> H/8, W/8
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)

        x = self.res_block5(x)

        x = self.pyramid_pooling(x)

        x = self.cbr_final(x)
        
        x = x.mean(dim=(2, 3))
        x = self.classification(x)

        return x

