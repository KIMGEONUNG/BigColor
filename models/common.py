import torch
import torch.nn as nn
import torchvision.transforms as transforms
from .encoders import (EncoderF_Res, EncoderF32_Res, EncoderF64_Res,
                       EncoderZ_Res, EncoderF8_Res)
from .biggan import Generator

class VGG16Perceptual(nn.Module):
    '''
    Dimension for each layers
    layer 00 [64 , 224, 224]
    layer 01 [64 , 224, 224]
    layer 02 [64 , 224, 224]
    layer 03 [64 , 224, 224]
    layer 04 [64 , 112, 112]
    layer 05 [128, 112, 112]
    layer 06 [128, 112, 112]
    layer 07 [128, 112, 112]
    layer 08 [128, 112, 112]
    layer 09 [128, 56 , 56 ]
    layer 10 [256, 56 , 56 ]
    layer 11 [256, 56 , 56 ]
    layer 12 [256, 56 , 56 ]
    layer 13 [256, 56 , 56 ]
    layer 14 [256, 56 , 56 ]
    layer 15 [256, 56 , 56 ]
    layer 16 [256, 28 , 28 ]
    layer 17 [512, 28 , 28 ]
    layer 18 [512, 28 , 28 ]
    layer 19 [512, 28 , 28 ]
    layer 20 [512, 28 , 28 ]
    layer 21 [512, 28 , 28 ]
    layer 22 [512, 28 , 28 ]
    layer 23 [512, 14 , 14 ]
    layer 24 [512, 14 , 14 ]
    layer 25 [512, 14 , 14 ]
    layer 26 [512, 14 , 14 ]
    layer 27 [512, 14 , 14 ]
    layer 28 [512, 14 , 14 ]
    layer 29 [512, 14 , 14 ]
    layer 30 [512, 7  , 7  ]
    '''

    def __init__(self, path_vgg: str,
            id_targets = [1, 2, 13, 20],
            resize=True,
            normalized_input=True):
        super().__init__()

        import pickle 
        with open(path_vgg, 'rb') as f:
            self.model = pickle.load(f).eval()

        self.normalized_intput = normalized_input
        self.idx_targets = id_targets

        preprocess = []
        if resize:
            preprocess.append(transforms.Resize(256))
            preprocess.append(transforms.CenterCrop(224))
        if normalized_input:
            preprocess.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]))

        self.preprocess = transforms.Compose(preprocess)

    def get_mid_feats(self, x):
        x = self.preprocess(x)
        feats = []
        for i, layer in enumerate(self.model.features[:max(self.idx_targets) + 1]):
            x = layer(x)
            if i in self.idx_targets:
                feats.append(x)

        return feats


    def forward(self, x1, x2):
        size_batch = x1.shape[0]
        x1_feats = self.get_mid_feats(x1)
        x2_feats = self.get_mid_feats(x2)

        loss = 0
        for feat1, feat2 in zip(x1_feats, x2_feats):
            loss += feat1.sub(feat2).pow(2).mean()

        return loss / size_batch / len(self.idx_targets)


class Colorizer(nn.Module):
    def __init__(self, 
                 config, 
                 path_ckpt_g, 
                 norm_type,
                 activation='relu',
                 id_mid_layer=2, 
                 fix_g=False,
                 load_g=True,
                 init_e=None,
                 use_attention=False,
                 use_res=True,
                 dim_f=16):
        super().__init__()

        self.id_mid_layer = id_mid_layer  
        self.use_attention = use_attention
        self.use_res = use_res

        if not use_res:
            print('Warning: without residual path')

        if dim_f == 64:
            self.E = EncoderF64_Res(norm=norm_type,
                                  activation=activation,
                                  init=init_e,
                                  use_att=use_attention)
            self.id_mid_layer = 4  
        elif dim_f == 32:
            self.E = EncoderF32_Res(norm=norm_type,
                                  activation=activation,
                                  init=init_e,
                                  use_att=use_attention)
            self.id_mid_layer = 3  
        elif dim_f == 16:
            self.E = EncoderF_Res(norm=norm_type,
                                  activation=activation,
                                  init=init_e,
                                  use_res=use_res,
                                  use_att=use_attention)
            self.id_mid_layer = 2  
        elif dim_f == 8:
            self.E = EncoderF8_Res(norm=norm_type,
                                  activation=activation,
                                  init=init_e,
                                  use_att=use_attention)
            self.id_mid_layer = 1 
        elif dim_f == 1:
            self.E = EncoderZ_Res(norm=norm_type,
                                  activation=activation,
                                  init=init_e,
                                  use_att=use_attention)
            self.id_mid_layer = 0
        else:
            raise Exception('In valid dim_f')

        self.G = Generator(**config)
        if load_g:
            print('Use pretraind G')
            self.G.load_state_dict(torch.load(path_ckpt_g, map_location='cpu'),
                                   strict=False)
        self.fix_g = fix_g
        if fix_g:
            for p in self.G.parameters():
                p.requires_grad = False

    def forward(self, x_gray, c, z):
        f = self.E(x_gray, self.G.shared(c)) 
        output = self.G.forward_from(z, self.G.shared(c), 
                self.id_mid_layer, f)
        return output

    def forward_with_c(self, x_gray, c_embd, z):
        f = self.E(x_gray, c_embd) 
        output = self.G.forward_from(z, c_embd, 
                self.id_mid_layer, f)
        return output

    def forward_with_c2(self, x_gray, c_embds, z):
        f = self.E(x_gray, c_embds[0]) 
        output = self.G.forward_from(z, c_embds[1], 
                self.id_mid_layer, f)
        return output

    def forward_with_cp(self, x_gray, c_embds, z):
        c_embds_E = c_embds[:5]
        c_embds_G = c_embds[5:]
        f = self.E.forward_with_cp(x_gray, c_embds_E) 
        output = self.G.forward_from_with_cp(z, c_embds_G, 
                                             self.id_mid_layer, f)
        return output

    def train(self, mode=True):
        if self.fix_g:
            self.E.train(mode)
        else:
            super().train(mode)
