import os
from os.path import join, exists
import numpy as np
from train import Colorizer
import torch
import pickle
import argparse
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage, Resize, Compose
from tqdm import tqdm
from torch_ema import ExponentialMovingAverage
from utils.common_utils import set_seed, rgb2lab, lab2rgb
from math import ceil


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=2)

    # I/O
    parser.add_argument('--path_config', default='./pretrained/config.pickle')
    parser.add_argument('--path_ckpt_g', default='./pretrained/G_ema_256.pth')
    parser.add_argument('--path_ckpt', default='./ckpts/baseline_1000')
    parser.add_argument('--path_output', default='./results')
    parser.add_argument('--path_imgnet_val', default='./imgnet/val')

    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--no_upsample', action='store_true')
    parser.add_argument('--use_rgb', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--dim_f', type=int, default=16)

    parser.add_argument('--type_resize', type=str, default='powerof',
            choices=['absolute', 'original', 'square', 'patch', 'powerof'])
    parser.add_argument('--num_power', type=int, default=4)
    parser.add_argument('--size_target', type=int, default=256)
    parser.add_argument('--iter_max', type=int, default=50000)

    return parser.parse_args()


def main(args):
    size_target = 256

    if args.seed >= 0:
        set_seed(args.seed)

    print('Target Epoch is %03d' % args.epoch)

    path_eg = join(args.path_ckpt, 'EG_%03d.ckpt' % args.epoch)
    path_eg_ema = join(args.path_ckpt, 'EG_EMA_%03d.ckpt' % args.epoch)
    path_args = join(args.path_ckpt, 'args.pkl')

    if not exists(path_eg):
        raise FileNotFoundError(path_eg)
    if not exists(path_args):
        raise FileNotFoundError(path_args)

    # Load Configuratuion
    with open(args.path_config, 'rb') as f:
        config = pickle.load(f)
    with open(path_args, 'rb') as f:
        args_loaded = pickle.load(f)

    dev = args.device

    grays = ImageFolder(args.path_imgnet_val,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Grayscale()]))

    EG = Colorizer(config, 
                   args.path_ckpt_g,
                   args_loaded.norm_type,
                   id_mid_layer=args_loaded.num_layer,
                   activation=args_loaded.activation, 
                   use_attention=args_loaded.use_attention,
                   # use_res=not args_loaded.no_res,
                   dim_f=args.dim_f)
    EG.load_state_dict(torch.load(path_eg, map_location='cpu'), strict=True)
    EG_ema = ExponentialMovingAverage(EG.parameters(), decay=0.99)
    EG_ema.load_state_dict(torch.load(path_eg_ema, map_location='cpu'))

    EG.eval()
    EG.float()
    EG.to(dev)

    resizer = None
    if args.type_resize == 'absolute':
        resizer = Resize((args.size_target))
    elif args.type_resize == 'original':
        resizer = Compose([])
    elif args.type_resize == 'square':
        resizer = Resize((args.size_target, args.size_target))
    elif args.type_resize == 'powerof':
        assert args.size_target % (2 ** args.num_power) == 0

        def resizer(x):
            length_long = max(x.shape[-2:])
            length_sort = min(x.shape[-2:])
            unit = ceil((length_long * (args.size_target / length_sort)) 
                                        / (2 ** args.num_power))
            long = unit * (2 ** args.num_power)

            if x.shape[-1] > x.shape[-2]:
                fn = Resize((args.size_target, long))
            else:
                fn = Resize((long, args.size_target))

            return fn(x)
    elif args.type_resize == 'patch':
        resizer = Resize((args.size_target))
    else:
        raise Exception('Invalid resize type')
    
    if args.use_ema:
        print('Use EMA')
        EG_ema.copy_to()

    if not os.path.exists(args.path_output):
        os.mkdir(args.path_output)

    for i, (x, c) in enumerate(tqdm(grays)):
        if i >= args.iter_max:
            break

        size_original = x.shape[1:]

        c = torch.LongTensor([c])
        x = x.unsqueeze(0)
        x, c = x.to(dev), c.to(dev)
        z = torch.zeros((1, args_loaded.dim_z)).to(dev)
        z.normal_(mean=0, std=0.8)
        x_down = resizer(x)

        with torch.no_grad():
            output = EG(x_down, c, z)
            output = output.add(1).div(2)

        x = x.squeeze(0).cpu()
        x_down = x_down.squeeze(0).cpu()
        output = output.squeeze(0)
        output = output.detach().cpu()

        if args.no_upsample:
            output = Resize(x_down.shape[-2:])(output)
            lab_fusion = fusion(x_down, output)
        else:
            output = Resize(size_original)(output)
            lab_fusion = fusion(x, output)

        if args.use_rgb:
            im = ToPILImage()(output)
        else:
            im = ToPILImage()(lab_fusion)
        im.save('%s/%05d.jpg' % (args.path_output, i))


def fusion(img_gray, img_rgb):
    img_gray *= 100
    ab = rgb2lab(img_rgb)[..., 1:, :, :]
    lab = torch.cat([img_gray, ab], dim=0)
    rgb = lab2rgb(lab)
    return rgb 


if __name__ == '__main__':
    args = parse()
    main(args)
