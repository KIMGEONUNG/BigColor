import sys
import os
sys.path.append(os.path.abspath(os.curdir))
from PIL import Image
from os.path import join, exists
from skimage.color import rgb2lab, lab2rgb
import numpy as np
from train import Colorizer
import torch
import pickle
import argparse
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from tqdm import tqdm
from torch_ema import ExponentialMovingAverage


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=2)
    # 5 --> 32, 4 --> 16, ...
    parser.add_argument('--max_iter', default=1000)
    parser.add_argument('--num_row', type=int, default=8)
    parser.add_argument('--class_index', type=int, default=15)
    parser.add_argument('--size_batch', type=int, default=8)
    parser.add_argument('--num_worker', default=8)
    parser.add_argument('--epoch', type=int, default=12)

    # I/O
    parser.add_argument('--path_config', default='./pretrained/config.pickle')
    parser.add_argument('--path_ckpt_g', default='./pretrained/G_ema_256.pth')
    parser.add_argument('--path_ckpt', default='./ckpts/name')
    parser.add_argument('--path_output', default='./output/directory/name')
    parser.add_argument('--path_input', default='./input/directory/name')
    parser.add_argument('--use_ema', action='store_true')

    parser.add_argument('--num_layer', default=2)
    parser.add_argument('--norm_type', default='instance', 
            choices=['instance', 'batch', 'layer'])
    parser.add_argument('--postfix', default='')

    # Dataset
    parser.add_argument('--dim_z', type=int, default=119)

    # User Input 
    parser.add_argument('--use_rgb', action='store_true')
    parser.add_argument('--max_img', type=int, default=10000)
    parser.add_argument('--classes', type=int, nargs='+', default=[88])
    parser.add_argument('--c_scale', type=float, default=1.)
    parser.add_argument('--c_bias', type=float, default=0.)

    parser.add_argument('--device', default='cuda:0')

    return parser.parse_args()


def set_seed(seed):
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def main(args):
    size_target = 256

    if args.seed >= 0:
        set_seed(args.seed)

    print('Target checkpoint is %s' % args.path_ckpt)
    print('Target Epoch is %03d' % args.epoch)
    print('Target classes is', args.classes)

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

    prep=transforms.Compose([
                transforms.ToTensor(),
                transforms.Grayscale()])
    grays = [join(args.path_input, p) for p in os.listdir(args.path_input)]
    grays = [Image.open(g) for g in grays]
    grays = [prep(g) for g in grays]

    EG = Colorizer(config, args.path_ckpt_g, args_loaded.norm_type,
            id_mid_layer=args.num_layer)
    EG.load_state_dict(torch.load(path_eg, map_location='cpu'), strict=True)
    EG_ema = ExponentialMovingAverage(EG.parameters(), decay=0.99)
    EG_ema.load_state_dict(torch.load(path_eg_ema, map_location='cpu'))

    EG.eval()
    EG.float()
    EG.to(dev)
    
    if args.use_ema:
        print('Use EMA')
        EG_ema.copy_to()

    if not os.path.exists(args.path_output):
        os.mkdir(args.path_output)

    for i, x, in enumerate(tqdm(grays)):
        size = x.shape[1:]
        for c in args.classes: 

            c = torch.LongTensor([c])
            x = x.unsqueeze(0)
            x, c = x.to(dev), c.to(dev)
            z = torch.zeros((1, args.dim_z)).to(dev)
            z.normal_(mean=0, std=0.8)

            c_embd = EG.G.shared(c)
            c_embd = args.c_scale * c_embd + args.c_bias

            x_resize = transforms.Resize((size_target))(x)
            with torch.no_grad():

                output = EG.forward_with_c(x_resize, c_embd, z)
                output = output.add(1).div(2)

            x = x.squeeze(0).cpu()
            output = output.squeeze(0)
            output = output.detach().cpu()
            output = transforms.Resize(size)(output)

            if args.use_rgb:
                pass
            else:
                output = fusion(x, output)
            im = ToPILImage()(output)
            im.save('./%s/%04d_c%04d%s.jpg' %
                    (args.path_output, i, c, args.postfix))
            
        if i  >= args.max_img - 1:
            break;


def fusion(gray, color):
    # Resize
    light = gray.permute(1, 2, 0).numpy() * 100

    color = color.permute(1, 2, 0)
    color = rgb2lab(color)
    ab = color[:, :, 1:]

    lab = np.concatenate((light, ab), axis=-1)
    lab = lab2rgb(lab)
    lab = torch.from_numpy(lab)
    lab = lab.permute(2, 0, 1)
     
    return lab


if __name__ == '__main__':
    args = parse()
    main(args)
