from typing import List
import numpy as np
import torch
import pickle
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, ToPILImage, Grayscale, Resize, Compose
from tqdm import tqdm
from torch_ema import ExponentialMovingAverage
import skimage.color
from PIL import Image
import timm
from math import ceil
import tempfile
from cog import BasePredictor, Path, Input, BaseModel

from train import Colorizer
from utils.common_utils import set_seed, rgb2lab, lab2rgb


MODEL2SIZE = {"resnet50d": 224, "tf_efficientnet_l2_ns_475": 475}


class ModelOutput(BaseModel):
    image: Path


class Predictor(BasePredictor):
    def setup(self):
        path_eg = "ckpts/bigcolor/EG_011.ckpt"
        path_eg_ema = "ckpts/bigcolor/EG_EMA_011.ckpt"
        path_args = "ckpts/bigcolor/args.pkl"
        path_config = "pretrained/config.pickle"
        path_ckpt_g = "pretrained/G_ema_256.pth"
        self.dev = "cuda:0"
        use_ema = True
        cls_model = "tf_efficientnet_l2_ns_475"

        with open(path_config, "rb") as f:
            config = pickle.load(f)
        with open(path_args, "rb") as f:
            self.args_loaded = pickle.load(f)

        # Load Colorizer
        self.EG = Colorizer(
            config,
            path_ckpt_g,
            self.args_loaded.norm_type,
            id_mid_layer=self.args_loaded.num_layer,
            activation=self.args_loaded.activation,
            use_attention=self.args_loaded.use_attention,
            dim_f=16,
        )

        self.EG.load_state_dict(torch.load(path_eg, map_location="cpu"), strict=True)
        EG_ema = ExponentialMovingAverage(self.EG.parameters(), decay=0.99)
        EG_ema.load_state_dict(torch.load(path_eg_ema, map_location="cpu"))

        self.EG.eval()
        self.EG.float()
        self.EG.to(self.dev)

        if use_ema:
            print("Use EMA")
            EG_ema.copy_to()

        # Load Classifier
        self.classifier = timm.create_model(
            cls_model, pretrained=True, num_classes=1000
        ).to(self.dev)
        self.classifier.eval()
        self.size_cls = MODEL2SIZE[cls_model]

        self.prep = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])

    def predict(
        self,
        image: Path = Input(
            description="Input image.",
        ),
        mode: str = Input(
            default="Real Gray Colorization",
            choices=[
                "Real Gray Colorization",
                "Multi-modal class vector c",
            ],
            description="Choose the colorization mode.",
        ),
        classes: str = Input(
            default="88",
            description="Specify classes for Multi-modal class vector c, separate the classes with space.",
        ),
    ) -> List[ModelOutput]:

        # some defualt settings for the simplicity of the demo
        topk = 5
        seed = -1
        no_upsample = False
        use_rgb = False
        num_power = 4
        type_resize = "powerof"
        size_target = 256
        dim_z = 119
        c_bias = 0.0
        c_scale = 1.0

        model_output = []

        if mode == "Real Gray Colorization":

            def resizer(x):
                length_long = max(x.shape[-2:])
                length_sort = min(x.shape[-2:])
                unit = ceil(
                    (length_long * (size_target / length_sort)) / (2**num_power)
                )
                long = unit * (2**num_power)

                if x.shape[-1] > x.shape[-2]:
                    fn = Resize((size_target, long))
                else:
                    fn = Resize((long, size_target))

                return fn(x)

            im = Image.open(str(image))
            x = ToTensor()(im)
            if x.shape[0] != 1:
                x = Grayscale()(x)

            size = x.shape[1:]

            x = x.unsqueeze(0)
            x = x.to(self.dev)
            z = torch.zeros((1, self.args_loaded.dim_z)).to(self.dev)
            z.normal_(mean=0, std=0.8)

            # Classification
            x_cls = x.repeat(1, 3, 1, 1)
            x_cls = Resize((self.size_cls, self.size_cls))(x_cls)
            c = self.classifier(x_cls)
            cs = torch.topk(c, topk)[1].reshape(-1)
            c = torch.LongTensor([cs[0]]).to(self.dev)

            for c in cs:
                c = torch.LongTensor([c]).to(self.dev)
                x_resize = resizer(x)

                with torch.no_grad():
                    output = self.EG(x_resize, c, z)
                    output = output.add(1).div(2)

                if no_upsample:
                    size_output = x_resize.shape[-2:]
                    x_rs = x_resize.squeeze(0).cpu()
                else:
                    size_output = size
                    x_rs = x.squeeze(0).cpu()

                output = transforms.Resize(size_output)(output)
                output = output.squeeze(0)
                output = output.detach().cpu()

                if use_rgb:
                    x_img = output
                else:
                    x_img = fusion(x_rs, output)
                im = ToPILImage()(x_img)

                out_path = Path(tempfile.mkdtemp()) / "output.png"
                im.save(str(out_path))
                model_output.append(ModelOutput(image=out_path))

        else:

            classes = [int(a) for a in classes.split()]

            x = self.prep(Image.open(str(image)))
            size = x.shape[1:]

            for i, c in enumerate(classes):

                c = torch.LongTensor([c])
                x = x.unsqueeze(0)
                x, c = x.to(self.dev), c.to(self.dev)
                z = torch.zeros((1, dim_z)).to(self.dev)
                z.normal_(mean=0, std=0.8)

                c_embd = self.EG.G.shared(c)
                c_embd = c_scale * c_embd + c_bias

                x_resize = transforms.Resize((size_target))(x)
                with torch.no_grad():

                    output = self.EG.forward_with_c(x_resize, c_embd, z)
                    output = output.add(1).div(2)

                x = x.squeeze(0).cpu()

                output = output.squeeze(0)
                output = output.detach().cpu()
                output = transforms.Resize(size)(output)

                if use_rgb:
                    pass
                else:
                    output = fusion_mulit_c(x, output)

                im = ToPILImage()(output)

                out_path = Path(tempfile.mkdtemp()) / f"output_{i}.png"
                im.save(str(out_path))
                model_output.append(ModelOutput(image=out_path))

        return model_output


def fusion(img_gray, img_rgb):
    img_gray *= 100
    ab = rgb2lab(img_rgb)[..., 1:, :, :]
    lab = torch.cat([img_gray, ab], dim=0)
    rgb = lab2rgb(lab)
    return rgb


def fusion_mulit_c(gray, color):
    # Resize
    light = gray.permute(1, 2, 0).numpy() * 100

    color = color.permute(1, 2, 0)
    color = skimage.color.rgb2lab(color)
    ab = color[:, :, 1:]

    lab = np.concatenate((light, ab), axis=-1)
    lab = skimage.color.lab2rgb(lab)
    lab = torch.from_numpy(lab)
    lab = lab.permute(2, 0, 1)

    return lab
