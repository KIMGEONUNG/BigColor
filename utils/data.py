from PIL import Image, ImageEnhance
import torch
from torchvision.transforms import ToPILImage,ToTensor


class ColorEnhance(object):
    def __init__(self, factor=1.5):
        self.factor=factor

    def __call__(self, x):
        if isinstance(x, Image.Image):
            return self._enhance_PIL(x)
        elif isinstance(x, torch.Tensor):
            if len(x.shape) == 4:
                xs = [self._enhance_Tensor(each)[None, ...] for each in x]
                x = torch.cat(xs)
                return x

            elif len(x.shape) == 3:
                return self._enhance_Tensor(x)
            else:
                raise Exception('Invalid tensor shape')

        return x 

    def _enhance_PIL(self, x: Image.Image):
        x = ImageEnhance.Color(x)
        x = x.enhance(self.factor)
        return x

    def _enhance_Tensor(self, x: torch.Tensor):
        x = ToPILImage()(x)
        x = self._enhance_PIL(x)
        x = ToTensor()(x)
        return x
