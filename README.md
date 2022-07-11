## BigColor: Colorization using a Generative Color Prior for Natural Images<br><sub>Official PyTorch Implementation of the ECCV 2022 Paper</sub>

![Teaser image 1](./srcs/teaser_1.png)

**BigColor: Colorization using a Generative Color Prior for Natural Images**<br>
Geonung Kim, Kyoungkook Kang, Seongtae Kim, Hwayoon Lee,Sehoon Kim, Jonghyun Kim, Seung-Hwan Baek, Sunghyun Cho<br>

[\[Paper\]](https://github.com/KIMGEONUNG/BigColor)
[\[Supple\]](https://github.com/KIMGEONUNG/BigColor)
[\[Project Page\]](https://github.com/KIMGEONUNG/BigColor)

Abstract: *For realistic and vivid colorization, generative priors have recently been exploited. However, such generative priors often fail for in-the-wild complex images due to their limited representation space. In this paper, we propose BigColor, a novel colorization approach that provides vivid colorization for diverse in-the-wild images with complex structures. While previous generative priors are trained to synthesize both image structures and colors, we learn a generative color prior to focus on color synthesis given the spatial structure of an image. In this way, we reduce the burden of synthesizing image structures from the generative prior and expand its representation space to cover diverse images. To this end, we propose a BigGAN-inspired encoder-generator network that uses a spatial feature map instead of a spatially-flattened BigGAN latent code, resulting in an enlarged representation space. Our method enables robust colorization for diverse inputs in a single forward pass, supports arbitrary input resolutions, and provides multi-modal colorization results. We demonstrate that BigColor significantly outperforms existing methods especially on in-the-wild images with complex structures.*

### Requirements

```
conda env create -f environment.yml
```


### Pretrained Model

```
pip install gdown
pip install --upgrade gdown
```

```
./download-bigcolor.sh
```

```
./download-pretrained.sh
```

### Colorization
![Teaser image 2](./srcs/teaser_2.png)

#### ImageNet1K Validation 

inference results available


```
./scripts/infer.bigcolor.e011.sh
```

#### Real Gray Colorization

```
./scripts/colorize.real.sh
```

#### Multi-modal Solutions

```
./scripts/colorize.multi_c.sh
```

```
./scripts/colorize.multi_z.sh
```

### Citation

```
@inproceedings{Kim2022Bigcolor,
  title     = {BigColor: Colorization using a Generative Color Prior for Natural Images},
  author    = {Geonung Kim, Kyoungkook Kang, Seongtae Kim, Hwayoon Lee,Sehoon Kim, Jonghyun Kim, Seung-Hwan Baek, Sunghyun Cho},
  booktitle = {Proc. ECCV},
  year      = {2022}
}

```
