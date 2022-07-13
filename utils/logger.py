import torch
from os.path import join
from .common_utils import lab_fusion, make_grid_multi
from torch.cuda.amp import autocast


def make_log_ckpt(EG, D,
                  optim_g, optim_d,
                  schedule_g, schedule_d, 
                  ema_g, 
                  num_iter, args, epoch, path_ckpts):
    # Encoder&Generator
    name = 'EG_%03d.ckpt' % epoch 
    path = join(path_ckpts, name) 
    torch.save(EG.state_dict(), path) 

    # Discriminator
    name = 'D_%03d.ckpt' % epoch 
    path = join(path_ckpts, name) 
    torch.save(D.state_dict(), path) 

    # EMA Encoder&Generator
    name = 'EG_EMA_%03d.ckpt' % epoch 
    path = join(path_ckpts, name) 
    torch.save(ema_g.state_dict(), path) 

    # Oters
    name = 'OTHER_%03d.ckpt' % epoch 
    path = join(path_ckpts, name) 
    torch.save({'optim_g': optim_g.state_dict(),
                'optim_d': optim_d.state_dict(),
                'schedule_g': schedule_g.state_dict(),
                'schedule_d': schedule_d.state_dict(),
                'num_iter': num_iter}, path)


def load_for_retrain(EG, D,
                     optim_g, optim_d, schedule_g, schedule_d, 
                     epoch, path_ckpts, dev):
    # Encoder&Generator
    name = 'EG_%03d.ckpt' % epoch 
    path = join(path_ckpts, name) 
    state = torch.load(path, map_location=dev)
    EG.load_state_dict(state)

    # Discriminator
    name = 'D_%03d.ckpt' % epoch 
    path = join(path_ckpts, name) 
    state = torch.load(path, map_location=dev)
    D.load_state_dict(state)

    # Oters
    name = 'OTHER_%03d.ckpt' % epoch 
    path = join(path_ckpts, name) 
    state = torch.load(path, map_location=dev)
    optim_g.load_state_dict(state['optim_g'])
    optim_d.load_state_dict(state['optim_d'])
    schedule_g.load_state_dict(state['schedule_g'])
    schedule_d.load_state_dict(state['schedule_d'])

    return state['num_iter']

def load_for_retrain_EMA(ema_g, epoch, path_ckpts, dev):
    # Encoder&Generator
    name = 'EG_EMA_%03d.ckpt' % epoch 
    path = join(path_ckpts, name) 
    state = torch.load(path, map_location=dev)
    ema_g.load_state_dict(state)


def make_log_scalar(writer, num_iter, loss_dic: dict):
    loss_g = loss_dic['loss_g']
    loss_d = loss_dic['loss_d']
    writer.add_scalars('GAN loss', 
        {'G': loss_g.item(), 'D': loss_d.item()}, num_iter)

    del loss_dic['loss_g']
    del loss_dic['loss_d']
    for key, value in loss_dic.items():
        writer.add_scalar(key, value.item(), num_iter)


def make_log_img(EG, dim_z, writer, args, sample, dev, num_iter, name,
        ema=None):
    outputs_rgb = []
    outputs_fusion = []
    
    EG.eval()
    for id_sample in range(len(sample['xs'])):
        z = torch.zeros((args.size_batch, dim_z))
        z.normal_(mean=0, std=0.8)
        x_gt = sample['xs'][id_sample]

        x = sample['xs_gray'][id_sample]
        c = sample['cs'][id_sample]
        z, x, c = z.to(dev), x.to(dev), c.to(dev)

        with torch.no_grad():
            with autocast():
                if ema is None:
                    output = EG(x, c, z)
                else:
                    with ema.average_parameters():
                        output = EG(x, c, z)
            output = output.add(1).div(2).detach().cpu()
        output_fusion = lab_fusion(x_gt, output)
        outputs_rgb.append(output)
        outputs_fusion.append(output_fusion)

    grid = make_grid_multi(outputs_rgb, nrow=4)
    writer.add_image('recon_%s_rgb' % name, 
            grid, num_iter)

    grid = make_grid_multi(outputs_fusion, nrow=4)
    writer.add_image('recon_%s_fusion' % name, 
            grid, num_iter)

    writer.flush()
