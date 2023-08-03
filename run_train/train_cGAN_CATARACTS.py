import logging
from datetime import datetime
import gc

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp
import torch.autograd as autograd
from torchmetrics.classification import Accuracy
from torchmetrics.image import KernelInceptionDistance
from tqdm import tqdm
import numpy as np

from lib.model.cgan import Generator, Discriminator, decay_gauss_std
from lib.utils.factory import get_CATARACTS_data, get_optimizer, get_scheduler
from lib.utils.logging import save_checkpoint, save_examples, plot_grad_flow
from lib.utils.pre_train import parse_args, get_configs, get_log_path, dump_configs
from lib.utils.train import weights_init, set_grads

#
# Pre-train
#
print("##### Pre-train")

args = parse_args()

data_conf, model_conf, diffusion_conf, train_conf = get_configs(args)

DEV = args.device_list[0]

LOG_PATH = get_log_path(args)

dump_configs(data_conf, model_conf, diffusion_conf, train_conf, LOG_PATH)

writer = SummaryWriter(log_dir=LOG_PATH)

print(f"Avail. GPUs: ", torch.cuda.device_count())

#
# Data
#
print("##### Loading data.")

train_ds, train_dl, val_ds, val_dl = get_CATARACTS_data(args, data_conf, train_conf)

#
# Model etc.
#
print("##### Loading model etc.")
# TODO: Add support for n_gpus > 1

netG = Generator(
    label_embedding_dim=model_conf['EMBED_DIM'],
    latent_dim=model_conf['LATENT_DIM'],
    base_hidden_dim=model_conf['GEN_BASE_HIDDEN_DIM'],
    dropout=model_conf['GEN_DROPOUT'],
    n_phase_classes=train_ds.num_phases_classes,
    n_tool_dims=train_ds.num_tool_classes
).to(DEV)
netG = torch.nn.DataParallel(netG, device_ids=args.device_list) if len(args.device_list) > 1 else netG
netG.apply(weights_init)

netD = Discriminator(
    img_shape=eval(data_conf['SHAPE']),
    label_embedding_dim=model_conf['EMBED_DIM'],
    base_hidden_dim=model_conf['DISC_BASE_HIDDEN_DIM'],
    noise_std=model_conf['DISC_NOISE_STD'],
    dropout=model_conf['DISC_DROPOUT'],
    n_phase_classes=train_ds.num_phases_classes,
    n_tool_dims=train_ds.num_tool_classes,
).to(DEV)
netD = torch.nn.DataParallel(netD, device_ids=args.device_list) if len(args.device_list) > 1 else netD
netD.apply(weights_init)

match model_conf['ADV_LOSS_FUNC'].upper():
    case 'LS':
        adv_loss = nn.MSELoss(reduction='mean')
    case 'BCE':
        # adv_loss = nn.BCELoss()
        adv_loss = nn.BCEWithLogitsLoss()
    case _:
        raise ValueError('ADV_LOSS_FUNC should be one of LS / BCE')

# Establish convention for real and fake labels during training
real_label = model_conf['REAL_LABEL']  # One-sided label smoothing
fake_label = 0.

optimizerG = get_optimizer(netG.parameters(), train_conf)
schedG = get_scheduler(optimizerG, train_conf)

# train_conf['LR'] = train_conf['LR'] * 0.1  # Less greedy optimization for the discriminator
optimizerD = get_optimizer(netD.parameters(), train_conf)
schedD = get_scheduler(optimizerD, train_conf)

scaler = amp.GradScaler(enabled=train_conf['USE_AMP'])

kid = KernelInceptionDistance(subset_size=train_conf['VAL_SAMPLES'] // 4).to(DEV)
acc = Accuracy(threshold=0.5, task='binary').to(DEV)

# TODO: Warmstart

#
# Training loop
#
print("##### Training")

best_kid = 1e6
start_epoch, step = 0, 0
for epoch in range(train_conf['EPOCHS']):

    # Training
    netD.train()
    netG.train()

    gen_loss_sum = 0.
    disc_real_loss_sum = 0.
    disc_fake_loss_sum = 0.

    for i, (img, _, name, phase_label, tool_label) in enumerate(tqdm(train_dl)):

        if i == train_conf['STEPS']:
            break

        img = img.to(DEV)
        phase_label = phase_label.long().to(DEV)
        tool_label = tool_label.float().to(DEV)

        noise = torch.randn(img.size(0), model_conf['LATENT_DIM'], dtype=img.dtype, device=DEV)
        with amp.autocast(enabled=train_conf['USE_AMP']):

            fake_img = netG(noise, phase_label, tool_label)
            fake_out = netD(fake_img, phase_label, tool_label)
            real_out = netD(img, phase_label, tool_label)

            lossG = adv_loss(fake_out, torch.ones_like(fake_out))
            gen_loss_sum += lossG.item()

            lossD_real = adv_loss(real_out, torch.empty_like(real_out).uniform_(0.9, 1.0))
            disc_real_loss_sum += lossD_real.item()

            lossD_fake = adv_loss(fake_out, torch.empty_like(fake_out).uniform_(0.0, 0.1))
            disc_fake_loss_sum += lossD_fake.item()

            lossD = (lossD_real + lossD_fake)*.5

        scaled_gradsG = autograd.grad(scaler.scale(lossG), netG.parameters(), retain_graph=True)
        scaled_gradsD = autograd.grad(scaler.scale(lossD), netD.parameters())

        set_grads(scaled_gradsG, netG.parameters())
        set_grads(scaled_gradsD, netD.parameters())

        scaler.step(optimizerD)
        if i == 0:
            writer.add_figure(tag='traing/disc_grads', figure=plot_grad_flow(netD.named_parameters()),
                              global_step=epoch)
        optimizerD.zero_grad(set_to_none=True)

        scaler.step(optimizerG)
        if i == 0:
            writer.add_figure(tag='traing/gen_grads', figure=plot_grad_flow(netG.named_parameters()),
                              global_step=epoch)
        optimizerG.zero_grad(set_to_none=True)

        scaler.update()

        if i == 0:
            for id in args.device_list:
                writer.add_text(tag=f'mem_{id}', text_string=torch.cuda.memory_summary(id))

    writer.add_scalar(tag='train/avg_gen_loss', scalar_value=gen_loss_sum / i, global_step=epoch)
    writer.add_scalar(tag='train/avg_disc_real_loss', scalar_value=disc_real_loss_sum / i, global_step=epoch)
    writer.add_scalar(tag='train/avg_disc_fake_loss', scalar_value=disc_fake_loss_sum / i, global_step=epoch)
    writer.add_scalar(tag='train/gen_lr', scalar_value=optimizerG.param_groups[0]['lr'], global_step=epoch)
    writer.add_scalar(tag='train/disc_lr', scalar_value=optimizerD.param_groups[0]['lr'], global_step=epoch)
    train_log_str = f"\t{datetime.now().strftime('%H:%M:%S')}" \
                    f"\tEpoch {epoch}|{train_conf['EPOCHS']}" \
                    f"\tAvg. G loss: {gen_loss_sum / i}" \
                    f"\tAvg. D real loss: {disc_real_loss_sum / i}" \
                    f"\tAvg. D fake loss: {disc_fake_loss_sum / i}" \
                    f"\tLR: {optimizerG.param_groups[0]['lr']}"
    print(train_log_str)
    logging.info(msg=train_log_str)

    # Validation
    netD.eval()
    netG.eval()
    if not epoch % train_conf['VAL_FREQ']:

        with torch.no_grad():

            fake_acc_sum = 0.
            real_acc_sum = 0.

            for i, (img, _, name, phase_label, tool_label) in enumerate(tqdm(val_dl)):

                img, phase_label, tool_label = img.to(DEV), phase_label.long().to(DEV), tool_label.float().to(DEV)

                if i == train_conf['STEPS']:
                    break

                val_noise = torch.randn(train_conf['VAL_SAMPLES'], model_conf['LATENT_DIM'], dtype=torch.float32, device=DEV)

                gen_sample = netG(val_noise, phase_label, tool_label)
                _gen_sample = (gen_sample + 1.0) * 0.5
                _gen_sample = (_gen_sample * 255).type(torch.uint8)

                _real_sample = (img + 1.0) * 0.5
                _real_sample = (_real_sample * 255).type(torch.uint8)

                N = img.shape[0]
                real_target = torch.full((N, 1), 1, dtype=torch.float, device=DEV, requires_grad=False)
                real_pred = netD(img, phase_label, tool_label)
                real_acc_sum += acc(real_pred, real_target).item()

                fake_target = torch.full((N, 1), 0, dtype=torch.float, device=DEV, requires_grad=False)
                fake_pred = netD(gen_sample, phase_label, tool_label)
                fake_acc_sum += acc(fake_pred, fake_target).item()

                kid.update(_real_sample, real=True)
                kid.update(_gen_sample, real=False)

            kid_mean, kid_std = kid.compute()
            writer.add_scalar(tag='val/KID mean', scalar_value=kid_mean, global_step=epoch)
            writer.add_scalar(tag='val/KID std', scalar_value=kid_std, global_step=epoch)
            writer.add_scalar(tag='val/Acc fake', scalar_value=fake_acc_sum / i, global_step=epoch)
            writer.add_scalar(tag='val/Acc real', scalar_value=real_acc_sum / i, global_step=epoch)
            val_log_str = f"\t{datetime.now().strftime('%H:%M:%S')}" \
                          f"\tKID mean: {kid_mean}" \
                          f"\tKID std: {kid_std}" \
                          f"\tAcc fake: {fake_acc_sum / i}" \
                          f"\tAcc real: {real_acc_sum / i}"
            print(val_log_str)
            logging.info(msg=val_log_str)

            if kid_mean < best_kid:
                best_kid = kid_mean

                save_examples([gen_sample, img],
                              target_shape=np.array(train_ds.original_shape)//np.array([1, 4, 4]),
                              log_path=LOG_PATH,
                              epoch=epoch,
                              names=['gen', 'real'],
                              device=DEV)

                save_checkpoint(
                    [netG.state_dict(), netD.state_dict(), optimizerG.state_dict(), optimizerD.state_dict()],
                    LOG_PATH
                )

    # End of epoch
    schedD.step()
    schedG.step()

    decay_gauss_std(netG)
    decay_gauss_std(netD)

    gc.collect()
    torch.cuda.empty_cache()
