import logging
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import KernelInceptionDistance
from tqdm import tqdm
import numpy as np

from lib.model.vqvae2 import VQVAE
from lib.utils.factory import get_CATARACTS_data, get_optimizer, get_scheduler
from lib.utils.logging import save_checkpoint, save_examples, plot_grad_flow
from lib.utils.pre_train import parse_args, get_configs, get_log_path, dump_configs


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

m = VQVAE(
    in_channel=eval(data_conf['SHAPE'])[0],
    channel=model_conf['CHANNELS'],
    n_res_block=model_conf['N_RES_BLOCKS'],
    n_res_channel=model_conf['RES_CHANNELS'],
    embed_dim=model_conf['EMBED_DIM'],
    n_embed=model_conf['N_EMBEDDINGS'],
    decay=model_conf['EMA_DECAY']
).to(DEV)
m = torch.nn.DataParallel(m, device_ids=args.device_list) if len(args.device_list) > 1 else m

optim = get_optimizer(m.parameters(), train_conf)
# TODO: Cycle Scheduler
sched = get_scheduler(optim, train_conf)

commitment_cost = float(model_conf['COMMITMENT_COST'])

kid = KernelInceptionDistance(subset_size=train_conf['VAL_SAMPLES'] // 4).to(DEV)

#
# Training loop
#
print("##### Training")

best_val_rec = 1e6
start_epoch, step = 0, 0
for epoch in range(train_conf['EPOCHS']):

    # Training
    m.train()

    recon_error_sum = 0.
    latent_loss_sum = 0.  # Aka codebook loss

    for i, (img, _, _, _, _) in enumerate(tqdm(train_dl)):

        if i == train_conf['STEPS']:
            break

        img = img.to(DEV)

        optim.zero_grad()

        img_recon, latent_loss = m(img)
        recon_error = F.mse_loss(img_recon, img)

        loss = recon_error + commitment_cost * latent_loss
        loss.backward()

        recon_error_sum += recon_error.item()
        latent_loss_sum += latent_loss.item()

        try:
            torch.nn.utils.clip_grad_norm_(
                m.parameters(), train_conf['CLIP_GRAD_NORM']
            )
        except Exception:
            pass

        optim.step()

        if i == 0:
            writer.add_figure(tag='traing/grads', figure=plot_grad_flow(m.named_parameters()),
                              global_step=epoch)

    writer.add_scalar(tag='train/avg_recon_loss', scalar_value=recon_error_sum / i, global_step=epoch)
    writer.add_scalar(tag='train/avg_latent_loss', scalar_value=latent_loss_sum / i, global_step=epoch)
    writer.add_scalar(tag='train/lr', scalar_value=optim.param_groups[0]['lr'], global_step=epoch)
    train_log_str = f"\t{datetime.now().strftime('%H:%M:%S')}" \
                    f"\tEpoch {epoch}|{train_conf['EPOCHS']}" \
                    f"\tAvg. recon loss: {recon_error_sum / i}" \
                    f"\tAvg. latent loss: {latent_loss_sum / i}" \
                    f"\tLR: {optim.param_groups[0]['lr']}"
    print(train_log_str)
    logging.info(msg=train_log_str)

    # Validation
    m.eval()
    if not epoch % train_conf['VAL_FREQ']:

        with torch.no_grad():

            rec_loss_sum = 0.

            for i, (img, _, _, _, _) in enumerate(tqdm(val_dl)):

                if i == train_conf['STEPS']:
                    break

                img = img.to(DEV)

                reconstructions, _ = m(img)
                rec_loss_sum += F.mse_loss(reconstructions, img).item()

                _real_sample = (img + 1.0) * 0.5
                _real_sample = (_real_sample * 255).type(torch.uint8).to(DEV)

                _gen_sample = (reconstructions + 1.0) * 0.5
                _gen_sample = (_gen_sample * 255).type(torch.uint8).to(DEV)

                kid.update(_real_sample, real=True)
                kid.update(_gen_sample, real=False)

            kid_mean, kid_std = kid.compute()
            avg_val_rec_loss = rec_loss_sum / i
            writer.add_scalar(tag='val/Avg. rec loss', scalar_value=avg_val_rec_loss, global_step=epoch)
            writer.add_scalar(tag='val/KID mean', scalar_value=kid_mean, global_step=epoch)
            writer.add_scalar(tag='val/KID std', scalar_value=kid_std, global_step=epoch)
            val_log_str = f"\t{datetime.now().strftime('%H:%M:%S')}" \
                          f"\tAvg. rec loss: {avg_val_rec_loss}" \
                          f"\tKID mean: {kid_mean}" \
                          f"\tKID std: {kid_std}"
            print(val_log_str)
            logging.info(msg=val_log_str)

            if avg_val_rec_loss < best_val_rec:
                best_val_rec = avg_val_rec_loss

                save_examples([reconstructions, img],
                              target_shape=np.array(train_ds.original_shape)//np.array([1, 4, 4]),
                              log_path=LOG_PATH,
                              epoch=epoch,
                              names=['rec', 'real'],
                              device=DEV)

                save_checkpoint([m.state_dict(), optim.state_dict()], LOG_PATH)

    # End of epoch
    sched.step()
