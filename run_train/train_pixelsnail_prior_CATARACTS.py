import logging
import time
from datetime import datetime
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lib.dataset.cataracts_dataset import CATARACTSDataset
from lib.utils.factory import get_optimizer, get_scheduler, get_LMBD_data
from lib.utils.logging import save_checkpoint, plot_grad_flow
from lib.utils.pre_train import parse_args, get_configs, get_log_path, dump_configs
from lib.model.pixelsnail_prior import PixelSNAIL

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

CodeRow = namedtuple('CodeRow', ['top', 'bottom', 'phase_label', 'tool_label', 'filename'])
train_ds, train_dl, val_ds, val_dl = get_LMBD_data(data_conf, train_conf)

#
# Model etc.
#
print("##### Loading model etc.")

if model_conf['HIERARCHY'] == 'top':
    model = PixelSNAIL(
        shape=[16, 16],
        n_class=model_conf['N_CLASS'],
        channel=model_conf['CHANNELS'],
        kernel_size=5,
        n_block=model_conf['N_BLOCKS'],
        n_res_block=model_conf['N_BLOCKS'],
        res_channel=model_conf['RES_CHANNELS'],
        dropout=model_conf['DROPOUT'],
        n_out_res_block=model_conf['N_OUT_RES_BLOCKS'],
        n_phase_labels=CATARACTSDataset.num_phases_classes,
        n_tool_labels=CATARACTSDataset.num_tool_classes,
        label_cond_ch=model_conf['LABEL_COND_CH']
    )

elif model_conf['HIERARCHY'] == 'bottom':
    model = PixelSNAIL(
        shape=[32, 32],
        n_class=model_conf['N_CLASS'],
        channel=model_conf['CHANNELS'],
        kernel_size=5,
        n_block=model_conf['N_BLOCKS'],
        n_res_block=model_conf['N_RES_BLOCKS'],
        res_channel=model_conf['RES_CHANNELS'],
        attention=False,
        dropout=model_conf['DROPOUT'],
        n_cond_res_block=model_conf['N_COND_RES_BLOCKS'],
        cond_res_channel=model_conf['RES_CHANNELS'],
        n_phase_labels=CATARACTSDataset.num_phases_classes,
        n_tool_labels=CATARACTSDataset.num_tool_classes,
        label_cond_ch=model_conf['LABEL_COND_CH']
    )
else:
    raise ValueError("'HIERARCHY' must be 'bottom' or 'top'")
m = model.to(DEV)
m = torch.nn.DataParallel(m, device_ids=args.device_list) if len(args.device_list) > 1 else m

optim = get_optimizer(m.parameters(), train_conf)
# TODO: Cycle Scheduler
sched = get_scheduler(optim, train_conf)


#
# Training loop
#
print("##### Training")

best_val = 1e6
for epoch in range(train_conf['EPOCHS']):

    # Training
    m.train()

    loss_sum = 0.
    acc_sum = 0.

    for i, (e_top, e_bottom, phase_label, tool_label, filename) in enumerate(tqdm(train_dl)):

        if i == train_conf['STEPS']:
            break

        e_top = e_top.to(DEV)  # e1
        e_bottom = e_bottom.to(DEV)  # e2
        phase_label = phase_label.to(DEV).long()
        tool_label = tool_label.to(DEV).float()

        optim.zero_grad()

        if model_conf['HIERARCHY'] == 'top':
            target = e_top
            out, _ = m(e_top, phase_label=phase_label, tool_label=tool_label)
        elif model_conf['HIERARCHY'] == 'bottom':
            target = e_bottom
            out, _ = m(e_bottom, condition=e_top, phase_label=phase_label, tool_label=tool_label)
        else:
            raise ValueError("'HIERARCHY' must be 'bottom' or 'top'")

        loss = F.cross_entropy(out, target)

        loss.backward()

        loss_sum += loss.item()

        _, pred = out.max(1)
        correct = (pred == target).float()

        accuracy = correct.sum() / target.numel()
        acc_sum += accuracy.item()

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

    writer.add_scalar(tag='train/avg_loss', scalar_value=loss_sum / i, global_step=epoch)
    writer.add_scalar(tag='train/avg_acc', scalar_value=acc_sum / i, global_step=epoch)
    writer.add_scalar(tag='train/lr', scalar_value=optim.param_groups[0]['lr'], global_step=epoch)
    train_log_str = f"\t{datetime.now().strftime('%H:%M:%S')}" \
                    f"\tEpoch {epoch}|{train_conf['EPOCHS']}" \
                    f"\tAvg. loss: {loss_sum / i}" \
                    f"\tAvg. acc: {acc_sum / i}" \
                    f"\tLR: {optim.param_groups[0]['lr']}"
    print(train_log_str)
    logging.info(msg=train_log_str)

    # Validation
    m.eval()
    if not epoch % train_conf['VAL_FREQ']:

        with torch.no_grad():

            loss_sum = 0.
            acc_sum = 0.

            for i, (e_top, e_bottom, phase_label, tool_label, filename) in enumerate(tqdm(val_dl)):

                if i == train_conf['STEPS']:
                    break

                e_top = e_top.to(DEV)
                e_bottom = e_bottom.to(DEV)
                phase_label = phase_label.to(DEV).long()
                tool_label = tool_label.to(DEV).float()

                if model_conf['HIERARCHY'] == 'top':
                    target = e_top
                    out, _ = m(e_top, phase_label=phase_label, tool_label=tool_label)
                elif model_conf['HIERARCHY'] == 'bottom':
                    target = e_bottom
                    out, _ = m(e_bottom, condition=e_top, phase_label=phase_label, tool_label=tool_label)
                else:
                    raise ValueError("'HIERARCHY' must be 'bottom' or 'top'")

                loss = F.cross_entropy(out, target)

                loss_sum += loss.item()

                _, pred = out.max(1)
                correct = (pred == target).float()
                accuracy = correct.sum() / target.numel()
                acc_sum += accuracy.item()

            writer.add_scalar(tag='val/Avg. loss', scalar_value=loss_sum / i, global_step=epoch)
            writer.add_scalar(tag='val/Avg. acc.', scalar_value=acc_sum / i, global_step=epoch)
            val_log_str = f"\t{datetime.now().strftime('%H:%M:%S')}" \
                          f"\tAvg. loss: {loss_sum / i}" \
                          f"\tAvg. acc.: {acc_sum / i}"
            print(val_log_str)
            logging.info(msg=val_log_str)

            if (loss_sum / i) < best_val:
                best_val = (loss_sum / i)

                save_checkpoint([m.state_dict(), optim.state_dict()], LOG_PATH)

    # End of epoch
    sched.step()
