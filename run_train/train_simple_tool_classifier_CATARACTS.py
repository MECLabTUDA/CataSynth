import logging
from datetime import datetime

import timm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import F1Score, AUROC
from tqdm import tqdm

from lib.utils.factory import get_CATARACTS_data, get_optimizer, get_scheduler
from lib.utils.logging import plot_grad_flow, save_checkpoint
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

EPOCHS = train_conf['EPOCHS']
DATA_SHAPE = eval(data_conf['SHAPE'])

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

match model_conf['TYPE'].upper():
    case 'INCEPTIONV4':
        m = timm.create_model('inception_v4',
                              pretrained=True,
                              num_classes=train_ds.dataset.num_tool_classes).to(DEV)
    case 'RESNET18':
        m = timm.create_model('resnet18',
                              pretrained=True,
                              num_classes=train_ds.dataset.num_tool_classes).to(DEV)
    case 'RESNET50':
        m = timm.create_model('resnet50',
                              pretrained=True,
                              num_classes=train_ds.dataset.num_tool_classes).to(DEV)
    case _:
        raise NotImplementedError
m = torch.nn.DataParallel(m, device_ids=args.device_list) if len(args.device_list) > 1 else m
optim = get_optimizer(params=m.parameters(), train_conf=train_conf)
sched = get_scheduler(optim, train_conf)

class_weights = torch.load('../lib/configs/weights_per_class.pt')
tool_loss = nn.BCELoss(weight=class_weights["tool"], reduction='sum').to(DEV)
f1_score = F1Score(num_labels=train_ds.dataset.num_tool_classes, task='binary').to(DEV)
auroc_score = AUROC(num_labels=train_ds.dataset.num_tool_classes, task='binary').to(DEV)

#
# Training loop
#
print("##### Training")

best_val = 1e6
train_losses = []
val_losses = []
for epoch in range(EPOCHS):

    # Training
    m.train()

    loss_sum = 0.
    f1_sum = 0.
    auroc_sum = 0.
    for i, (img, _, file_name, _, tool_label) in enumerate(tqdm(train_dl)):

        if i == train_conf['STEPS']:
            break

        img, tool_label = img.to(DEV), tool_label.to(DEV)

        optim.zero_grad()

        tool_pred = torch.sigmoid(m(img))
        # tool_pred = m(img)

        loss = tool_loss(tool_pred, tool_label.float())
        loss_sum += loss.item()

        tool_f1 = f1_score(tool_pred, tool_label.long())
        f1_sum += tool_f1.item()

        tool_auroc = auroc_score(tool_pred, tool_label.long())
        auroc_sum += tool_auroc.item()

        loss.backward()

        try:
            torch.nn.utils.clip_grad_norm_(m.parameters(), train_conf['CLIP_GRAD_NORM'])
        except Exception:
            pass

        optim.step()

        if i == 0:
            writer.add_figure(tag='traing/model_grads', figure=plot_grad_flow(m.named_parameters()),
                              global_step=epoch)

    avg_loss = loss_sum / i
    train_losses.append(avg_loss)
    avg_train_f1 = f1_sum / i
    avg_train_auroc = auroc_sum / i
    writer.add_scalar(tag='train/avg_loss', scalar_value=avg_loss, global_step=epoch)
    writer.add_scalar(tag='train/F1_score', scalar_value=avg_train_f1, global_step=epoch)
    writer.add_scalar(tag='train/AUROC', scalar_value=avg_train_auroc, global_step=epoch)
    writer.add_scalar(tag='train/lr', scalar_value=optim.param_groups[0]['lr'], global_step=epoch)
    train_log_str = f"\t{datetime.now().strftime('%H:%M:%S')}" \
                    f"\tEpoch {epoch}|{EPOCHS}" \
                    f"\tAvg. loss: {avg_loss}" \
                    f"\tAvg. F1: {avg_train_f1}" \
                    f"\tAvg. AUROC: {avg_train_auroc}" \
                    f"\tLR: {optim.param_groups[0]['lr']}"
    print(train_log_str)
    logging.info(msg=train_log_str)

    # Validation
    if not epoch % train_conf['VAL_FREQ']:

        m.eval()
        with torch.no_grad():

            loss_sum = 0
            f1_sum = 0.
            auroc_sum = 0.

            for i, (img, _, _, _, tool_label) in enumerate(val_dl):
                if i == train_conf['STEPS']:
                    break
                img, tool_label = img.to(DEV), tool_label.to(DEV)

                # tool_pred = m(img)
                tool_pred = torch.sigmoid(m(img))

                loss = tool_loss(tool_pred, tool_label.float())
                loss_sum += loss.item()

                tool_f1 = f1_score(tool_pred, tool_label.long())
                f1_sum += tool_f1.item()

                tool_auroc = auroc_score(tool_pred, tool_label.long())
                auroc_sum += tool_auroc.item()

            avg_loss = loss_sum / i
            val_losses.append(avg_loss)
            avg_f1 = f1_sum / i
            avg_auroc = auroc_sum / i
            writer.add_scalar(tag='val/avg_loss', scalar_value=avg_loss, global_step=epoch)
            writer.add_scalar(tag='val/F1_score', scalar_value=avg_f1, global_step=epoch)
            writer.add_scalar(tag='val/AUROC', scalar_value=avg_auroc, global_step=epoch)
            val_log_str = f"\t{datetime.now().strftime('%H:%M:%S')}" \
                          f"\tAvg val. loss: {avg_loss}" \
                          f"\tAvg. val F1 score: {avg_f1}" \
                          f"\tAvg. val AUROC: {avg_auroc}"
            print(val_log_str)
            logging.info(msg=val_log_str)

            if avg_loss < best_val:
                best_val = avg_loss

                states = [m.state_dict(), optim.state_dict(), sched.state_dict()]
                save_checkpoint(states, LOG_PATH)

    # End of epoch
    sched.step()

    plt.figure()
    plt.plot(np.arange(0, epoch + 1), np.array(train_losses), color='red', label='train')
    plt.plot(np.arange(0, epoch + 1, train_conf['VAL_FREQ']), np.array(val_losses), color='blue', label='val')
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOG_PATH + f"losses.png")
    plt.close()
