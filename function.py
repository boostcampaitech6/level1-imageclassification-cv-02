import os, json
import wandb
import random
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    
def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(
        figsize=(12, 18 + 2)
    )  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(
        top=0.8
    )  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n**0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join(
            [
                f"{task} - gt: {gt_label}, pred: {pred_label}"
                for gt_label, pred_label, task in zip(
                    gt_decoded_labels, pred_decoded_labels, tasks
                )
            ]
        )
        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)
    return figure

def training(model, train_set, val_set, criterion, optimizer, scheduler, dataset, dataset_module, device, train_logger, run, save_dir, args):
    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    train_logger.info(f"train_set : {len(train_set)}")
    train_logger.info(f"val_set : {len(val_set)}")
    
    num_workers = multiprocessing.cpu_count()//2
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    if args.category_train:
        best_val_acc = 0
        best_val_loss = np.inf
        for epoch in range(args.epochs):
            # train loop
            model.train()
            mask_loss_value = 0
            gender_loss_value = 0
            age_loss_value = 0
            mask_matches = 0
            gender_matches = 0
            age_matches = 0
            for idx, train_batch in enumerate(train_loader):
                inputs, mask_labels, gender_labels, age_labels = train_batch
                inputs = inputs.to(device)
                mask_labels, gender_labels, age_labels = mask_labels.to(device), gender_labels.to(device), age_labels.to(device)

                optimizer.zero_grad()

                mask_prediction, gender_prediction, age_prediction = model(inputs)
                
                mask_preds = torch.argmax(mask_prediction, dim=-1)
                gender_preds = torch.argmax(gender_prediction, dim=-1)
                age_preds = torch.argmax(age_prediction, dim=-1)
                
                mask_loss = criterion(mask_prediction, mask_labels)
                gender_loss = criterion(gender_prediction, gender_labels)
                age_loss = criterion(age_prediction, age_labels)
                
                loss = mask_loss+gender_loss+age_loss
                loss.backward()
                optimizer.step()

                mask_loss_value += mask_loss.item()
                gender_loss_value += gender_loss.item()
                age_loss_value += age_loss.item()

                mask_matches += (mask_preds == mask_labels).sum().item()
                gender_matches += (gender_preds == gender_labels).sum().item()
                age_matches += (age_preds == age_labels).sum().item()
                if (idx + 1) % args.log_interval == 0:
                    train_mask_loss = mask_loss_value / args.log_interval
                    train_gender_loss = gender_loss_value / args.log_interval
                    train_age_loss = age_loss_value / args.log_interval

                    train_mask_acc = mask_matches / args.batch_size / args.log_interval
                    train_gender_acc = gender_matches / args.batch_size / args.log_interval
                    train_age_acc = age_matches / args.batch_size / args.log_interval

                    train_loss = train_mask_loss+train_gender_loss+train_age_loss
                    train_acc = (train_mask_acc+train_gender_acc+train_age_acc)/3

                    run.log({"train_loss":train_loss})
                    run.log({"train_acc":train_acc})

                    current_lr = get_lr(optimizer)
                    

                    train_logger.info(
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss mask/gender/age {train_mask_loss:4.4}/{train_gender_loss:4.4}/{train_age_loss:4.4} || training accuracy mask/gender/age {train_mask_acc:4.2%}/{train_gender_acc:4.2%}/{train_age_acc:4.2%} || lr {current_lr}"
                    )
                    logger.add_scalar(
                        "Train/mask loss", train_mask_loss, epoch * len(train_loader) + idx
                    )
                    logger.add_scalar(
                        "Train/gender loss", train_gender_loss, epoch * len(train_loader) + idx
                    )
                    logger.add_scalar(
                        "Train/age loss", train_age_loss, epoch * len(train_loader) + idx
                    )
                    logger.add_scalar(
                        "Train/mask accuracy", train_mask_acc, epoch * len(train_loader) + idx
                    )
                    logger.add_scalar(
                        "Train/gender accuracy", train_gender_acc, epoch * len(train_loader) + idx
                    )
                    logger.add_scalar(
                        "Train/age accuracy", train_age_acc, epoch * len(train_loader) + idx
                    )

                    mask_loss_value = 0
                    gender_loss_value = 0
                    age_loss_value = 0
                    mask_matches = 0
                    gender_matches = 0
                    age_matches = 0

            scheduler.step()

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_mask_loss_items = []
                val_gender_loss_items = []
                val_age_loss_items = []
                val_mask_acc_items = []
                val_gender_acc_items = []
                val_age_acc_items = []
                figure = None
                for val_batch in val_loader:
                    inputs, mask_labels, gender_labels, age_labels = val_batch
                    inputs = inputs.to(device)
                    mask_labels, gender_labels, age_labels = mask_labels.to(device), gender_labels.to(device), age_labels.to(device)
                    labels = mask_labels*6+gender_labels*3+age_labels

                    mask_prediction, gender_prediction, age_prediction = model(inputs)

                    mask_preds = torch.argmax(mask_prediction, dim=-1)
                    gender_preds = torch.argmax(gender_prediction, dim=-1)
                    age_preds = torch.argmax(age_prediction, dim=-1)
                    preds = mask_preds*6+gender_preds*3+age_preds
                    
                    mask_loss_item = criterion(mask_prediction, mask_labels).item()
                    gender_loss_item = criterion(gender_prediction, gender_labels).item()
                    age_loss_item = criterion(age_prediction, age_labels).item()

                    mask_acc_item = (mask_labels == mask_preds).sum().item()
                    gender_acc_item = (gender_labels == gender_preds).sum().item()
                    age_acc_item = (age_labels == age_preds).sum().item()

                    val_mask_loss_items.append(mask_loss_item)
                    val_gender_loss_items.append(gender_loss_item)
                    val_age_loss_items.append(age_loss_item)

                    val_mask_acc_items.append(mask_acc_item)
                    val_gender_acc_items.append(gender_acc_item)
                    val_age_acc_items.append(age_acc_item)

                    if figure is None:
                        inputs_np = (
                            torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                        )
                        inputs_np = dataset_module.denormalize_image(
                            inputs_np, dataset.mean, dataset.std
                        )
                        figure = grid_image(
                            inputs_np,
                            labels,
                            preds,
                            n=16,
                            shuffle=args.dataset != "MaskSplitByProfileDataset",
                        )

                val_mask_loss = np.sum(val_mask_loss_items) / len(val_loader)
                val_gender_loss = np.sum(val_gender_loss_items) / len(val_loader)
                val_age_loss = np.sum(val_age_loss_items) / len(val_loader)

                val_mask_acc = np.sum(val_mask_acc_items) / len(val_set)
                val_gender_acc = np.sum(val_gender_acc_items) / len(val_set)
                val_age_acc = np.sum(val_age_acc_items) / len(val_set)

                val_acc = (val_mask_acc+val_gender_acc+val_age_acc)/3
                val_loss = val_mask_loss+val_gender_loss+val_age_loss
                run.log({"val_loss":val_loss})
                run.log({"val_acc":val_acc})
                best_val_loss = min(best_val_loss, val_loss)
                if val_acc > best_val_acc:
                    train_logger.info(
                        f"New best model for val mask/gender/age accuracy : {val_mask_acc:4.2%}/{val_gender_acc:4.2%}/{val_age_acc:4.2%}! saving the best model.."
                    )
                    torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                    artifact = wandb.Artifact(args.name, type='model')
                    artifact.add_file(f"{save_dir}/best.pth")
                    run.log_artifact(artifact)
                    best_val_acc = val_acc
                torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
                train_logger.info(
                    f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                )
                logger.add_scalar("Val/mask loss", val_mask_loss, epoch)
                logger.add_scalar("Val/gender loss", val_gender_loss, epoch)
                logger.add_scalar("Val/age loss", val_age_loss, epoch)
                logger.add_scalar("Val/mask accuracy", val_mask_acc, epoch)
                logger.add_scalar("Val/gender accuracy", val_gender_acc, epoch)
                logger.add_scalar("Val/age accuracy", val_age_acc, epoch)
                logger.add_figure("results", figure, epoch)
                print()
    
    else:
        best_val_acc = 0
        best_val_loss = np.inf
        for epoch in range(args.epochs):
            # train loop
            model.train()
            loss_value = 0
            matches = 0
            for idx, train_batch in enumerate(train_loader):
                inputs, labels = train_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                loss = criterion(outs, labels)

                loss.backward()
                optimizer.step()

                loss_value += loss.item()
                matches += (preds == labels).sum().item()
                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    train_acc = matches / args.batch_size / args.log_interval
                    current_lr = get_lr(optimizer)
                    run.log({"train_loss":train_loss})
                    run.log({"train_acc":train_acc})
                    train_logger.info(
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                    )
                    logger.add_scalar(
                        "Train/loss", train_loss, epoch * len(train_loader) + idx
                    )
                    logger.add_scalar(
                        "Train/accuracy", train_acc, epoch * len(train_loader) + idx
                    )

                    loss_value = 0
                    matches = 0

            scheduler.step()

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items = []
                val_acc_items = []
                figure = None
                for idx, val_batch in enumerate(val_loader):
                    inputs, labels = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)

                    loss_item = criterion(outs, labels).item()
                    acc_item = (labels == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)
                    
                    if figure is None:
                        inputs_np = (
                            torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                        )
                        inputs_np = dataset_module.denormalize_image(
                            inputs_np, dataset.mean, dataset.std
                        )
                        figure = grid_image(
                            inputs_np,
                            labels,
                            preds,
                            n=16,
                            shuffle=args.dataset != "MaskSplitByProfileDataset",
                        )

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(val_set)
                run.log({"val_loss":val_loss})
                run.log({"val_acc":val_acc})
                best_val_loss = min(best_val_loss, val_loss)
                
                if val_acc > best_val_acc:
                    train_logger.info(
                        f"New best model for val accuracy : {val_acc:4.2%}! saving the best model.."
                    )
                    torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                    artifact = wandb.Artifact(args.name, type='model')
                    artifact.add_file(f"{save_dir}/best.pth")
                    run.log_artifact(artifact)
                    best_val_acc = val_acc
                torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
                train_logger.info(
                    f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                )
                logger.add_scalar("Val/loss", val_loss, epoch)
                logger.add_scalar("Val/accuracy", val_acc, epoch)
                logger.add_figure("results", figure, epoch)