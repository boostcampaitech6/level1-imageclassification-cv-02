import glob
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion
from function import training

import wandb

def getDataloader(train_set, val_set, batch_size, valid_batch_size, num_workers, use_cuda):
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=valid_batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )
    return train_loader, val_loader

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_optimizer(optimizer_name, m_parameters, lr):
    opt_module = getattr(import_module("torch.optim"), optimizer_name)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, m_parameters),
        lr=lr,
        weight_decay=5e-4,
    )
    return optimizer

def get_model(model_name):
    model_module = getattr(import_module("model"), model_name)  # default: BaseModel
    model = model_module(num_classes=18)
    model = torch.nn.DataParallel(model)
    return model

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


def increment_path(path, exist_ok=False):
    """Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(data_dir, model_dir, args, train_logger):
    train_logger.info(args)
    
    run = wandb.init(
        project="mask_classification", 
        entity = "ai_tech_6th_cv_level1",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=args.name,
        # Track hyperparameters and run metadata
        config = { 
            "seed" : args.seed,
            "epochs" : args.epochs,
            "dataset" : args.dataset,
            "augmentation" : args.augmentation,
            "resize" : args.resize,
            "batch_size" : args.batch_size,
            "valid_batch_size" : args.valid_batch_size,
            "model" : args.model,
            "optimizer" : args.optimizer,
            "lr" : args.lr,
            "val_ratio" : args.val_ratio,
            "criterion" : args.criterion,
            "lr_decay_step" : args.lr_decay_step,
            "log_interval" : args.log_interval,
            "data_dir" : args.data_dir,
            "model_dir" : args.model_dir,
            "k_fold": args.k_fold,
            "category_train": args.category_train
        }
    )

    train_logger.info("wandb init")

    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_logger.info(f"cuda device : {device}")

    # -- dataset
    dataset_module = getattr(
        import_module("dataset"), args.dataset
    )  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
        category_train=args.category_train
    )

    # -- augmentation
    transform_module = getattr(
        import_module("dataset"), args.augmentation
    )  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    n_splits = args.k_fold
    if n_splits == -1:
        # -- data_loader
        train_set, val_set = dataset.split_dataset()
        torch.cuda.empty_cache()
        # -- model
        model = get_model(args.model)
        model = model.to(device)

        # -- loss & metric
        criterion = create_criterion(args.criterion)  # default: cross_entropy
        optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr)
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

        training(model, train_set, val_set, criterion, optimizer, scheduler, dataset, dataset_module, device, train_logger, run, save_dir, args)
        


    else:
        labels = [
            dataset.encode_multi_class(mask, gender, age)
            for mask, gender, age in zip(
                dataset.mask_labels, dataset.gender_labels, dataset.age_labels
            )
        ]
        # 5-fold Stratified KFold 5개의 fold를 형성하고 5번 Cross Validation을 진행합니다.
        skf = StratifiedKFold(n_splits=n_splits)
        for i, (train_idx, valid_idx) in enumerate(skf.split(dataset.image_paths, labels)):
            print(f"Fold:{i}, Train set: {len(train_idx)}, Valid set:{len(valid_idx)}")
            fold_dir = f"{save_dir}/fold{i}"

            # -- data_loader
            train_set, val_set = Subset(dataset, train_idx), Subset(dataset, valid_idx)
            torch.cuda.empty_cache()
            # -- model
            model = get_model(args.model)
            model = model.to(device)

            # -- loss & metric
            criterion = create_criterion(args.criterion)  # default: cross_entropy
            optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr)
            scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

            training(model, train_set, val_set, criterion, optimizer, scheduler, dataset, dataset_module, device, train_logger, run, fold_dir, args)