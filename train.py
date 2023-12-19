import argparse
import glob
import json
import multiprocessing
import os
import random
import re
import redis
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
import logging
import logging.handlers
import traceback
from git import Repo, exc

repo_path = os.getcwd()

def pull_repo(repo_path):
    try:
        repo = Repo(repo_path)
        current = repo.head.commit
        repo.remotes.origin.pull()
        if current != repo.head.commit:
            print("새로운 업데이트가 pull 되었습니다.")
            print(repo.head.commit)
        else:
            print("이미 최신 버전 입니다.")
    except exc.GitCommandError as e:
        print(f"Git 명령 실행 중 오류 발생: {e}")

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

def set_logger(log_path):
    train_logger = logging.getLogger(log_path)
    train_logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] >> %(message)s')
    fileHandler = logging.handlers.TimedRotatingFileHandler(filename=log_path, encoding='utf-8')
    fileHandler.setFormatter(formatter)
    train_logger.addHandler(fileHandler)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    train_logger.addHandler(streamHandler)
    return train_logger

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_optimizer(optimizer_name, m_parameters):
    opt_module = getattr(import_module("torch.optim"), optimizer_name)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, m_parameters),
        lr=args.lr,
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

        # -- model
        model = get_model(args.model)
        model = model.to(device)

        # -- loss & metric
        criterion = create_criterion(args.criterion)  # default: cross_entropy
        optimizer = get_optimizer(args.optimizer, model.parameters())
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

            # -- model
            model = get_model(args.model)
            model = model.to(device)

            # -- loss & metric
            criterion = create_criterion(args.criterion)  # default: cross_entropy
            optimizer = get_optimizer(args.optimizer, model.parameters())
            scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

            training(model, train_set, val_set, criterion, optimizer, scheduler, dataset, dataset_module, device, train_logger, run, fold_dir, args)

if __name__ == "__main__":
    # pull_repo(repo_path)
    parser = argparse.ArgumentParser()
    # Data and model checkpoints directories
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train (default: 1)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MaskBaseDataset",
        help="dataset augmentation type (default: MaskBaseDataset)",
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        default="BaseAugmentation",
        help="data augmentation type (default: BaseAugmentation)",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        default=[128, 96],
        help="resize size for image when training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--valid_batch_size",
        type=int,
        default=1000,
        help="input batch size for validing (default: 1000)",
    )
    parser.add_argument(
        "--model", type=str, default="BaseModel", help="model type (default: BaseModel)"
    )
    parser.add_argument(
        "--optimizer", type=str, default="SGD", help="optimizer type (default: SGD)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="ratio for validaton (default: 0.2)",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="cross_entropy",
        help="criterion type (default: cross_entropy)",
    )
    parser.add_argument(
        "--lr_decay_step",
        type=int,
        default=20,
        help="learning rate scheduler deacy step (default: 20)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=20,
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--name", default="exp", help="model save at {SM_MODEL_DIR}/{name}"
    )

    # Container environment
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train/images"),
    )
    parser.add_argument(
        "--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model")
    )

    parser.add_argument(
        "--k_fold", type=int, default=-1
    )
    parser.add_argument(
        "--category_train", type=bool, default=False
    )

    parser.add_argument(
        "--ip", type=str, required=True
    )
    parser.add_argument(
        "--port", type=int, required=True
    )

    args = parser.parse_args()
    
    redis_server = redis.Redis(host=args.ip, port=args.port, db=0)
    while True:
        element = redis_server.brpop(keys='train_hyun', timeout=None) # 큐가 비어있을 때 대기
        config = json.loads(element[1].decode('utf-8'))

        args.seed = config['seed']
        args.epochs = config['epochs']
        args.dataset = config['dataset']
        args.augmentation = config['augmentation']
        args.resize = config['resize']
        args.batch_size = config['batch_size']
        args.valid_batch_size = config['valid_batch_size']
        args.model = config['model']
        args.optimizer = config['optimizer']
        args.lr = config['lr']
        args.val_ratio = config['val_ratio']
        args.criterion = config['criterion']
        args.lr_decay_step = config['lr_decay_step']
        args.log_interval = config['log_interval']
        args.name = config['name']
        args.data_dir = config['data_dir']
        args.model_dir = config['model_dir']
        args.k_fold = config['k_fold']
        args.category_train = config['category_train']

        data_dir = args.data_dir
        model_dir = args.model_dir

        if not os.path.exists('logs'): os.mkdir('logs')
        train_logger = set_logger(f'logs/{args.name}.log')
        train_logger.info(f"{args.name} Start")
        try:
            # pull_repo(repo_path)
            train(data_dir, model_dir, args, train_logger)
            wandb.finish()
            train_logger.info(f"{args.name} Finished")
        except:
            train_logger.error(traceback.format_exc())
            wandb.finish()
