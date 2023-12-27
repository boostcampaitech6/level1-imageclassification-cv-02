import argparse
import multiprocessing
import os
from importlib import import_module

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset


def load_model(saved_model, num_classes, device):
    """
    저장된 모델의 가중치를 로드하는 함수입니다.

    Args:
        saved_model (str): 모델 가중치가 저장된 디렉토리 경로
        num_classes (int): 모델의 클래수 수
        device (torch.device): 모델이 로드될 장치 (CPU 또는 CUDA)

    Returns:
        model (nn.Module): 가중치가 로드된 모델
    """
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(num_classes=num_classes)

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    # 모델 가중치를 로드한다.
    model_path = os.path.join(saved_model, "best.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18    
    models = []
    for i in range(5):
        model_fold_dir = os.path.join(model_dir, f'fold{i}')
        model = load_model(model_fold_dir, num_classes, device).to(device)
        models.append(model)
        
    for model in models:
        model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    # 폴드의 수만큼 예측 결과를 저장할 리스트를 초기화
    all_preds = [[] for _ in range(len(models))]
    # 각 모델에 대한 추론을 실행
    for model_idx, model in enumerate(models):
        preds = []
        with torch.no_grad():
            for idx, images in enumerate(loader):
                images = images.to(device)
                if not args.category_train:
                    pred = model(images)
                    pred = pred.argmax(dim=-1)
                else:
                    mask_pred, gender_pred, age_pred = model(images)
                    mask_pred, gender_pred, age_pred = mask_pred.argmax(dim=-1), gender_pred.argmax(dim=-1), age_pred.argmax(dim=-1)
                    pred = mask_pred * 6 + gender_pred * 3 + age_pred
                preds.extend(pred.cpu().numpy())
        all_preds[model_idx] = preds

    final_preds = []
    for idx in range(len(all_preds[0])):
        # 각 샘플에 대해, 모든 폴드에서 가장 많이 예측된 클래스를 선택합니다.
        voting = np.bincount([all_preds[model_idx][idx] for model_idx in range(len(models))])
        final_pred = voting.argmax()
        final_preds.append(final_pred)
        
    info['ans'] = final_preds
    save_path = os.path.join(output_dir, f'output.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == "__main__":
    # 커맨드 라인 인자를 파싱한다.
    parser = argparse.ArgumentParser()

    # 데이터와 모델 체크포인트 디렉터리 관련 인자
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="input batch size for validing (default: 1000)",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        default=(512, 384),
        help="resize size for image when you trained (default: (96, 128))",
    )
    parser.add_argument(
        "--model", type=str, default="Hyun_Resnet34_Model", help="model type (default: BaseModel)"
    )

    # 컨테이너 환경 변수
    parser.add_argument(
        "--data_dir",
        type=str,
        default='/data/ephemeral/home/datasets/eval'
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_MODEL", "./model/exp"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output"),
    )
    parser.add_argument(
        "--category_train", type=bool, default=False
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # 모델 추론을 수행한다.
    inference(data_dir, model_dir, output_dir, args)
