import redis, json, time, os, argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ip", type=str, required=True, help="redis ip"
    )
    parser.add_argument(
        "--port", type=int, required=True, help="redis port"
    )
    parser.add_argument(
        "--user", type=str, required=True, help="user name"
    )
    args = parser.parse_args()

    redis_server = redis.Redis(host=args.ip, port=args.port, db=0)
    
    config = {
        "mode" :  "train",
        "seed" : 42,
        "epochs" : 5,
        "dataset" : "MaskBaseDataset",
        "augmentation" : "BaseAugmentation",
        "resize" : [512, 384],
        "batch_size" : 64,
        "valid_batch_size" : 64,
        "model" : "BaseModel",
        "optimizer" : "SGD",
        "lr" : 1e-3,
        "val_ratio" : 0.2,
        "criterion" : "cross_entropy",
        "lr_decay_step" : 20,
        "log_interval" : 10,
        "data_dir" : '/data/ephemeral/home/datasets/train/images',
        "model_dir" : os.environ.get("SM_MODEL_DIR", "./model"),
        "kfold" : "None", # None, KFold, Stratified
        "splits" : 5 # !주의! KFold를 사용하면 epochs * splits 만큼 학습을 진행합니다.
    }
    experiment_name = f"{time.strftime('%y%m%d%H%M')}_{args.user}_{config['model']}_{config['optimizer']}_{config['criterion']}_{config['batch_size']}_{config['augmentation']}_{config['epochs']}"
    config["name"] = experiment_name
    print(config)
    redis_server.lpush(config["mode"], json.dumps(config))