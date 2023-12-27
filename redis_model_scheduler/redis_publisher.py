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
        "ip" : args.ip,
        "port" : args.port,
        "mode" :  "train",
        "seed" : 42,
        "epochs" : 100,
        "dataset" : "MaskBaseDataset",
        "augmentation" : "ImgaugAugmentation",
        "resize" : [260, 260],
        "batch_size" : 64,
        "valid_batch_size" : 1000,
        "model" : "EfficientNet_b2",
        "optimizer" : "Adam",
        "lr" : 1e-3,
        "val_ratio" : 0.2,
        "criterion" : "f1",
        "lr_decay_step" : 20,
        "log_interval" : 10,
        "data_dir" : '/data/ephemeral/home/datasets/trainQA/images',
        "model_dir" : os.environ.get("SM_MODEL_DIR", "./model"),
        "k_fold" : -1,
        "category_train" : False
    }
    experiment_name = f"{time.strftime('%y%m%d%H%M')}_{args.user}_{config['model']}_{config['optimizer']}_{config['criterion']}_{config['batch_size']}_{config['augmentation']}_{config['epochs']}"
    config["name"] = experiment_name
    print(config)
    redis_server.lpush(config["mode"], json.dumps(config))