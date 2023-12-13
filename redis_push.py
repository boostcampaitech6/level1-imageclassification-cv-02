import redis, json, time, os

if __name__ == "__main__":
    redis_server = redis.Redis(host='10.28.224.190', port=30062, db=0)

    config = {
        "mode" :  "train", # train or eval
        "experiment_name" : "hyun_test_experiment",
        "seed" : 42,
        "epochs" : 10,
        "dataset" : "MaskBaseDataset",
        "augmentation" : "BaseAugmentation",
        "resize" : [128, 96],
        "batch_size" : 64,
        "valid_batch_size" : 1000,
        "model" : "BaseModel",
        "optimizer" : "SGD",
        "lr" : 1e-3,
        "val_ratio" : 0.2,
        "criterion" : "cross_entropy",
        "lr_decay_step" : 20,
        "log_interval" : 20,
        "name" : "exp",
        "data_dir" : 'datasets/train/images',
        "model_dir" : os.environ.get("SM_MODEL_DIR", "./model")
    }
    print(config)
    redis_server.lpush('train', json.dumps(config))