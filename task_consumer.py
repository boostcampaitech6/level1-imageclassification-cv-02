import redis
import wandb
import os, sys, json, time, traceback, argparse, logging, logging.handlers

from train import train
from git import Repo, exc

repo_path = os.getcwd()

def restart_program():
    python = sys.executable
    os.execl(python, python, *sys.argv)

def is_queue_empty(redis_instance, queue_list):
    queue_length = 0
    for queue_name in queue_list:
        queue_length += redis_instance.llen(queue_name)
    return queue_length == 0

def pull_repo(repo_path):
    try:
        repo = Repo(repo_path)
        current = repo.head.commit
        repo.remotes.origin.pull()
        if current != repo.head.commit:
            print("새로운 업데이트가 pull 되었습니다.")
            print("프로세스를 재시작합니다.")
            restart_program() #변경사항이 존재하면 현재 arg값 그대로 프로세스 재시작
        else:
            print("이미 최신 버전 입니다.")
    except exc.GitCommandError as e:
        print(f"Git 명령 실행 중 오류 발생: {e}")

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

def json2argparse(config):
    parser = argparse.ArgumentParser()
    for key, value in config.items():
        if key == 'mode': continue
        parser.add_argument(f'--{key}', default=value)
    return parser.parse_args()

if __name__ == "__main__":
    pull_repo(repo_path)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ip", type=str, required=True
    )
    parser.add_argument(
        "--port", type=int, required=True
    )
    parser.add_argument(
        "--mode", type=str, nargs='+', required=True
    )

    args = parser.parse_args()
    print(args.mode)

    redis_server = redis.Redis(host=args.ip, port=args.port, db=0)
    while True:
        if not is_queue_empty(redis_server, args.mode): # queue에 message가 존재하면 진입
            pull_repo(repo_path) #git pull
            element = redis_server.brpop(keys=args.mode, timeout=None)
            config = json.loads(element[1].decode('utf-8'))

            conf_args = json2argparse(config)

            data_dir = conf_args.data_dir
            model_dir = conf_args.model_dir

            if not os.path.exists('logs'): os.mkdir('logs')
            train_logger = set_logger(f'logs/{conf_args.name}.log')
            train_logger.info(f"{conf_args.name} Start")
            try:
                train(data_dir, model_dir, conf_args, train_logger)
                wandb.finish()
                train_logger.info(f"{conf_args.name} Finished")
            except:
                train_logger.error(traceback.format_exc())
                wandb.finish()
        else:
            time.sleep(10)