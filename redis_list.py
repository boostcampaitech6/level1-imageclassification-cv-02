import argparse
import redis
import json

def list_tasks(redis_server):
    """대기 중인 작업 목록을 조회하는 함수"""
    waiting_tasks = redis_server.lrange('train', 0, -1)
    if not waiting_tasks:
        print("No waiting tasks.")
        return

    print("Waiting tasks:")
    for task in waiting_tasks:
        task_config = json.loads(task.decode('utf-8'))
        task_name = task_config.get("name", "Unknown")  # name 키가 없는 경우 "Unknown" 출력
        print(f"- {task_name}")


def delete_task(redis_server, deletejob):
    """특정 작업을 삭제하는 함수"""
    # 삭제할 작업을 찾기 위해 대기 목록을 순회
    waiting_tasks = redis_server.lrange('train', 0, -1)
    for task in waiting_tasks:
        if json.loads(task.decode('utf-8'))["name"] == deletejob:
            redis_server.lrem('train', 0, task)
            print(f"Task '{deletejob}' deleted.")
            return
    print(f"Task '{deletejob}' not found.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, required=True, help="Redis server IP")
    parser.add_argument("--port", type=int, required=True, help="Redis server port")
    parser.add_argument("--deletejob", type=str, help="Name of the task to delete")
    args = parser.parse_args()

    redis_server = redis.Redis(host=args.ip, port=args.port, db=0)

    if args.deletejob:
        delete_task(redis_server, args.deletejob)
    else:
        list_tasks(redis_server)

if __name__ == "__main__":
    main()

'''
python redis_list.py --ip {redis server ip주소} --port {redis port} --deletejob {삭제하려는 queue name}
'''