import argparse
import redis
import json

def list_tasks(redis_server):
    """대기 중인 작업 목록을 조회하는 함수"""
    print("\nFetching waiting tasks...")
    waiting_tasks = redis_server.lrange('train', 0, -1)
    if not waiting_tasks:
        print("No waiting tasks.")
    else:
        print("Waiting tasks:")
        for task in waiting_tasks:
            task_config = json.loads(task.decode('utf-8'))
            task_name = task_config.get("name", "Unknown")
            print(f"- {task_name}")

def delete_tasks(redis_server, delete_jobs):
    """여러 작업을 삭제하는 함수"""
    for deletejob in delete_jobs:
        found = False
        waiting_tasks = redis_server.lrange('train', 0, -1)
        for task in waiting_tasks:
            if json.loads(task.decode('utf-8'))["name"] == deletejob.strip():
                redis_server.lrem('train', 0, task)
                print(f"Task '{deletejob.strip()}' deleted.")
                found = True
                break
        if not found:
            print(f"Task '{deletejob.strip()}' not found.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, required=True, help="Redis server IP")
    parser.add_argument("--port", type=int, required=True, help="Redis server port")
    args = parser.parse_args()

    redis_server = redis.Redis(host=args.ip, port=args.port, db=0)

    while True:
        command = input("Enter 'refresh' to list tasks, 'deletejob' to delete a task, or 'exit' to close: ")
        if command == "refresh":
            list_tasks(redis_server)
        elif command == "deletejob":
            job_names = input("Enter the names of the tasks to delete (separated by commas): ")
            delete_jobs = job_names.split(',')
            delete_tasks(redis_server, delete_jobs)
        elif command == "exit":
            break

if __name__ == "__main__":
    main()
