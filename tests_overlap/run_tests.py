import queue
import argparse
import threading
import subprocess

test_cases = """
python test_chunk.py
python test_sort_map.py
python test_token_scatter.py
python test_zip.py

python test_forward_backward.py
python test_forward_backward.py --act swiglu,1.0
python test_forward_backward.py --act situ,4.0
python test_forward_backward.py --act situ,4.0,25.0

python test_forward_backward_fp8.py
python test_forward_backward_fp8.py --act swiglu,1.0
python test_forward_backward_fp8.py --act situ,4.0
python test_forward_backward_fp8.py --act situ,4.0,25.0
"""

GPUS = []
task_queue = queue.Queue()
failed_cases = []


def test_runner(tid):
    gpu = GPUS[tid]
    header = f"[WORKER {tid}][GPU {gpu}]"

    while cmd := task_queue.get():
        print(end=f"{header} {cmd}\n", flush=True)
        with open(f"workerlog.{tid}", "a") as f:
            f.write("=" * 80 + "\n")
            f.write(cmd + "\n")

        msg = f"workerlog.{tid}: {cmd}\n"
        cmd = f"CUDA_VISIBLE_DEVICES={gpu} {cmd} >> workerlog.{tid} 2>&1"
        p = subprocess.Popen(cmd, shell=True, stdin=subprocess.DEVNULL)
        ret = p.wait()

        with open(f"workerlog.{tid}", "a") as f:
            f.write(f"{ret}\n")
        if ret:
            failed_cases.append(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gpus", help="Specify gpus to use, such as '1,2,3,4'")
    args = parser.parse_args()
    GPUS = [int(i) for i in args.gpus.split(",")]

    threads = []
    for tid in range(len(GPUS)):
        thread = threading.Thread(target=test_runner, args=(tid,))
        thread.start()
        threads.append(thread)

    print(end=f"[MAIN] Launched {len(GPUS)} workers\n", flush=True)

    for line in test_cases.splitlines():
        if line := line.strip():
            task_queue.put(line)
    for _ in range(len(GPUS)):
        task_queue.put(None)

    for thread in threads:
        thread.join()

    if failed_cases:
        print(end="[MAIN] FAILED CASES:\n\n" + "".join(failed_cases) + "\n", flush=True)
        exit(1)
    else:
        print(end="[MAIN] ALL PASSED\n", flush=True)
