import platform
import re
import traceback
from queue import Empty

from torch.multiprocessing import Manager, Process, current_process, get_context

is_main_process = not bool(re.match(r'((.*Process)|(SyncManager)|(.*PoolWorker))-\d+', current_process().name))


def main_process_print(self, *args, sep=' ', end='\n', file=None):
    if is_main_process:
        print(self, *args, sep=sep, end=end, file=file)


def chunked_worker_run(map_func, args, results_queue=None, device_id=None):
    if device_id is not None:
        import torch
        torch.cuda.set_device(device_id)
        owner = getattr(map_func, '__self__', None)
        if owner is not None:
            owner.device = torch.device(f'cuda:{device_id}')
            if hasattr(owner, 'lr'):
                owner.lr = owner.lr.to(owner.device)
    for a in args:
        # noinspection PyBroadException
        try:
            res = map_func(*a)
            results_queue.put(res)
        except KeyboardInterrupt:
            break
        except Exception:
            traceback.print_exc()
            results_queue.put(None)


def chunked_multiprocess_run(map_func, args, num_workers, q_max_size=1000, device_ids=None):
    num_jobs = len(args)
    if num_jobs == 0:
        return
    if num_workers < 1:
        raise ValueError("num_workers must be positive.")
    if num_jobs < num_workers:
        num_workers = num_jobs
        if device_ids is not None:
            device_ids = device_ids[:num_workers]
    if device_ids is not None and len(device_ids) != num_workers:
        raise ValueError("One device ID is required per worker.")

    manager = Manager()
    queues = [manager.Queue(maxsize=max(1, q_max_size // num_workers)) for _ in range(num_workers)]
    if platform.system().lower() != 'windows':
        process_creation_func = get_context('spawn').Process
    else:
        process_creation_func = Process

    workers = []
    for i in range(num_workers):
        worker = process_creation_func(
            target=chunked_worker_run, args=(map_func, args[i::num_workers], queues[i],
                  None if device_ids is None else device_ids[i]), daemon=True
        )
        workers.append(worker)
        worker.start()

    try:
        for i in range(num_jobs):
            index = i % num_workers
            while True:
                try:
                    result = queues[index].get(timeout=1)
                    break
                except Empty:
                    if not workers[index].is_alive():
                        raise RuntimeError(f'Preprocessing worker {index} exited before returning its result.')
            yield result
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join()
            worker.close()
        manager.shutdown()
