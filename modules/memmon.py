import threading
import time
from collections import defaultdict

import torch
import importlib.util

hthpu = None
if importlib.util.find_spec("habana_frameworks") is not None:
    import habana_frameworks.torch.hpu as hthpu

class MemUsageMonitor(threading.Thread):
    run_flag = None
    device = None
    disabled = False
    opts = None
    data = None

    def __init__(self, name, device, opts):
        threading.Thread.__init__(self)
        self.name = name
        self.device = device
        self.opts = opts

        self.daemon = True
        self.run_flag = threading.Event()
        self.data = defaultdict(int)

        try:
            if self.device.type == 'cuda':
                self.cuda_mem_get_info()
                torch.cuda.memory_stats(self.device)
            elif self.device.type == 'hpu' and hthpu and hthpu.is_available():
                self.hpu_mem_get_info()
            else:
                raise Exception("Unsupported device type")
        except Exception as e:
            print(f"Warning: caught exception '{e}', memory monitor disabled")
            self.disabled = True

    def cuda_mem_get_info(self):
        index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        return torch.cuda.mem_get_info(index)

    def hpu_mem_get_info(self):
        return hthpu.memory_info()

    def run(self):
        if self.disabled:
            return

        while True:
            self.run_flag.wait()

            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            self.data.clear()

            if self.opts.memmon_poll_rate <= 0:
                self.run_flag.clear()
                continue

            if self.device.type == 'cuda':
                self.data["min_free"] = self.cuda_mem_get_info()[0]
            elif self.device.type == 'hpu':
                self.data["min_free"] = self.hpu_mem_get_info().free

            while self.run_flag.is_set():
                if self.device.type == 'cuda':
                    free, total = self.cuda_mem_get_info()
                elif self.device.type == 'hpu':
                    mem_info = self.hpu_mem_get_info()
                    free, total = mem_info.free, mem_info.total

                self.data["min_free"] = min(self.data["min_free"], free)

                time.sleep(1 / self.opts.memmon_poll_rate)

    def dump_debug(self):
        print(self, 'recorded data:')
        for k, v in self.read().items():
            print(k, -(v // -(1024 ** 2)))

        if self.device.type == 'cuda':
            print(self, 'raw torch memory stats:')
            tm = torch.cuda.memory_stats(self.device)
            for k, v in tm.items():
                if 'bytes' not in k:
                    continue
                print('\t' if 'peak' in k else '', k, -(v // -(1024 ** 2)))

            print(torch.cuda.memory_summary())

    def monitor(self):
        self.run_flag.set()

    def read(self):
        if not self.disabled:
            if self.device.type == 'cuda':
                free, total = self.cuda_mem_get_info()
                torch_stats = torch.cuda.memory_stats(self.device)
                self.data["active"] = torch_stats["active.all.current"]
                self.data["active_peak"] = torch_stats["active_bytes.all.peak"]
                self.data["reserved"] = torch_stats["reserved_bytes.all.current"]
                self.data["reserved_peak"] = torch_stats["reserved_bytes.all.peak"]
            elif self.device.type == 'hpu':
                mem_info = self.hpu_mem_get_info()
                free, total = mem_info.free, mem_info.total
                self.data["active"] = hthpu.memory_allocated()  # Current HPU memory occupied by tensors
                self.data["active_peak"] = hthpu.max_memory_allocated()  # Peak HPU memory allocated by tensors
                self.data["reserved"] = mem_info.used
                self.data["reserved_peak"] = mem_info.used
                # Additional HPU memory stats
                hpu_stats = hthpu.memory_stats()
                self.data["total_memory"] = hpu_stats['Limit']  # Total memory on HPU device
                self.data["num_allocs"] = hpu_stats['NumAllocs']  # Number of allocations
                self.data["num_frees"] = hpu_stats['NumFrees']  # Number of freed chunks

            self.data["free"] = free
            self.data["total"] = total
            self.data["system_peak"] = total - self.data["min_free"]

        return self.data

    def stop(self):
        self.run_flag.clear()
        return self.read()
