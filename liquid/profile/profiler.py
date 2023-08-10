from __future__ import annotations
from typing import Any, Dict, List
import os
from ..datasets.converter import Converter
import torch.multiprocessing as mp
from .Liquid import Liquid_profile_load, Liquid_profile_prep


class Profiler:
    def __init__(self, converter: Converter, num_shards: int, init_method: str = 'tcp://127.0.0.1:12345') -> None:
        self.converter = converter
        self.num_shards = num_shards
        self.init_method = init_method
        self.queue = mp.get_context('spawn').SimpleQueue()


    def profile(self, batch_size: int, size_0: int, size_100: int, num_threads: int = 1, buffer_size: int = 0,
                interval: int = 10, num_warmups: int = 100, num_steps: int = 200, delete: bool = False) -> int:
        '''
        size_0, size_100, buffer_size: bytes
        '''
        buffer_ratio = buffer_size / size_0
        # _, _, prep_thpt = self._profile(
        #     0, batch_size, num_threads, buffer_ratio, num_warmups, num_steps)
        # max_thpt = prep_thpt
        max_thpt = 0
        opt_ratio = 0

        cand = list(range(interval, 101, interval))
        left = 0
        right = len(cand)

        while left < right:
            mid = (left + right) // 2
            mid = 4

            buffer_ratio = 100 * buffer_size / ((100 - cand[mid]) * size_0 + cand[mid] * size_100)
            load_thpt, decode_thpt, prep_thpt = self._profile(
                cand[mid], batch_size, num_threads, buffer_ratio, num_warmups, num_steps)

            if max_thpt < prep_thpt:
                max_thpt = prep_thpt
                opt_ratio = cand[mid]

            if delete:
                self.converter.delete('part', cand[mid])

            if load_thpt < decode_thpt:
                right = mid
            else:
                left = mid + 1
            break

        if delete:
            self.converter.delete('part', 0)
            self.converter.delete('part', 100)

        print('optimal ratio: {} %'.format(opt_ratio), flush=True)
        return opt_ratio


    def _profile(self, ratio: int, batch_size: int, num_threads: int = 1, buffer_ratio: float = 0,
                 num_warmups: int = 10, num_steps: int = 20) -> int:
        self.converter.convert('part', ratio)
        os.system('sudo sh -c "sync; echo 1 > /proc/sys/vm/drop_caches"')

        self._spawn(self._load_worker, nprocs=self.num_shards, args=(batch_size, num_threads, buffer_ratio,
                    self.num_shards, num_warmups, num_steps, self.converter.to_dir, self.converter.info,
                    self.queue, self.init_method))
        load_thpt = self.queue.get()

        self._spawn(self._prep_worker, nprocs=self.num_shards, args=(batch_size, num_threads, buffer_ratio,
                    self.num_shards, num_warmups, num_steps, self.converter.to_dir, self.converter.info,
                    self.queue, self.init_method))
        decode_thpt, prep_thpt = self.queue.get()

        print('ratio: {} %, load thpt: {}, decode thpt: {}, prep_thpt: {}'.format(ratio, load_thpt, decode_thpt, prep_thpt), flush=True)
        return load_thpt, decode_thpt, prep_thpt


    @staticmethod
    def _spawn(fn, args=(), nprocs=1):
        if nprocs == 1:
            fn(0, *args)
        else:
            mp.spawn(fn=fn, nprocs=nprocs, args=args)


    @staticmethod
    def _load_worker(device_id: int, batch_size: int, num_threads: int, buffer_ratio: float, num_shards: int,
                     num_warmups: int, num_steps: int, dir_path: str, info: Dict[str, List[Any]],
                     queue: mp.SimpleQueue, init_method: str) -> None:
        load_thpt = Liquid_profile_load(device_id, batch_size, num_threads, buffer_ratio,
                                        num_shards, num_warmups, num_steps, [dir_path], info, init_method)
        if device_id == 0:
            queue.put(load_thpt)


    @staticmethod
    def _prep_worker(device_id: int, batch_size: int, num_threads: int, buffer_ratio: float,
                num_shards: int, num_warmups: int, num_steps: int, dir_path: str, info: Dict[str, List[Any]],
                queue: mp.SimpleQueue, init_method: str) -> None:
        decode_thpt, prep_thpt = Liquid_profile_prep(
            device_id, batch_size, num_threads, buffer_ratio, num_shards, num_warmups, num_steps, [dir_path], info, init_method)
        if device_id == 0:
            queue.put((decode_thpt, prep_thpt))
