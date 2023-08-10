from typing import Any, List
import os
from nvidia.dali.plugin.pytorch import _DaliBaseIterator
import torch.distributed as dist
import torch


def get_num_samples(index_paths: List[str]) -> int:
    num = 0
    for index_path in index_paths:
        with open(index_path, 'r') as f:
            num += int(f.readline().rstrip('\n').split(' ')[1])
    return num


def get_dataset_bytes(paths: List[str]) -> int:
    size = 0
    for path in paths:
        size += os.path.getsize(path)
    return size


def run_one_epoch(iter: _DaliBaseIterator, distributed: bool) -> None:
    try:
        while True:
            if distributed:
                dist.barrier()
            _ = iter._get_outputs()
            iter._schedule_runs()
            iter._advance_and_check_drop_last()
    except StopIteration:
        pass


def run_one_step(iter: _DaliBaseIterator, distributed: bool) -> None:
    if distributed:
        dist.barrier()
    try:
        _ = iter._get_outputs()
    except StopIteration:
        _ = iter._get_outputs()
    iter._schedule_runs()
    iter._advance_and_check_drop_last()


def reduce_object(object: Any) -> Any:
    tensor = torch.tensor([object], device=torch.device('cpu'))
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    return tensor.item()
