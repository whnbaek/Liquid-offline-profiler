from typing import Any, Dict, List, Tuple, Union
import torch.distributed as dist
import os
import time
import nvidia.dali.fn as fn
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.backend import SetHostBufferShrinkThreshold, SetHostBufferGrowthFactor
from nvidia.dali.plugin.pytorch import _DaliBaseIterator, LastBatchPolicy
from .utils import get_num_samples, run_one_epoch, run_one_step, reduce_object
import psutil

@pipeline_def
def _get_load_pipe(ext: Union[str, List[str]], paths: Union[str, List[str]],
                dtypes: Union[Any, List[Any]], index_paths: Union[str, List[str]],
                initial_fill: int, num_shards: int, shard_id: int):
    _ = fn.readers.webdataset(ext=ext, paths=paths, dtypes=dtypes, index_paths=index_paths,
                            pad_last_batch=True,  preserve=True,
                            random_shuffle=True, initial_fill=initial_fill,
                            missing_component_behavior='error', num_shards=num_shards,
                            shard_id=shard_id, stick_to_shard=True, tensor_init_bytes=0,
                            name='Reader')
    return fn.constant(device='cpu', idata=0, preserve=True)


@pipeline_def
def _get_prep_pipe(ext: Union[str, List[str]], paths: Union[str, List[str]],
                   dtypes: Union[Any, List[Any]], index_paths: Union[str, List[str]],
                   initial_fill: int, num_shards: int, shard_id: int, is_images: List[str],
                   output_type: List[Union[Any, None]], write_desc: int):
    data = fn.readers.webdataset(ext=ext, paths=paths, dtypes=dtypes, index_paths=index_paths,
                                 pad_last_batch=True, preserve=True,
                                 random_shuffle=True, initial_fill=initial_fill,
                                 missing_component_behavior='error', num_shards=num_shards,
                                 shard_id=shard_id, stick_to_shard=True, tensor_init_bytes=0,
                                 name='Reader')
    for i, is_image in enumerate(is_images):
        if is_image:
            data[i] = fn.decoders.image(data[i], device='mixed', device_memory_padding=0,
                                        host_memory_padding=0, hybrid_huffman_threshold=0,
                                        output_type=output_type[i], preserve=True,
                                        write_desc=write_desc)
    return tuple(data)


def Liquid_profile_load(device_id: int, batch_size: int, num_threads: int, buffer_ratio: float, num_shards: int,
                        num_warmups: int, num_steps: int, dir_paths: List[str], info: Dict[str, List[Any]],
                        init_method: str) -> float:
    distributed = num_shards > 1
    if distributed:
        dist.init_process_group('gloo', init_method=init_method, world_size=num_shards, rank=device_id)

    paths = []
    for dir_path in dir_paths:
        for tar in os.listdir(dir_path):
            if '.tar' in tar:
                paths.append(os.path.join(dir_path, tar))
    paths.sort()
    index_paths = [path.replace('.tar', '.idx') for path in paths]

    SetHostBufferShrinkThreshold(1)
    SetHostBufferGrowthFactor(1)

    nsamples = get_num_samples(index_paths)
    initial_fill = int(nsamples * buffer_ratio / num_shards)

    load_pipe = _get_load_pipe(batch_size=batch_size, num_threads=num_threads, device_id=device_id,
                               ext=info['ext'], paths=paths, dtypes=info['dtypes'], index_paths=index_paths,
                               initial_fill=initial_fill, num_shards=num_shards, shard_id=device_id)
    load_pipe.build()
    load_iter = _DaliBaseIterator([load_pipe], reader_name='Reader', auto_reset=True, last_batch_policy=LastBatchPolicy.PARTIAL)

    # warm up
    for _ in range(num_warmups):
        run_one_step(load_iter, distributed)

    if device_id == 0:
        start = time.perf_counter()

    for _ in range(num_steps):
        run_one_step(load_iter, distributed)

    if device_id == 0:
        elapsed = time.perf_counter() - start
        load_thpt = num_steps * num_shards * batch_size / elapsed
        return load_thpt

    return 0

def Liquid_profile_prep(device_id: int, batch_size: int, num_threads: int, buffer_ratio: float,
            num_shards: int, num_warmups: int, num_steps: int, dir_paths: List[str], info: Dict[str, List[Any]],
            init_method: str) -> Tuple[float, float]:
    if device_id == 0:
        start_bytes = psutil.virtual_memory()._asdict()['used']
    distributed = num_shards > 1
    if distributed:
        dist.init_process_group('gloo', init_method=init_method, world_size=num_shards, rank=device_id)

    paths = []
    for dir_path in dir_paths:
        for tar in os.listdir(dir_path):
            if '.tar' in tar:
                paths.append(os.path.join(dir_path, tar))
    paths.sort()
    index_paths = [path.replace('.tar', '.idx') for path in paths]

    SetHostBufferShrinkThreshold(1)
    SetHostBufferGrowthFactor(1)

    nsamples = get_num_samples(index_paths)
    initial_fill = int(nsamples * buffer_ratio / num_shards)

    rd, wd = os.pipe()
    decode_pipe = _get_prep_pipe(batch_size=batch_size, num_threads=num_threads, device_id=device_id,
                                 ext=info['ext'], paths=paths, dtypes=info['dtypes'], index_paths=index_paths,
                                 initial_fill=initial_fill, num_shards=num_shards, shard_id=device_id,
                                 is_images=info['is_images'], output_type=info['output_type'], write_desc=wd)
    decode_pipe.build()
    decode_iter = _DaliBaseIterator([decode_pipe], reader_name='Reader', auto_reset=True, last_batch_policy=LastBatchPolicy.PARTIAL)

    nbatches = sum(info['is_images'])
    rf = os.fdopen(rd, 'r')

    # run_one_epoch(decode_iter, distributed)
    # warm up
    for _ in range(num_warmups):
        run_one_step(decode_iter, distributed)
    for _ in range(num_warmups * nbatches):
        _ = rf.readline()

    if device_id == 0:
        start = time.perf_counter()

    for _ in range(num_steps):
        run_one_step(decode_iter, distributed)

    if device_id == 0:
        elapsed = time.perf_counter() - start
        prep_thpt = num_steps * num_shards * batch_size / elapsed

    decode_elapsed = 0
    for _ in range(num_steps * nbatches):
        decode_elapsed += int(rf.readline()) * 1e-9
    rf.close()
    nsamples_decoded = num_shards * num_steps * batch_size

    if distributed:
        decode_elapsed = reduce_object(decode_elapsed)

    if device_id == 0:
        decode_thpt = nsamples_decoded * num_shards / decode_elapsed
        bytes_used = psutil.virtual_memory()._asdict()['used'] - start_bytes
        print(bytes_used)
        return decode_thpt, prep_thpt

    return 0, 0
