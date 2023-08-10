from typing import List, Tuple
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import _DaliBaseIterator, LastBatchPolicy
from nvidia.dali.backend import SetHostBufferShrinkThreshold, SetHostBufferGrowthFactor
import time
import os
import torch.distributed as dist
from .utils import reduce_object, run_one_step, run_one_epoch


@pipeline_def
def _get_imagenet_load_pipe(file_root: str, num_shards: int, shard_id: int, cache_size: int):
    _ = fn.readers.file(file_root=file_root, num_shards=num_shards, pad_last_batch=True,
                        preserve=True, shuffle_after_epoch=True, shard_id=shard_id,
                        tensor_init_bytes=0, cache_size=cache_size, name='Reader')
    return fn.constant(device='cpu', idata=0, preserve=True)


@pipeline_def
def _get_imagenet_decode_pipe(file_root: str, num_shards: int, shard_id: int, cache_size: int,
                             write_desc: int):
    images, labels = fn.readers.file(file_root=file_root, num_shards=num_shards, pad_last_batch=True,
                                     preserve=True, shuffle_after_epoch=True, shard_id=shard_id,
                                     tensor_init_bytes=0, cache_size=cache_size, name='Reader', resume=False)
    images = fn.decoders.image(images, device='mixed', device_memory_padding=0, host_memory_padding=0,
                               hybrid_huffman_threshold=0, preserve=True, write_desc=write_desc)
    return images, labels


@pipeline_def
def _get_cityscapes_load_pipe(img_files: List[str], lbl_files: List[str], num_shards: int,
                             shard_id: int, cache_size: int):
    _ = fn.readers.file(files=img_files, num_shards=num_shards, pad_last_batch=True, preserve=True,
                        shard_id=shard_id, tensor_init_bytes=0, cache_size=cache_size,
                        name='Reader', shuffle_after_epoch=True)
    _ = fn.readers.file(files=lbl_files, num_shards=num_shards, pad_last_batch=True, preserve=True,
                        shard_id=shard_id, tensor_init_bytes=0, cache_size=cache_size, shuffle_after_epoch=True)
    return fn.constant(device='cpu', idata=0, preserve=True)


@pipeline_def
def _get_cityscapes_decode_pipe(img_files: List[str], lbl_files: List[str], num_shards: int, shard_id: int, cache_size: int, write_desc: int):
    images, _ = fn.readers.file(files=img_files, num_shards=num_shards, pad_last_batch=True, preserve=True,
                                shard_id=shard_id, tensor_init_bytes=0, cache_size=cache_size, shuffle_after_epoch=True,
                                name='Reader', resume=False)
    labels, _ = fn.readers.file(files=lbl_files, num_shards=num_shards, pad_last_batch=True, preserve=True,
                                shard_id=shard_id, tensor_init_bytes=0, cache_size=cache_size, resume=False, shuffle_after_epoch=True)
    images = fn.decoders.image(images, device='mixed', device_memory_padding=0, host_memory_padding=0, hybrid_huffman_threshold=0,
                               preserve=True, write_desc=write_desc)
    labels = fn.decoders.image(labels, device='mixed', device_memory_padding=0, host_memory_padding=0, hybrid_huffman_threshold=0,
                               preserve=True, write_desc=write_desc)
    return images, labels


def CoorDL_profile_load(device_id: int, batch_size: int, num_threads: int, buffer_ratio: float, num_shards: int,
                        num_warmups: int, num_steps: int, dir_paths: List[str], init_method: str, kind: str) -> float:
    '''
    ex) dir_path: ~/datasets/imagenet/train, ~/datasets/cityscapes
    '''
    distributed = num_shards > 1
    if distributed:
        dist.init_process_group('gloo', init_method=init_method, world_size=num_shards, rank=device_id)

    SetHostBufferShrinkThreshold(1)
    SetHostBufferGrowthFactor(1)

    if kind == 'imagenet':
        paths = []
        assert len(dir_paths) == 1
        dir_path = dir_paths[0]
        for root, _, files in os.walk(dir_path):
            for file in files:
                paths.append(os.path.join(root, file))
        nsamples = len(paths)
        cache_size = int(nsamples * buffer_ratio / num_shards)

        load_pipe = _get_imagenet_load_pipe(batch_size=batch_size, num_threads=num_threads, device_id=device_id,
                                            file_root=dir_path, num_shards=num_shards, shard_id=device_id, cache_size=cache_size)
    else:  # cityscapes
        img_files = []
        lbl_files = []
        for subdir in dir_paths:
            for root, _, files in os.walk(subdir):
                for file in files:
                    path = os.path.join(root, file)
                    img_files.append(path)
        img_files.sort()
        for img_file in img_files:
            root, file = os.path.split(img_file)
            annot_kind = 'gtCoarse' if 'train_extra' in root else 'gtFine'
            image_kind = 'leftImg8bit' if 'leftImg8bit' in root else 'rightImg8bit'
            lbl_path = os.path.join(root.replace(image_kind, annot_kind),
                                    file.replace(image_kind, annot_kind + '_labelTrainIds'))
            lbl_files.append(lbl_path)
        nsamples = len(img_files)
        cache_size = int(nsamples * buffer_ratio / num_shards)

        load_pipe = _get_cityscapes_load_pipe(batch_size=batch_size, num_threads=num_threads, device_id=device_id,
                                              img_files=img_files, lbl_files=lbl_files, num_shards=num_shards, shard_id=device_id, cache_size=cache_size)

    load_pipe.build()
    load_iter = _DaliBaseIterator(
        [load_pipe], reader_name='Reader', auto_reset=True, last_batch_policy=LastBatchPolicy.PARTIAL)

    # load one epoch
    run_one_epoch(load_iter, distributed)

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


def CoorDL_profile_prep(device_id: int, batch_size: int, num_threads: int, buffer_ratio: float,
            num_shards: int, num_warmups: int, num_steps: int, dir_paths: List[str], init_method: str, kind: str) -> Tuple[float, float]:
    '''
    ex) dir_path: ~/datasets/imagenet/train, ~/datasets/cityscapes
    '''
    assert kind in ['imagenet', 'cityscapes']
    distributed = num_shards > 1
    if distributed:
        dist.init_process_group('gloo', init_method=init_method, world_size=num_shards, rank=device_id)

    SetHostBufferShrinkThreshold(1)
    SetHostBufferGrowthFactor(1)

    rd, wd = os.pipe()

    if kind == 'imagenet':
        assert len(dir_paths) == 1
        dir_path = dir_paths[0]
        paths = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                paths.append(os.path.join(root, file))
        nsamples = len(paths)
        cache_size = int(nsamples * buffer_ratio / num_shards)

        decode_pipe = _get_imagenet_decode_pipe(batch_size=batch_size, num_threads=num_threads, device_id=device_id,
                                                file_root=dir_path, num_shards=num_shards, shard_id=device_id, cache_size=cache_size, write_desc=wd)
    else:  # cityscapes
        img_files = []
        lbl_files = []
        for subdir in dir_paths:
            for root, _, files in os.walk(subdir):
                for file in files:
                    path = os.path.join(root, file)
                    img_files.append(path)
        img_files.sort()
        for img_file in img_files:
            root, file = os.path.split(img_file)
            annot_kind = 'gtCoarse' if 'train_extra' in root else 'gtFine'
            image_kind = 'leftImg8bit' if 'leftImg8bit' in root else 'rightImg8bit'
            lbl_path = os.path.join(root.replace(image_kind, annot_kind),
                                    file.replace(image_kind, annot_kind + '_labelTrainIds'))
            lbl_files.append(lbl_path)
        nsamples = len(img_files)
        cache_size = int(nsamples * buffer_ratio / num_shards)

        decode_pipe = _get_cityscapes_decode_pipe(batch_size=batch_size, num_threads=num_threads, device_id=device_id,
                                                  img_files=img_files, lbl_files=lbl_files, num_shards=num_shards, shard_id=device_id, cache_size=cache_size, write_desc=wd)

    decode_pipe.build()
    decode_iter = _DaliBaseIterator(
        [decode_pipe], reader_name='Reader', auto_reset=True, last_batch_policy=LastBatchPolicy.PARTIAL)

    nbatches = 1 if kind == 'imagenet' else 2
    rf = os.fdopen(rd, 'r')

    run_one_epoch(decode_iter, distributed)
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
        return decode_thpt, prep_thpt

    return 0, 0
