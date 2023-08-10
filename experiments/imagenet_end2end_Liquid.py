import argparse
import os
import time
import psutil

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from torch.nn.parallel import DistributedDataParallel as DDP

from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

from liquid.profile.utils import get_dataset_bytes
from glob import glob


def parse():
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--path', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')

    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('--buffer_size', default=0, type=int,
                        help='size of buffer to use (KB)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU to be used')
    args = parser.parse_args()
    return args


@pipeline_def
def create_dali_pipeline(paths, idxs, initial_fill, num_shards, shard_id):
    images, labels = fn.readers.webdataset(device='cpu', ext=['image', 'label'], paths=paths,
                                           dtypes=[
                                               types.DALIDataType.UINT8, types.DALIDataType.INT16], index_paths=idxs, initial_fill=initial_fill,
                                           missing_component_behavior='error', num_shards=num_shards, pad_last_batch=True,
                                           random_shuffle=True, shard_id=shard_id, stick_to_shard=True,
                                           tensor_init_bytes=0, name='Reader', preserve=True)
    images = fn.decoders.image(images, device="mixed", device_memory_padding=0,
                               host_memory_padding=0, hybrid_huffman_threshold=0,
                               output_type=types.DALIImageType.RGB, preserve=True)
    labels = fn.copy(labels)
    labels = labels.gpu()
    images = fn.random_resized_crop(
        images, device='gpu', size=224, preserve=True)
    images = fn.crop_mirror_normalize(images, device='gpu',
                                      mean=[0.485 * 255, 0.456 *
                                            255, 0.406 * 255],
                                      std=[0.229 * 255, 0.224 * 255, 0.225 * 255], preserve=True)
    return images, labels


def run_one_step(model, criterion, optimizer, load_iter, scaler):
    try:
        data = next(load_iter)
    except StopIteration:
        data = next(load_iter)
    images = data[0]["data"]
    target = data[0]["label"].squeeze(-1).long()
    # compute output
    with torch.cuda.amp.autocast():
        output = model(images)
        loss = criterion(output, target)
    # compute gradient and do SGD step
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

def run_one_epoch(model, criterion, optimizer, load_iter, scaler):
    for data in load_iter:
        images = data[0]["data"]
        target = data[0]["label"].squeeze(-1).long()
        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def main():
    global args
    args = parse()

    args.local_rank = 0
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])

    if args.local_rank == 0:
        start_bytes = psutil.virtual_memory()._asdict()['used']
        print('env: Liquid, model: {}'.format(args.arch))

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    cudnn.benchmark = True

    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        dist.init_process_group(backend='nccl', init_method='env://')
        args.world_size = dist.get_world_size()

    torch.cuda.set_device(args.gpu)

    args.total_batch_size = args.world_size * args.batch_size

    # create model
    model = models.__dict__[args.arch]()

    model = model.cuda()

    args.lr = args.lr * float(args.total_batch_size) / 256.
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.distributed:
        model = DDP(model, device_ids=[
                    args.local_rank], output_device=args.local_rank)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    args.path = os.path.expanduser(args.path)
    # Data loading code
    dir_paths = glob(os.path.join(args.path, 'train_*'))
    if args.local_rank == 0:
        print(dir_paths)
    paths = []
    for dir_path in dir_paths:
        for tar in os.listdir(dir_path):
            if '.tar' in tar:
                paths.append(os.path.join(dir_path, tar))
    paths.sort()
    idxs = [path.replace('.tar', '.idx') for path in paths]
    buffer_ratio = args.buffer_size / (get_dataset_bytes(paths) // 1024)
    args.initial_fill = 1281167 * buffer_ratio // args.world_size

    pipe = create_dali_pipeline(batch_size=args.batch_size, num_threads=args.workers,
                                device_id=args.gpu, seed=12 + args.local_rank, paths=paths, idxs=idxs,
                                initial_fill=args.initial_fill, num_shards=args.world_size, shard_id=args.local_rank)
    pipe.build()
    loader = DALIClassificationIterator(pipe, reader_name="Reader", auto_reset=True,
                                        last_batch_policy=LastBatchPolicy.PARTIAL)

    scaler = torch.cuda.amp.GradScaler()
    model.train()

    run_one_epoch(model, criterion, optimizer, loader, scaler)
    load_iter = iter(loader)
    # warm up
    for _ in range(100):
        run_one_step(model, criterion, optimizer, load_iter, scaler)

    if args.distributed:
        dist.barrier()
    if args.local_rank == 0:
        start = time.perf_counter()

    for _ in range(200):
        run_one_step(model, criterion, optimizer, load_iter, scaler)

    if args.distributed:
        dist.barrier()
    if args.local_rank == 0:
        end = time.perf_counter()
        bytes_used = psutil.virtual_memory()._asdict()['used'] - start_bytes
        print(200 * args.total_batch_size / (end - start), bytes_used)


if __name__ == '__main__':
    main()
