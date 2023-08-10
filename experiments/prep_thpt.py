import os
import argparse
import multiprocessing as mp
from liquid.profile import Liquid_profile_load, Liquid_profile_prep, CoorDL_profile_load, CoorDL_profile_prep, DALI_profile_load, DALI_profile_prep
from liquid.datasets import ImageNetConverter, CityscapesConverter
from glob import glob

BATCH_SIZE = 128
NUM_WARMUPS = 100
NUM_STEPS = 200


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str,
                        choices=['Liquid', 'CoorDL', 'DALI'])
    parser.add_argument('--kind', type=str, choices=['imagenet', 'cityscapes'])
    parser.add_argument('--mode', type=str, choices=['load', 'prep'])
    parser.add_argument('--path', type=str)
    parser.add_argument('--buffer_size', type=int)
    args = parser.parse_args()
    return args


def main():
    num_cpus = mp.cpu_count()
    args = parse_args()
    args.path = os.path.expanduser(args.path)

    local_rank = int(os.environ['LOCAL_RANK']
                     ) if 'LOCAL_RANK' in os.environ else 0
    world_size = int(os.environ['WORLD_SIZE']
                     ) if 'WORLD_SIZE' in os.environ else 1

    if local_rank == 0:
        print('env: {}, kind: {}'.format(args.env, args.kind))

    info = ImageNetConverter.info if args.kind == 'imagenet' else CityscapesConverter.info
    if args.env == 'Liquid':
        dir_paths = glob(os.path.join(args.path, 'imagenet/train_*')) if args.kind == 'imagenet' else \
                    glob(os.path.join(args.path, 'cityscapes/train_*'))
    else:
        dir_paths = [os.path.join(args.path, 'imagenet/train')] if args.kind == 'imagenet' else \
                    [os.path.join(args.path, 'cityscapes_jpeg/leftImg8bit/train'),
                     os.path.join(args.path, 'cityscapes_jpeg/leftImg8bit/train_extra'),
                     os.path.join(args.path, 'cityscapes_jpeg/rightImg8bit/train'),
                     os.path.join(args.path, 'cityscapes_jpeg/rightImg8bit/train_extra')]
    dir_paths = [os.path.expanduser(path) for path in dir_paths]
    dataset_size = 0
    if args.env == 'Liquid':
        for path in dir_paths:
            for tar in os.listdir(path):
                if '.tar' in tar:
                    dataset_size += os.path.getsize(os.path.join(path, tar))
    else:
        for path in dir_paths:
            for root, _, files in os.walk(path):
                for file in files:
                    dataset_size += os.path.getsize(os.path.join(root, file))
    buffer_ratio = args.buffer_size / (dataset_size // 1024)
    num_threads = num_cpus // world_size

    if args.env == 'Liquid':
        if args.mode == 'load':
            load_thpt = Liquid_profile_load(local_rank, BATCH_SIZE, num_threads, buffer_ratio,
                                            world_size, NUM_WARMUPS, NUM_STEPS, dir_paths, info, 'env://')
            if local_rank == 0:
                print('load thpt: {}'.format(load_thpt))
        else:
            decode_thpt, prep_thpt = Liquid_profile_prep(
                local_rank, BATCH_SIZE, num_threads, buffer_ratio, world_size, NUM_WARMUPS, NUM_STEPS, dir_paths, info, 'env://')
            if local_rank == 0:
                print('decode thpt: {}, prep thpt: {}'.format(
                    decode_thpt, prep_thpt))
    elif args.env == 'CoorDL':
        if args.mode == 'load':
            load_thpt = CoorDL_profile_load(local_rank, BATCH_SIZE, num_threads, buffer_ratio,
                                            world_size, NUM_WARMUPS, NUM_STEPS, dir_paths, 'env://', args.kind)
            if local_rank == 0:
                print('load thpt: {}'.format(load_thpt))
        else:
            decode_thpt, prep_thpt = CoorDL_profile_prep(
                local_rank, BATCH_SIZE, num_threads, buffer_ratio, world_size, NUM_WARMUPS, NUM_STEPS, dir_paths, 'env://', args.kind)
            if local_rank == 0:
                print('decode thpt: {}, prep thpt: {}'.format(
                    decode_thpt, prep_thpt))
    else:
        if args.mode == 'load':
            load_thpt = DALI_profile_load(local_rank, BATCH_SIZE, num_threads,
                                          world_size, NUM_WARMUPS, NUM_STEPS, dir_paths, 'env://', args.kind)
            if local_rank == 0:
                print('load thpt: {}'.format(load_thpt))
        else:
            decode_thpt, prep_thpt = DALI_profile_prep(
                local_rank, BATCH_SIZE, num_threads, world_size, NUM_WARMUPS, NUM_STEPS, dir_paths, 'env://', args.kind)
            if local_rank == 0:
                print('decode thpt: {}, prep thpt: {}'.format(
                    decode_thpt, prep_thpt))


if __name__ == '__main__':
    main()
