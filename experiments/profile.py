from liquid.profile import Profiler
from liquid.datasets import ImageNetConverter, CityscapesConverter, CityPersonsConverter
from liquid.profile import Liquid_profile_load, Liquid_profile_prep
import multiprocessing as mp
import torch
import argparse
import os


BATCH_SIZE = 128  # per GPU
IMG_SIZE_0 = 145706350  # KB
IMG_SIZE_100 = 824851660  # KB
CITY_SIZE_0 = 106227340  # KB
CITY_SIZE_100 = 376436360  # KB
PERSONS_SIZE_0 = 105598650  # KB
PERSONS_SIZE_100 = 282451740  # KB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    # parser.add_argument('--i', type=int)
    args = parser.parse_args()
    args.path = os.path.expanduser(args.path)
    img_path = os.path.join(args.path, 'imagenet')
    city_path = os.path.join(args.path, 'cityscapes')
    persons_path = os.path.join(args.path, 'citypersons')

    num_cpus = mp.cpu_count()
    num_gpus = torch.cuda.device_count()
    num_threads = num_cpus // num_gpus

    # img_cvtr = ImageNetConverter(
    #     img_path, img_path, num_readers=num_cpus)
    # optimal = Profiler(img_cvtr, num_gpus).profile(BATCH_SIZE, IMG_SIZE_0,
    #                                                IMG_SIZE_100, num_threads, 43725619)
    # img_cvtr.convert('train', 10)
    # img_cvtr.convert('train', 0)

    # city_cvtr = CityscapesConverter(
    #     city_path, city_path, num_readers=num_cpus)
    # optimal = Profiler(city_cvtr, num_gpus).profile(BATCH_SIZE, CITY_SIZE_0,
    #                                                 CITY_SIZE_100, num_threads, 31771853)
    # city_cvtr.convert('train', 30).convert('train_extra', 30).convert(
    #     'train_right', 30).convert('train_extra_right', 30)
    # city_cvtr.convert('train', 0).convert('train_extra', 0).convert(
    #     'train_right', 0).convert('train_extra_right', 0)

    persons_cvtr = CityPersonsConverter(
        persons_path, city_path, num_readers=num_cpus)
    prof = Profiler(persons_cvtr, num_gpus)
    # for i in range(11):
    # prof.profile(BATCH_SIZE, PERSONS_SIZE_0, PERSONS_SIZE_100, num_threads, 194025195 * args.i // 10)
    # persons_cvtr.convert('train', 30).convert('train_extra', 30).convert(
    #     'train_right', 30).convert('train_extra_right', 30)
    persons_cvtr.convert('train', 70).convert('train_extra', 70).convert(
        'train_right', 70).convert('train_extra_right', 70)


if __name__ == '__main__':
    main()
