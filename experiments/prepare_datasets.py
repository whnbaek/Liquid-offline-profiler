# from liquid.datasets.utils import cityPersons2COCO, extract_imagenet_train_tar
from liquid.datasets import CityPersonsConverter
import multiprocessing as mp


def main():
    num_cpus = mp.cpu_count()
    # extract_imagenet_train_tar('~/datasets/imagenet', num_workers=num_cpus)
    # cityPersons2COCO('~/datasets/cityscapes/citypersons.json', '~/datasets/cityscapes',
    #                  ['train', 'train_extra', 'train_right', 'train_extra_right'])


if __name__ == '__main__':
    main()
