from liquid.datasets import Cifar100Converter
import multiprocessing as mp


def main():
    num_cpus = mp.cpu_count()
    for i in range(0, 101, 10):
        Cifar100Converter('/datasets/cifar100', './pytorch-cifar/data/cifar-100-python',
                        num_readers=num_cpus, formats=('raw', 'png')).delete('train', i).convert('train', i).delete('test', i).convert('test', i)


if __name__ == '__main__':
    main()
