from typing import Any, Dict, List, Tuple, Union
import os
import random
import webdataset as wds
import numpy as np
import cv2 as cv
from multiprocessing import Process, Queue
from subprocess import call
from .converter import Converter
import nvidia.dali.types as types
import pickle

class Cifar100Converter(Converter):
    num_images = {'train': 50000, 'test': 10000}
    info = {
        'ext': ['image', 'label'],
        'dtypes': [types.UINT8, types.UINT8],
        'is_images': [True, False],
        'output_type': [types.RGB, None]
    }

    def _convert(self, pattern: str, split: str, percent: int) -> None:
        '''
        <from_path>/
            train/
                n01440764/
                    n01440764_10026.JPEG
                    n01440764_10027.JPEG
                    ...
                n01443537/
                ...
            val/
        '''
        assert split in ['train', 'test'], 'No support for {}'.format(split)

        from_file = os.path.join(self.from_path, split)
        assert os.path.isfile(from_file), 'No file {}'.format(from_file)

        with open(from_file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')

        numbers = list(range(self.num_images[split]))
        num_images = len(numbers)

        random.shuffle(numbers)

        is_firsts = []
        num_raw = 0
        for i in range(num_images):
            if 100 * num_raw < percent * (i + 1):
                num_raw += 1
                is_firsts.append(True)
            else:
                is_firsts.append(False)

        queues = []
        processes = []
        for i in range(self.num_readers):
            queues.append(Queue(2))
            processes.append(Process(target=self._reader, args=(queues[-1], self.formats, dict,
                                    numbers[i::self.num_readers], is_firsts[i::self.num_readers])))
            processes[-1].start()

        image_shapes = []
        label_shapes = []
        with wds.ShardWriter(pattern, encoder = False) as writer:
            for i in range(num_images):
                image, image_shape, label, label_shape = queues[i % self.num_readers].get()
                image_shapes.append(image_shape)
                label_shapes.append(label_shape)
                sample = {
                    '__key__': str(numbers[i]),
                    'image': image,
                    'label': label
                }
                writer.write(sample)

        for process in processes:
            process.join()

        i, acc = 0, 0
        while os.path.exists(pattern % i):
            tar_path = pattern % i
            idx_path = os.path.splitext(tar_path)[0] + '.idx'

            call(['wds2idx', tar_path, idx_path])

            with open(idx_path, 'r') as f:
                lines = f.readlines()

            for num in range(acc, acc + len(lines) - 1):
                param = lines[num - acc + 1].split(' ')
                param[-1] = param[-1].rstrip('\n')
                param.insert(0, str(int(is_firsts[num])))
                param.insert(5, str(image_shapes[num][0]))
                param.insert(6, str(image_shapes[num][1]))
                param.insert(7, str(image_shapes[num][2]))
                param.append(str(label_shapes[num][0]))
                param.append(str(label_shapes[num][1]))
                param.append(str(label_shapes[num][2]) + '\n')
                lines[num - acc + 1] = ' '.join(param)

            with open(idx_path, 'w') as f:
                f.writelines(lines)

            i += 1
            acc += len(lines) - 1


    def _convert_part(self, pattern: str, percent: int) -> None:
        self._convert(pattern, 'train', percent)


    @staticmethod
    def _reader(queue: Queue, formats: Tuple[Union[str, None], Union[str, None]],
                dict: Dict[str, Any], numbers: List[int], is_firsts: List[bool]) -> None:
        for number, is_first in zip(numbers, is_firsts):
            data = dict[b'data'][number].reshape(3, 32, 32)
            image = np.empty((32, 32, 3), dtype='uint8')
            for i in range(3):
                image[:, :, i] = data[i, :, :]
            label = dict[b'fine_labels'][number].to_bytes(1, 'little')
            format = formats[0] if is_first else formats[1]

            image_shape = (32, 32, 3)
            label_shape = (0, 0, 0)

            if (format is not None) and (format != 'raw'):
                image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
                _, image = cv.imencode('.' + format, image)
                image_shape = (0, 0, 0)

            image = image.tobytes()
            queue.put((image, image_shape, label, label_shape))
