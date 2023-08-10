from typing import Dict, List, Tuple, Union
import os
import random
import webdataset as wds
import numpy as np
import cv2 as cv
from multiprocessing import Process, Queue
from subprocess import call
from .converter import Converter
import nvidia.dali.types as types


class ImageNetConverter(Converter):
    num_images = {'train': 1281167, 'val': 50000, 'test': 100000}
    info = {
        'ext': ['image', 'label'],
        'dtypes': [types.UINT8, types.INT16],
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
        assert split in ['train', 'val', 'test'], 'No support for {}'.format(split)

        from_dir = os.path.join(self.from_path, split)
        assert os.path.isdir(from_dir), 'No directory {}'.format(from_dir)

        labels = os.listdir(from_dir)
        labels.sort()
        assert len(labels) == 1000, 'The number of classes is not 1000, checkout your directory'

        label_table = {}
        for i, label in enumerate(labels):
            label_table[label] = i

        image_paths = []
        for root, _, files in os.walk(from_dir):
            for file in files:
                image_paths.append(os.path.join(root, file))
        image_paths.sort()
        num_images = len(image_paths)
        assert num_images == self.num_images[split], 'The number of images is not correct, checkout your directory'

        random.shuffle(image_paths)

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
            processes.append(Process(target=self._reader, args=(queues[-1], self.formats, label_table,
                                    image_paths[i::self.num_readers], is_firsts[i::self.num_readers])))
            processes[-1].start()

        image_shapes = []
        label_shapes = []
        with wds.ShardWriter(pattern, encoder = False) as writer:
            for i in range(num_images):
                image, image_shape, label, label_shape = queues[i % self.num_readers].get()
                image_shapes.append(image_shape)
                label_shapes.append(label_shape)
                sample = {
                    '__key__': os.path.splitext(os.path.basename(image_paths[i]))[0],
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
        self._convert(pattern, 'val', percent)


    @staticmethod
    def _reader(queue: Queue, formats: Tuple[Union[str, None], Union[str, None]],
                label_table: Dict[str, int], image_paths: List[str], is_firsts: List[bool]) -> None:
        for image_path, is_first in zip(image_paths, is_firsts):
            label = label_table[os.path.split(os.path.split(image_path)[0])[1]].to_bytes(2, 'little')
            format = formats[0] if is_first else formats[1]

            with open(image_path, 'rb') as image_io:
                image = image_io.read()
                image_shape = (0, 0, 0)
                label_shape = (0, 0, 0)

                if format is not None:
                    image = np.fromstring(image, dtype = np.uint8)
                    image = cv.imdecode(image, cv.IMREAD_COLOR)
                    image_shape = image.shape

                    if format == 'raw':
                        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                    else:
                        _, image = cv.imencode('.' + format, image)
                        image_shape = (0, 0, 0)

                    image = image.tobytes()

                queue.put((image, image_shape, label, label_shape))
