from liquid.datasets.utils import png2others

def main():
    png2others('/datasets/cityscapes_bmp', 128, 'bmp')
    png2others('/datasets/cityscapes_jpeg', 128, 'jpeg')


if __name__ == '__main__':
    main()
