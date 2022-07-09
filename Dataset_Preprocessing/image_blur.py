import os
from PIL import Image, ImageFilter
import shutil
import random

input_directory = "Dataset/Testing/LR - No Blur"


def blur_all():
    output_directory = "Dataset/Testing/LR_All-Blur"
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.mkdir(output_directory)
    for filename in os.listdir(input_directory):
        image = Image.open(os.path.join(input_directory, filename))

        # GaussianBlur predefined kernel argument
        image = image.filter(ImageFilter.GaussianBlur(radius=1))
        output_path = os.path.join(output_directory, filename)
        image.save(output_path)


def random_blur():
    output_directory = "Dataset/Final/Chest X Ray - Random Blur/images/valid_LR_bicubic/X4"
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.mkdir(output_directory)
    # Blur random images in the dataset
    for filename in os.listdir(input_directory):
        image = Image.open(os.path.join(input_directory, filename))

        if bool(random.getrandbits(1)):
            # GaussianBlur predefined kernel argument
            image = image.filter(ImageFilter.GaussianBlur(radius=1))
            print(filename)

        output_path = os.path.join(output_directory, filename)
        image.save(output_path)


def main():
    blur_all()
    # random_blur()


if __name__ == "__main__":
    main()
