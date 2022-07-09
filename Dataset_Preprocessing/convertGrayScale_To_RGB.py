import os
from PIL import Image
import shutil


def grayscale_to_rgb():
    input_directory = "Dataset/Testing/Final_Results/ALL_BLUR/SRGAN_Results"
    output_directory = input_directory + "_RGB/"
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.mkdir(output_directory)

    for filename in os.listdir(input_directory):
        image = Image.open(os.path.join(input_directory, filename))
        rgb_image = Image.new("RGB", image.size)
        rgb_image.paste(image)
        output_path = os.path.join(output_directory, filename)
        rgb_image.save(output_path)


def main():
    grayscale_to_rgb()


if __name__ == "__main__":
    main()
