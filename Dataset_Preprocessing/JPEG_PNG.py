from PIL import Image
import os

directory = r'Dataset\Final\Chest X Ray\images\valid_LR_bicubic\X4'
output_directory = r'Dataset\Final\Chest X Ray\images2\valid_LR_bicubic\X4'

for filename in os.listdir(directory):
    im1 = Image.open(directory + "\\" + filename)
    newFile = (os.path.splitext(filename)[0]) + ".png"
    im1.save(output_directory + "\\" + newFile)
