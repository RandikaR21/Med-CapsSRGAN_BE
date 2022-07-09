import tensorflow as tf
import os

directory = r'Dataset\\Processed\\Val_HR'
output_directory_HR = r'Dataset\\Processed\\Val_HR_Resized_PNG\\'
output_directory_LR = r'Dataset\\Processed\\Val_LR_PNG\\'
for filename in os.listdir(directory):
    image_open = open(directory + "\\" + filename, 'rb')
    read_image = image_open.read()
    scale = 0.25
    image_decode = tf.image.decode_jpeg(read_image)
    width = 256
    height = 256

    resize_image_lr = tf.image.resize(image_decode, [width, height], method="bicubic")
    resize_image_lr = tf.cast(resize_image_lr, tf.uint8)
    enc = tf.image.encode_jpeg(resize_image_lr)
    fileN = (os.path.splitext(filename)[0]) + ".png"
    fwrite_lr = tf.io.write_file(tf.constant(output_directory_LR + filename), enc)

    resize_image_Hr = tf.image.resize(image_decode, [width * 4, height * 4], method="bicubic")
    resize_image_Hr = tf.cast(resize_image_Hr, tf.uint8)
    enc = tf.image.encode_jpeg(resize_image_Hr)
    fwrite_hr = tf.io.write_file(tf.constant(output_directory_HR + filename), enc)
