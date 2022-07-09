import os.path

from sr_model.capsule_srgan import generator
from trainer.utils import load_image, plot_sample
from sr_model import resolve_single
import tensorflow as tf
import datetime
import numpy as np
import csv

subset = "RANDOM_BLUR"


def run_test(generator_model, output_name_suffix, lr, hr, output_dir):
    # Load LR image
    lr_image = load_image(lr)
    # Start timer
    starting_time = datetime.datetime.now()
    # Generate SR image
    sr = resolve_single(generator_model, lr_image)
    # End timer
    end = datetime.datetime.now()
    # Calculate time
    time_taken = str((end - starting_time).total_seconds())

    # Calculate PSNR
    ground_truth = load_image(hr)
    expand_ground_truth = ground_truth[:, :, np.newaxis]
    psnr_value = tf.image.psnr(tf.expand_dims(expand_ground_truth, axis=0), sr, max_val=255)[0]
    # Save image to destination
    src_png = tf.image.encode_png(sr)
    split_filename = os.path.splitext(os.path.basename(lr))
    output_name = split_filename[0] + output_name_suffix + split_filename[1]
    output_path = os.path.join(output_dir, output_name)
    tf.io.write_file(tf.constant(output_path), src_png)

    return time_taken, psnr_value

allfieldnames = ['file_name', 'CapsSRGAN_NoBlur_PSNR', 'CapsSRGAN_NoBlur_RunTime', 'CapsSRGAN_RandomBlur_PSNR',
                 'CapsSRGAN_RandomBlur_RunTime', 'OriginalSRGAN_NoBlur_PSNR', 'OriginalSRGAN_NoBlur_RunTime',
                 'OriginalSRGAN_RandomBlur_PSNR', 'OriginalSRGAN_RandomBlur_RunTime', 'CapsSRGAN_AllBlur_PSNR',
                 'CapsSRGAN_AllBlur_RunTime', 'OriginalSRGAN_AllBlur_PSNR', 'OriginalSRGAN_AllBlur_RunTime']

fieldnames = ['file_name', 'CapsSRGAN_PSNR', 'CapsSRGAN_RunTime', 'OriginalSRGAN_PSNR', 'OriginalSRGAN_RunTime']

rows = []

caps_no_blur_generator_model = generator()
caps_no_blur_generator_model.load_weights('saved_model_weights/capsule_srgan/caps_gan_no_blur/caps_gan_generator.h5')

caps_random_blur_generator_model = generator()
caps_random_blur_generator_model.load_weights(
    'saved_model_weights/Final/CapSRGAN/All_Blur/Generator.h5')

caps_all_blur_generator_model = generator()
caps_all_blur_generator_model.load_weights(
    'saved_model_weights/capsule_srgan/caps_gan_random_blur/caps_gan_generator.h5')

original_no_blur_generator_model = generator()
original_no_blur_generator_model.load_weights(
    'saved_model_weights/Original_srgan/Original_gan_no_blur/Original_gan_generator.h5')

original_random_blur_generator_model = generator()
original_random_blur_generator_model.load_weights(
    'saved_model_weights/srgan/original_gan_random_blur/gan_generator_original.h5')
#
original_all_blur_generator_model = generator()
original_all_blur_generator_model.load_weights(
    'saved_model_weights/Final/CapSRGAN/All_Blur/Generator.h5')


def test_all():
    reference_dir = 'Dataset/Testing/Ground Truth/'
    input_dir = 'Dataset/Testing/LR - With Blur/'
    output_dir = 'Dataset/Testing/Results_With_Blur'
    for file in os.listdir(input_dir):
        input_image = os.path.join(input_dir, file)
        reference_image = os.path.join(reference_dir, file)

        # Test run
        run_test(caps_no_blur_generator_model, "_CapsSRGAN_NoBlur", input_image, reference_image, output_dir)

        ''' Capsule SRGAN - No Blur Training'''
        caps_srgan_no_blur_run_time, caps_srgan_no_blur_psnr = run_test(caps_no_blur_generator_model,
                                                                        "_CapsSRGAN_NoBlur", input_image,
                                                                        reference_image, output_dir)

        caps_srgan_no_blur_psnr = f'{caps_srgan_no_blur_psnr.numpy():3f}'
        print("Time taken for Capsule SRGAN - No Blur model :  ", caps_srgan_no_blur_run_time)
        print(f'PSNR = {caps_srgan_no_blur_psnr}')

        ''' Capsule SRGAN - Random Blur Training'''
        caps_srgan_random_blur_run_time, caps_srgan_random_blur_psnr = run_test(caps_random_blur_generator_model,
                                                                                "_CapsSRGAN_RandomBlur", input_image,
                                                                                reference_image, output_dir)

        caps_srgan_random_blur_psnr = f'{caps_srgan_random_blur_psnr.numpy():3f}'
        print("Time taken for Capsule SRGAN - Random Blur model :  ", caps_srgan_random_blur_run_time)
        print(f'PSNR = {caps_srgan_random_blur_psnr}')

        ''' Original SRGAN - No Blur Training'''

        original_srgan_no_blur_run_time, original_srgan_no_blur_psnr = run_test(original_no_blur_generator_model,
                                                                                "_OriginalSRGAN_NoBlur", input_image,
                                                                                reference_image, output_dir)
        original_srgan_no_blur_psnr = f'{original_srgan_no_blur_psnr.numpy():3f}'
        print("Time taken for Original SRGAN - No Blur model :  ", original_srgan_no_blur_run_time)
        print(f'PSNR = {original_srgan_no_blur_psnr}')

        ''' Original SRGAN - Random Blur Training'''
        original_srgan_random_blur_run_time, original_srgan_random_blur_psnr = run_test(
            original_random_blur_generator_model,
            "_OriginalSRGAN_RandomBlur", input_image, reference_image, output_dir)
        original_srgan_random_blur_psnr = f'{original_srgan_random_blur_psnr.numpy():3f}'
        print("Time taken for Original SRGAN - Random Blur model :  ", original_srgan_random_blur_run_time)
        print(f'PSNR = {original_srgan_random_blur_psnr}')

        rows.append({
            'file_name': input_image,
            'CapsSRGAN_NoBlur_PSNR': caps_srgan_no_blur_psnr,
            'CapsSRGAN_NoBlur_RunTime': caps_srgan_no_blur_run_time,
            'CapsSRGAN_RandomBlur_PSNR': caps_srgan_random_blur_psnr,
            'CapsSRGAN_RandomBlur_RunTime': caps_srgan_random_blur_run_time,
            'OriginalSRGAN_NoBlur_PSNR': original_srgan_no_blur_psnr,
            'OriginalSRGAN_NoBlur_RunTime': original_srgan_no_blur_run_time,
            'OriginalSRGAN_RandomBlur_PSNR': original_srgan_random_blur_psnr,
            'OriginalSRGAN_RandomBlur_RunTime': original_srgan_random_blur_run_time
        })

    # plot_sample(lr, sr)

    # Generate CSV Report
    with open('test_results_with_blur.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=allfieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_model():
    reference_dir = 'Dataset/Testing/Ground Truth/'
    input_dir = 'Dataset/Testing/LR_' + subset + '/'
    output_dir = 'Dataset/Testing/Final_Results/' + subset

    for file in os.listdir(input_dir):
        input_image = os.path.join(input_dir, file)
        reference_image = os.path.join(reference_dir, file)

        ''' Capsule SRGAN '''
        capsule_run_time, psnr = run_test(caps_random_blur_generator_model,
                                          "", input_image,
                                          reference_image,
                                          (output_dir + "/Med-CapSRGAN_Results/"))

        capsule_psnr = f'{psnr.numpy():3f}'
        print("Time taken for Capsule SRGAN model :  ", capsule_run_time)
        print(f'PSNR = {capsule_psnr}')

        ''' Original SRGAN '''
        original_run_time, psnr = run_test(
            original_random_blur_generator_model,
            "", input_image, reference_image, (output_dir + "/SRGAN_Results/"))
        original_psnr = f'{psnr.numpy():3f}'
        print("Time taken for Original SRGAN  model :  ", original_run_time)
        print(f'PSNR = {original_psnr}')

        rows.append({
            'file_name': input_image,
            'CapsSRGAN_PSNR': capsule_psnr,
            'CapsSRGAN_RunTime': capsule_run_time,
            'OriginalSRGAN_PSNR': original_psnr,
            'OriginalSRGAN_RunTime': original_run_time
        })

        # Generate CSV Report
        with open('test_results_' + subset + '.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def test_one(model):
    input_image = 'Demo/1.png'
    reference_image = 'Demo/Caps_Enhanced_1.png'
    output_dir = 'Demo'
    run_time, psnr = run_test(
        model,
        "_CapsEnhanced", input_image, reference_image, output_dir)
    psnr = f'{psnr.numpy():3f}'
    print("Time taken :  ", run_time)
    print(f'PSNR = {psnr}')


test_model()
