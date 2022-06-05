# import tensorflow as tf
import os
from load_dataset import MedicalDataSet
from sr_model.capsule_srgan import generator, capsule_discriminator, original_discriminator
from trainer.trainer_GAN import SRGAN_Trainer

# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

train_loader_medical = MedicalDataSet(scale=4, downgrade='bicubic', subset='train')

# Create a tf.data.Dataset
train_ds = train_loader_medical.dataset(batch_size=16, random_transform=True, repeat_count=None)

valid_loader = MedicalDataSet(scale=4, downgrade='bicubic', subset='valid')

# Create a tf.data.Dataset
valid_ds = valid_loader.dataset(batch_size=1, random_transform=False, repeat_count=1)

# Create a new generator and init it with pre-trained weights.
gan_generator = generator()
gan_generator.load_weights('saved_model_weights/srgan_no_blurred/pre_generator_no_blurred.h5')

# Create a training context for the GAN (generator + discriminator).
gan_trainer = SRGAN_Trainer(generator=gan_generator, discriminator=original_discriminator(), checkpoint_dir=f'checkpoints/Original_GAN_no_blur/')

# Train the GAN with 20,000 steps.
gan_trainer.train(train_ds, steps=20000)

# Save weights of generator and discriminator.
model_dir = "saved_model_weights/Original_srgan/Original_gan_no_blur"
os.makedirs(model_dir, exist_ok=True)
gan_trainer.generator.save_weights(model_dir+'/Original_gan_generator.h5')
gan_trainer.discriminator.save_weights(model_dir+'/Original_'
                                                 'gan_discriminator.h5')

