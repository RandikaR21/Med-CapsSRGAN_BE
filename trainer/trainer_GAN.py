import time
import tensorflow as tf
import datetime

from sr_model import evaluate
from sr_model import capsule_srgan

from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay


class SRGAN_Trainer:
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    current_time = "20220604-221622"
    train_log_dir = 'logs/GAN_Original_no_blur/' + current_time + '/train'
    test_log_dir = 'logs/GAN_Original_no_blur/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    def __init__(self,
                 generator,
                 discriminator,
                 content_loss='VGG54',
                 checkpoint_dir='./checkpoints/gan',
                 learning_rate=PiecewiseConstantDecay(boundaries=[100000], values=[1e-4, 1e-5])):

        if content_loss == 'VGG22':
            self.vgg = capsule_srgan.vgg_22()
        elif content_loss == 'VGG54':
            self.vgg = capsule_srgan.vgg_54()
        else:
            raise ValueError("content_loss must be either 'VGG22' or 'VGG54'")

        self.content_loss = content_loss
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = Adam(learning_rate=learning_rate)
        self.discriminator_optimizer = Adam(learning_rate=learning_rate)

        self.binary_cross_entropy = BinaryCrossentropy(from_logits=False)
        self.mean_squared_error = MeanSquaredError()

        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              perceptual_loss=tf.Variable(-1.0),
                                              optimizer=Adam(learning_rate),
                                              generator=generator,
                                              discriminator=discriminator)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory=checkpoint_dir,
                                                             max_to_keep=3)
        self.restore()

    @property
    def generator(self):
        return self.checkpoint.generator

    @property
    def discriminator(self):
        return self.checkpoint.discriminator

    def train(self, train_dataset, steps=200000):
        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint

        tf.summary.trace_on(graph=True, profiler=True)
        pls_metric = Mean()
        dls_metric = Mean()
        cls_metric = Mean()
        gls_metric = Mean()
        for lr, hr in train_dataset.take(steps - ckpt.step.numpy()):
            ckpt.step.assign_add(1)
            step = ckpt.step.numpy()

            pl, dl, cl, gl = self.train_step(lr, hr)
            pls_metric(pl)
            dls_metric(dl)
            cls_metric(cl)
            gls_metric(gl)
            with self.train_summary_writer.as_default():
                tf.summary.scalar('Content Loss', cls_metric.result(), step=step)
                tf.summary.scalar('Generator Loss', gls_metric.result(), step=step)
                tf.summary.scalar('Perceptual Loss', pls_metric.result(), step=step)
                tf.summary.scalar('Discriminator Loss', dls_metric.result(), step=step)

            if step % 50 == 0:
                print(
                    f'{step}/{steps}, perceptual loss = {pls_metric.result():.4f}, '
                    f'discriminator loss = {dls_metric.result():.4f}')
                pls_metric.reset_states()
                dls_metric.reset_states()

                ckpt.perceptual_loss = pls_metric.result()
                ckpt_mgr.save()
        with self.train_summary_writer.as_default():
            tf.summary.trace_export(
                name="my_func_trace",
                step=0,
                profiler_outdir='logs/gan_training/')

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.generator(lr, training=True)

            hr_output = self.discriminator(hr, training=True)
            sr_output = self.discriminator(sr, training=True)

            con_loss = self._content_loss(hr, sr)
            gen_loss = self._generator_loss(sr_output)
            perc_loss = con_loss + 0.001 * gen_loss
            disc_loss = self._discriminator_loss(hr_output, sr_output)

        gradients_of_generator = gen_tape.gradient(perc_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return perc_loss, disc_loss, con_loss, gen_loss

    @tf.function
    def _content_loss(self, hr, sr):
        sr = preprocess_input(tf.image.grayscale_to_rgb(sr))
        hr = preprocess_input(tf.image.grayscale_to_rgb(hr))
        sr_features = self.vgg(sr) / 12.75
        hr_features = self.vgg(hr) / 12.75
        return self.mean_squared_error(hr_features, sr_features)

    def _generator_loss(self, sr_out):
        return self.binary_cross_entropy(tf.ones_like(sr_out), sr_out)

    def _discriminator_loss(self, hr_out, sr_out):
        hr_loss = self.binary_cross_entropy(tf.ones_like(hr_out), hr_out)
        sr_loss = self.binary_cross_entropy(tf.zeros_like(sr_out), sr_out)
        return hr_loss + sr_loss

    @generator.setter
    def generator(self, value):
        self._generator = value

    @discriminator.setter
    def discriminator(self, value):
        self._discriminator = value

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')

