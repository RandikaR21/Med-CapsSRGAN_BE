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


class Trainer:
    def __init__(self,
                 model,
                 loss,
                 learning_rate,
                 checkpoint_dir='./checkpoints/generator'):

        self.now = None
        self.loss = loss
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              psnr=tf.Variable(-1.0),
                                              optimizer=Adam(learning_rate),
                                              model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory=checkpoint_dir,
                                                             max_to_keep=3)

        self.restore()

    @property
    def model(self):
        return self.checkpoint.model

    def train(self, train_dataset, valid_dataset, steps, evaluate_every=1000, save_best_only=False):
        loss_mean = Mean()
        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint

        self.now = time.perf_counter()

        log_dir = 'logs/pre_generator_no_blurred/'
        tf.summary.trace_on(graph=True, profiler=True)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        current_time = "20220603-235607"
        train_log_dir = log_dir + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        for lr, hr in train_dataset.take(steps - ckpt.step.numpy()):
            ckpt.step.assign_add(1)
            step = ckpt.step.numpy()
            loss, sr_image = self.train_step(lr, hr)
            loss_mean(loss)

            sr_image = tf.clip_by_value(sr_image, 0, 255)
            sr_image = tf.round(sr_image)
            sr_image = tf.cast(sr_image, tf.uint8)

            if step % evaluate_every == 0:
                loss_value = loss_mean.result()
                loss_mean.reset_states()

                # Compute PSNR on validation dataset
                psnr_value = self.evaluate(valid_dataset)

                duration = time.perf_counter() - self.now
                print(
                    f'{step}/{steps}: loss = {loss_value.numpy():.3f}, '
                    f'PSNR = {psnr_value.numpy():3f} ({duration:.2f}s)')

                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss_value.numpy(), step=step)
                    tf.summary.image('Super-Resolved-Image', sr_image, step=step, max_outputs=10)
                    tf.summary.scalar('PSNR', psnr_value.numpy(), step=step)

                if save_best_only and psnr_value <= ckpt.psnr:
                    self.now = time.perf_counter()
                    # skip saving checkpoint, no PSNR improvement
                    continue

                ckpt.psnr = psnr_value
                ckpt_mgr.save()
                self.now = time.perf_counter()
        with train_summary_writer.as_default():
            tf.summary.trace_export(
                name="my_func_trace",
                step=0,
                profiler_outdir=log_dir)

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.checkpoint.model(lr, training=True)
            loss_value = self.loss(hr, sr)

        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))

        return loss_value, sr

    def evaluate(self, dataset):
        return evaluate(self.checkpoint.model, dataset)

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')


class SRResNet_Trainer(Trainer):
    def __init__(self,
                 model,
                 checkpoint_dir,
                 learning_rate=1e-4):
        super().__init__(model, loss=MeanSquaredError(), learning_rate=learning_rate, checkpoint_dir=checkpoint_dir)

    def train(self, train_dataset, valid_dataset, steps=1000000, evaluate_every=1000, save_best_only=True):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)

