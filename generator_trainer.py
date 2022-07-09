import os
from load_dataset import MedicalDataSet
from sr_model.capsule_srgan import generator
from trainer.train import SRResNet_Trainer

train_loader_medical = MedicalDataSet(scale=4, downgrade='bicubic', subset='train')

# Create a tf.data.Dataset
train_ds = train_loader_medical.dataset(batch_size=16, random_transform=True, repeat_count=None)

valid_loader = MedicalDataSet(scale=4, downgrade='bicubic', subset='valid')

# Create a tf.data.Dataset
valid_ds = valid_loader.dataset(batch_size=1, random_transform=False, repeat_count=1)

# Create a training context for the generator (SRResNet) alone.
pre_trainer = SRResNet_Trainer(model=generator(with_batch_norm=True), checkpoint_dir=f'checkpoints/Final/'
                                                                                   f'Pre_Generator_ALL_BLur/')

# Pre-train the generator with 1,000,000 steps (100,000 works fine too).
pre_trainer.train(train_ds, valid_ds.take(10), steps=60000, evaluate_every=1000)

# Save weights of pre-trained generator (needed for fine-tuning with GAN).
model_dir = "saved_model_weights/Final/"
os.makedirs(model_dir, exist_ok=True)
pre_trainer.model.save_weights(model_dir+"/pre_generator_all_blurred_60000.h5")
