from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('agg')
import tensorflow as tf
import pix2pix
import tensorflow_datasets as tfds
from tensorflow.keras import backend as K
tfds.disable_progress_bar()
from IPython.display import clear_output
import matplotlib.pyplot as plt
from PIL import Image
sys.modules['Image'] = Image
import foundations

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')



# hyper_params = {'batch_size': 16,
#                 'epochs': 20,
#                 'learning_rate': 0.001,
#                 'decoder_neurons': [128, 64, 32, 16]
#                 }
#
# foundations.log_params(hyper_params)

hyper_params = foundations.load_parameters()

# Define some job paramenters
TRAIN_LENGTH = 200
BATCH_SIZE = hyper_params['batch_size']
BUFFER_SIZE = 200
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
OUTPUT_CHANNELS = 3
EPOCHS = hyper_params['epochs']
VALIDATION_STEPS = 50//BATCH_SIZE



# Load the dataset
train_data = np.load('train_data.npz', allow_pickle=True)
train_images = train_data['images']
train_masks = train_data['masks']

# Create a generator in order to convert numpy data into tf.data
def create_generator(images, masks):
    def callable_generator():
        for image, mask in zip(images, masks):
            yield image, mask
    return callable_generator

train_dataset = tf.data.Dataset.from_generator(create_generator(train_images[:200], train_masks[:200]), (tf.float32, tf.float32), ((128, 128, 3), (128, 128, 1)))
test_dataset = tf.data.Dataset.from_generator(create_generator(train_images[200:], train_masks[200:]), (tf.float32, tf.float32), ((128, 128, 3), (128, 128, 1)))

# A function to normalize the images between 0 and 1. The mask is rescaled to values 0,1,2
def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


@tf.function
def load_image_train(input_image, input_mask):
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def load_image_test(input_image, input_mask):
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

train = train_dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)

test = test_dataset.map(load_image_test)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)


def display(display_list, name=None):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.savefig(f"sample_{name}.png")
    foundations.save_artifact(f"sample_{name}.png", key=f"sample_{name}")
    # plt.show()

for image, mask in train.take(1):
    sample_image, sample_mask = image, mask
display([sample_image, sample_mask])

with tf.name_scope("encoder"):
    base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False

with tf.name_scope("decoder"):
    up_stack = [
        pix2pix.upsample(hyper_params['decoder_neurons'][0], 3),  # 4x4 -> 8x8
        pix2pix.upsample(hyper_params['decoder_neurons'][1], 3),  # 8x8 -> 16x16
        pix2pix.upsample(hyper_params['decoder_neurons'][2], 3),  # 16x16 -> 32x32
        pix2pix.upsample(hyper_params['decoder_neurons'][3], 3),   # 32x32 -> 64x64
    ]

def unet_model(output_channels):

    with tf.name_scope("output"):
        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(
          output_channels, 3, strides=2,
          padding='same', activation='softmax')  #64x64 -> 128x128

    with tf.name_scope("input"):
        inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    x = inputs
    with tf.name_scope("encoder"):
        # Downsampling through the model
        skips = down_stack(x)
        x = skips[-1]
        skips = reversed(skips[:-1])

    with tf.name_scope("decoder"):
        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

model = unet_model(OUTPUT_CHANNELS)
opt = tf.keras.optimizers.Adam(lr=hyper_params['learning_rate'])
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None, num=1, name=None):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)], name=name)
    else:
        display([sample_image, sample_mask,
                 create_mask(model.predict(sample_image[tf.newaxis, ...]))], name=name)

try:
    show_predictions(name='initial')
except Exception as e:
    print(e)

callbacks = []

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions(name=f'epoch_{epoch+1}')
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
callbacks.append(DisplayCallback())

foundations.set_tensorboard_logdir('train_logs/')

tb = tf.keras.callbacks.TensorBoard(log_dir='tflogs', write_graph=True, write_grads=True, histogram_freq=1)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5, min_delta=0.0001,
                           verbose=1)
callbacks.append(tb)
callbacks.append(es)

rp = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=2,
                       verbose=1)
callbacks.append(rp)


model.summary()

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          callbacks=callbacks)

train_loss, train_acc = model_history.history['loss'], model_history.history['accuracy']
val_loss, val_acc = model_history.history['val_loss'], model_history.history['val_accuracy']

foundations.log_metric('train_loss', train_loss)
foundations.log_metric('train_accuracy', train_acc)

foundations.log_metric('val_loss', val_loss)
foundations.log_metric('val_accuracy', val_acc)

model.save('trained_model.h5')

foundations.save_artifact('trained_model.h5', key='trained_model')
