from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import numpy as np
import matplotlib

matplotlib.use('Agg')
import tensorflow as tf
import pix2pix
import tensorflow_datasets as tfds
from tensorflow.keras.losses import sparse_categorical_crossentropy

tfds.disable_progress_bar()
import matplotlib.pyplot as plt
from PIL import Image

sys.modules['Image'] = Image

# import foundations here

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

print("getting hyper parameters for the job")
# define hyperparameters: Replace hyper_params by foundations.load_parameters()
hyper_params = {'batch_size': 16,
                'epochs': 10,
                'learning_rate': 0.0001,
                'decoder_neurons': [128, 64, 32, 16]
                }

# Define some job paramenters
TRAIN_LENGTH = 200
TEST_LENGTH = 50
BATCH_SIZE = hyper_params['batch_size']
BUFFER_SIZE = 200
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
OUTPUT_CHANNELS = 3
EPOCHS = hyper_params['epochs']
VALIDATION_STEPS = 50 // BATCH_SIZE

# Define summary writers for Tensorboard
train_log_dir = 'tflogs/gradient_tape/' + '/train'
test_log_dir = 'tflogs/gradient_tape/' + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)
tf.summary.experimental.set_step(1)


print("loading the dataset for the job")
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


train_dataset = tf.data.Dataset.from_generator(create_generator(train_images[:200], train_masks[:200]),
                                               (tf.float32, tf.float32), ((128, 128, 3), (128, 128, 1)))
test_dataset = tf.data.Dataset.from_generator(create_generator(train_images[200:], train_masks[200:]),
                                              (tf.float32, tf.float32), ((128, 128, 3), (128, 128, 1)))


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


# Add foundations artifact below plt.savefig i.e. foundations.save_artifact(f"sample_{name}.png", key=f"sample_{name}")
def display(display_list, name=None):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.savefig(f"sample_{name}.png")
    # plt.show()


for image, mask in train.take(1):
    sample_image, sample_mask = image, mask
display([sample_image, sample_mask], name='original')

with tf.name_scope("encoder"):
    base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',  # 64x64
    'block_3_expand_relu',  # 32x32
    'block_6_expand_relu',  # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',  # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False

with tf.name_scope("decoder"):
    up_stack = [
        pix2pix.upsample(hyper_params['decoder_neurons'][0], 3, name='conv2d_transpose_4x4_to_8x8'),  # 4x4 -> 8x8
        pix2pix.upsample(hyper_params['decoder_neurons'][1], 3, name='conv2d_transpose_8x8_to_16x16'),  # 8x8 -> 16x16
        pix2pix.upsample(hyper_params['decoder_neurons'][2], 3, name='conv2d_transpose_16x16_to_32x32'),  # 16x16 -> 32x32
        pix2pix.upsample(hyper_params['decoder_neurons'][3], 3, name='conv2d_transpose_32x32_to_64x64'),  # 32x32 -> 64x64
    ]


def unet_model(output_channels):
    with tf.name_scope("output"):
        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(
            output_channels, 3, strides=2,
            padding='same', activation='softmax', name='conv2d_transpose_64x64_to_128x128')  # 64x64 -> 128x128

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
            # Hint: Is something 'skipped'?

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
        show_predictions(name=f'epoch_{epoch + 1}')
        print('\nSample Prediction after epoch {}\n'.format(epoch + 1))


callbacks.append(DisplayCallback())

# Add tensorboard dir for foundations here  i.e. foundations.set_tensorboard_logdir('tflogs')


# tb = tf.keras.callbacks.TensorBoard(log_dir='tflogs', write_graph=True, write_grads=True, histogram_freq=1)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5, min_delta=0.0001,
                                      verbose=1)
#callbacks.append(tb)
callbacks.append(es)

rp = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=2,
                                          verbose=1)
callbacks.append(rp)

model.summary()


# tf 2.0 GradientTape and tracking gradients for Tensorboard

def train_with_gradient_tape(train_dataset, validation_dataset, model, epochs, callbacks):
    # Iterate over epochs.
    train_loss_results = []
    train_accuracy_results = []
    validation_loss_results = []
    validation_accuracy_results = []

    for epoch in range(epochs):
        print('Start of epoch %d' % (epoch,))
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        max_train_step = float(TRAIN_LENGTH) / BATCH_SIZE

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train)  # Logits for this minibatch
                loss_value = sparse_categorical_crossentropy(y_batch_train, logits)

            grads = tape.gradient(loss_value, model.trainable_weights)

            for grad, trainable_variable in zip(grads, model.trainable_variables):
                with train_summary_writer.as_default():
                    tf.summary.histogram(f'grad_{trainable_variable.name}', grad)

            opt.apply_gradients(zip(grads, model.trainable_weights))

            # Track progress
            epoch_loss_avg(loss_value)  # Add current batch loss

            # Compare predicted label to actual label
            epoch_accuracy(y_batch_train, model(x_batch_train))

            if step > max_train_step:
                break

        # End epoch and track train loss and accuracy
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())
        with train_summary_writer.as_default():
            tf.summary.scalar('training_loss', epoch_loss_avg.result())
            tf.summary.scalar('training_acc', epoch_accuracy.result())

        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # track validation loss and accuracy after each epoch
        max_eval_step = float(TEST_LENGTH) / BATCH_SIZE
        for step, (x_batch, y_batch) in enumerate(validation_dataset):
            logits = model(x_batch)
            epoch_accuracy(y_batch, logits)
            epoch_loss = sparse_categorical_crossentropy(y_batch, logits)
            epoch_loss_avg(epoch_loss)
            if step > max_eval_step:
                break

        validation_loss_results.append(epoch_loss_avg.result())
        validation_accuracy_results.append(epoch_accuracy.result())
        with test_summary_writer.as_default():
            tf.summary.scalar('validation_loss', epoch_loss_avg.result())
            tf.summary.scalar('validation_acc', epoch_accuracy.result())

        # use existing callbacks
        show_predictions(name=f'epoch_{epoch + 1}')

    return train_loss_results, train_accuracy_results, validation_loss_results, validation_accuracy_results


# optional: comment to use keras API
train_loss_results, train_accuracy_results, validation_loss_results, validation_accuracy_results = train_with_gradient_tape(train_dataset, test_dataset, model, EPOCHS, callbacks)
train_acc = train_accuracy_results[-1]
val_acc = validation_accuracy_results[-1]
train_loss = train_loss_results[-1]
validation_loss = validation_loss_results[-1]
print(f'train loss: {train_loss}, train accuracy: {train_acc},'
      f' validation loss: {validation_loss}, validation accuracy: {val_acc}')


# optional: uncomment to use keras API without tracking the gradients as an alternative
# model_history = model.fit(train_dataset, epochs=EPOCHS,
#                          steps_per_epoch=STEPS_PER_EPOCH,
#                          validation_steps=VALIDATION_STEPS,
#                          validation_data=test_dataset,
#                          callbacks=callbacks)
# train_acc = model_history.history['accuracy'][-1]
# val_acc = model_history.history['val_accuracy'][-1]


model.save("trained_model.h5")

# Add foundations log_metrics here


# Add foundations save_artifacts here to save the trained model
