import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import numpy as np
import matplotlib.pyplot as plt

# 1. Define the U-Net Model for Multi-Class Segmentation
def build_unet(input_size=(473, 473, 3), num_classes=20):
    inputs = layers.Input(input_size)
    
    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)
    
    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    
    # Decoder
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    
    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    
    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c9)
    
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# 2. Define Data Paths
# Replace 'path_to_LIP_dataset' with the actual path where the LIP dataset is stored
dataset_path = 'dataset/path_to_LIP_dataset'  # e.g., '/home/user/LIP'
train_images_dir = os.path.join(dataset_path, 'Train/Image')
train_masks_dir = os.path.join(dataset_path, 'Train/Mask')
val_images_dir = os.path.join(dataset_path, 'Val/Image')
val_masks_dir = os.path.join(dataset_path, 'Val/Mask')

# 3. Define Parameters
input_size = (473, 473, 3)  # Original LIP image size
num_classes = 20  # 19 classes + background
batch_size = 8
epochs = 50
buffer_size = 1000

# 4. Data Preprocessing Functions
def parse_image(image_path, mask_path):
    # Read and decode the image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [input_size[0], input_size[1]])
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]
    
    # Read and decode the mask
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [input_size[0], input_size[1]], method='nearest')
    mask = tf.cast(mask, tf.int32)
    mask = tf.squeeze(mask, axis=-1)  # Shape: (473, 473)
    
    return image, mask

def load_dataset(image_dir, mask_dir):
    image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(('.jpg', '.png'))])
    mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir) if fname.endswith(('.png', '.jpg'))])
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

# 5. Create Training and Validation Datasets
train_dataset = load_dataset(train_images_dir, train_masks_dir)
train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_dataset = load_dataset(val_images_dir, val_masks_dir)
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# 6. Define Data Augmentation (Optional but Recommended)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

def augment(image, mask):
    augmented = data_augmentation(image)
    return augmented, mask

train_dataset = train_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

# 7. Instantiate and Compile the Model
model = build_unet(input_size=input_size, num_classes=num_classes)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

model.summary()

# 8. Define Callbacks
checkpoint_path = 'unet_lip_best_model.h5'
callbacks = [
    ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
]

# 9. Train the Model
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=val_dataset,
    callbacks=callbacks
)

# 10. Plot Training History (Optional)
plt.figure(figsize=(12, 4))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['sparse_categorical_accuracy'], label='Train Accuracy')
plt.plot(history.history['val_sparse_categorical_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# 11. Load the Best Model
model.load_weights(checkpoint_path)

# 12. Evaluate the Model on Validation Set (Optional)
loss, accuracy = model.evaluate(val_dataset)
print(f'Validation Loss: {loss}')
print(f'Validation Accuracy: {accuracy}')

# 13. Convert the Model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Apply default optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Define a representative dataset for quantization (if needed)
def representative_data_gen():
    for image, mask in train_dataset.take(100):
        yield [image]

converter.representative_dataset = representative_data_gen
# Ensure that if using integer quantization, input and output types are set
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

try:
    tflite_model = converter.convert()
    print("Model converted to TFLite successfully.")
except Exception as e:
    print("Error during TFLite conversion:", e)
    tflite_model = converter.convert()  # Try without quantization

# 14. Save the TFLite Model
tflite_model_path = 'unet_lip_model.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f'TFLite model has been saved to {tflite_model_path}')

# 15. (Optional) Test the TFLite Model
def display_sample(image, mask, prediction, class_names=None):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Input Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='jet')
    plt.title('Ground Truth Mask')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(prediction, cmap='jet')
    plt.title('Predicted Mask')
    plt.axis('off')
    
    plt.show()

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Select a sample from the validation set
for image, mask in val_dataset.take(1):
    sample_image = image[0].numpy()
    sample_mask = mask[0].numpy()
    break

# Preprocess the sample image as per TFLite requirements
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']

# Resize and normalize if necessary
input_image = tf.image.resize(sample_image, [input_shape[1], input_shape[2]])
input_image = tf.cast(input_image * 255.0, tf.uint8)  # Assuming input type is uint8

# Add batch dimension
input_image = tf.expand_dims(input_image, axis=0)

# Set the tensor to point to the input data
interpreter.set_tensor(input_details[0]['index'], input_image)

# Run inference
interpreter.invoke()

# Get the output
output = interpreter.get_tensor(output_details[0]['index'])
output = tf.squeeze(output)  # Remove batch dimension

# Post-process the output
pred_mask = tf.argmax(output, axis=-1).numpy().astype(np.uint8)

# Display the results
display_sample(sample_image, sample_mask, pred_mask)

