import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
import os

# Define parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 4  # glioma, meningioma, no_tumor, pituitary

# Load dataset
train_dir = '../data/preprocessed/enhance_then_resize/Training/'
test_dir = '../data/preprocessed/enhance_then_resize/Testing/'

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = datagen.flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='training')
val_data = datagen.flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='validation')
test_data = datagen.flow_from_directory(test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

# Load MobileNetV2
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base model

# Build the model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
os.makedirs('logs', exist_ok=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
csv_logger = CSVLogger('logs/training_log.csv', append=True)
tensorboard = TensorBoard(log_dir='logs/tensorboard', histogram_freq=1)

# Train the model
model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=[checkpoint, csv_logger, tensorboard])

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_data)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

# Save the trained model
model.save('brain_tumor_mobilenetv2.h5')
