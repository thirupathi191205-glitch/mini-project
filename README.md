# ==========================================
# 1. Install dependencies
# ==========================================
!pip install kagglehub mediapipe tensorflow gTTS opencv-python
from IPython.display import display, Javascript, Audio
from google.colab import output
from gtts import gTTS
import cv2
import numpy as np
import tensorflow as tf
import time
import os
from base64 import b64decode

# ==========================================
# 2. Load Kaggle dataset
# ==========================================
import kagglehub
dataset_path = kagglehub.dataset_download("datamunge/sign-language-mnist")
print("Dataset path:", dataset_path)

# ==========================================
# 3. Prepare dataset
# ==========================================
import pandas as pd
train_df = pd.read_csv(os.path.join(dataset_path, 'sign_mnist_train.csv'))
test_df = pd.read_csv(os.path.join(dataset_path, 'sign_mnist_test.csv'))

# Use only required letters A-F
selected_labels = [0,1,2,3,4,5]  # assuming dataset has 0=A, 1=B...
train_df = train_df[train_df['label'].isin(selected_labels)]
test_df = test_df[test_df['label'].isin(selected_labels)]

# Split features and labels
X_train = train_df.drop('label', axis=1).values
y_train = train_df['label'].values
X_test = test_df.drop('label', axis=1).values
y_test = test_df['label'].values

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape to images
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

# ==========================================
# 4. Build model
# ==========================================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28,28,1)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ==========================================
# 5. Train model (5 epochs)
# ==========================================
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5)

# Save model in native Keras format
model.save("gesture_model.keras")

# Mapping labels to words
label_map = {0:'Hello', 1:'Thank You', 2:'Welcome', 3:'Good Morning', 4:'Good Night', 5:'Eat'}

# ==========================================
# 6. Capture webcam image in Colab
# ==========================================
gesture_prediction = ""

def capture_image():
    display(Javascript('''
        async function takePhoto() {
            const div = document.createElement('div');
            const video = document.createElement('video');
            div.appendChild(video);
            document.body.appendChild(div);
            const stream = await navigator.mediaDevices.getUserMedia({video: true});
            video.srcObject = stream;
            await video.play();
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const context = canvas.getContext('2d');

            // Capture frames for 5 seconds
            let endTime = Date.now() + 5000;
            while(Date.now() < endTime){
                context.drawImage(video, 0, 0, canvas.width, canvas.height);
                await new Promise(r => setTimeout(r, 100));
            }

            const dataUrl = canvas.toDataURL('image/jpeg', 1.0);
            stream.getTracks().forEach(track => track.stop());
            div.remove();
            google.colab.kernel.invokeFunction('notebook.capture', [dataUrl], {});
        }
        takePhoto();
    '''))

# ==========================================
# 7. Function to handle JS callback
# ==========================================
def decode_image(dataUrl):
    global gesture_prediction
    # Remove the prefix and decode
    header, encoded = dataUrl.split(",", 1)
    data = b64decode(encoded)
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Resize to 28x28
    img = cv2.resize(img, (28,28))
    img = img / 255.0
    img = img.reshape(1,28,28,1)

    # Predict
    pred = model.predict(img)
    label = np.argmax(pred)
    gesture_prediction = label_map.get(label, "Unknown")
    print("Predicted Gesture:", gesture_prediction)

    # Convert to speech
    speech = gTTS(text=gesture_prediction, lang='en', slow=False)
    speech.save("gesture.mp3")
    display(Audio("gesture.mp3", autoplay=True))

output.register_callback('notebook.capture', decode_image)

# ==========================================
# 8. Run capture
# ==========================================
capture_image()
