import cv2
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from skimage import io
import os
import time
from tensorflow.keras.layers import (Input, Conv2D, Conv2DTranspose, MaxPooling2D, 
                                     Concatenate, BatchNormalization, Activation)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from focal_loss import SparseCategoricalFocalLoss
from tensorflow.keras import layers

app = Flask(__name__)

# --- Custom Classes and Loss ---
class WeightedSparseCategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, class_weights=[3.0, 3.0, 3.0, 4.0, 3.0, 1.0], gamma=2, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = tf.constant(class_weights, dtype=tf.float32)
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
        ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_int, y_pred, from_logits=False)
        y_pred_soft = tf.nn.softmax(y_pred)
        y_true_onehot = tf.one_hot(y_true_int, depth=tf.shape(y_pred)[-1])
        p_t = tf.reduce_sum(y_pred_soft * y_true_onehot, axis=-1)
        focal_factor = tf.pow(1 - p_t, self.gamma)
        weights = tf.gather(self.class_weights, y_true_int)
        loss = weights * focal_factor * ce_loss
        return tf.reduce_mean(loss)

def decoder_block(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2,2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    x = GraphAttentionLayer(num_filters)(x)
    return x

class GraphAttentionLayer(layers.Layer):
    def __init__(self, channels, **kwargs):
        super(GraphAttentionLayer, self).__init__(**kwargs)
        self.channels = channels
        self.conv1 = Conv2D(channels, (1, 1), activation="relu")
        self.conv2 = Conv2D(channels, (1, 1), activation="sigmoid")
    
    def call(self, inputs):
        x = self.conv1(inputs)
        attention = self.conv2(x)
        return x * attention
    
class SelfRoutingCapsuleLayer(layers.Layer):
    def __init__(self, num_capsules, capsule_dim, num_routing=3, **kwargs):
        super(SelfRoutingCapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.num_routing = num_routing
        self.conv = Conv2D(num_capsules * capsule_dim, (3, 3), padding="same", activation="relu")
    
    def call(self, inputs):
        x = self.conv(inputs)
        batch_size = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        x = tf.reshape(x, (batch_size, H * W, self.num_capsules, self.capsule_dim))
        b = tf.zeros((batch_size, H * W, self.num_capsules))
        for i in range(self.num_routing):
            c = tf.nn.softmax(b, axis=-1)
            s = tf.reduce_sum(tf.expand_dims(c, -1) * x, axis=1)
            v = tf.nn.l2_normalize(s, axis=-1)
            b += tf.reduce_sum(x * tf.expand_dims(v, 1), axis=-1)
        v = tf.reshape(v, (batch_size, 1, 1, self.num_capsules * self.capsule_dim))
        v = tf.tile(v, [1, H, W, 1])
        return v

# --- Model Setup ---
class_weights = [3.0, 3.0, 3.0, 4.0, 3.0, 1.0]
loss_instance = WeightedSparseCategoricalFocalLoss(class_weights, gamma=2)
custom_objects = {
    "GraphAttentionLayer": GraphAttentionLayer,
    "SelfRoutingCapsuleLayer": SelfRoutingCapsuleLayer,
    "weighted_sparse_categorical_focal_loss": loss_instance,
    "WeightedSparseCategoricalFocalLoss": WeightedSparseCategoricalFocalLoss
}

model_path = r"C:\Users\Admin\OneDrive\Desktop\ML\models\model5(RESUNET WITH ALL TECHNIQUES)\best_student_model.h5"
try:
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# --- Utility Functions ---
COLOR_MAP = {
    0: { 'center': (255, 0, 0), 'range': [(245, 255), (0, 10), (0, 10)] },
    1: { 'center': (255, 192, 203), 'range': [(245, 255), (182, 202), (193, 213)] },
    2: { 'center': (0, 0, 255), 'range': [(0, 10), (0, 10), (245, 255)] },
    3: { 'center': (0, 255, 0), 'range': [(0, 10), (245, 255), (0, 10)] },
    4: { 'center': (255, 255, 0), 'range': [(245, 255), (245, 255), (0, 10)] },
    5: { 'center': (0, 0, 0), 'range': [(0, 10), (0, 10), (0, 10)] }
}

def rgb_to_label(mask_rgb):
    H, W, _ = mask_rgb.shape
    label_mask = np.zeros((H, W), dtype=np.uint8)
    for class_idx, color_info in COLOR_MAP.items():
        r_range, g_range, b_range = color_info['range']
        matches = (
            (mask_rgb[..., 0] >= r_range[0]) & (mask_rgb[..., 0] <= r_range[1]) &
            (mask_rgb[..., 1] >= g_range[0]) & (mask_rgb[..., 1] <= g_range[1]) &
            (mask_rgb[..., 2] >= b_range[0]) & (mask_rgb[..., 2] <= b_range[1])
        )
        label_mask[matches] = class_idx
    return label_mask

def label_to_rgb(label_mask):
    H, W = label_mask.shape
    rgb_mask = np.zeros((H, W, 3), dtype=np.uint8)
    for class_idx, color_info in COLOR_MAP.items():
        rgb_mask[label_mask == class_idx] = color_info['center']
    return rgb_mask

# --- Constants ---
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 192, 288, 3
UPLOAD_FOLDER = "static/uploads"
MASK_FOLDER = "static/masks"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)

# --- Flask Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided.'}), 400

        file = request.files['image']
        filename = file.filename
        img_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(img_path)

        # Load and preprocess image
        img = io.imread(img_path)
        if img.shape[-1] == 4:
            img = img[..., :3]

        # Resize the original image to match the model's input size (192x288)
        img_resized_for_save = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        # Save the resized original image (convert to BGR for OpenCV)
        img_resized_bgr = cv2.cvtColor(img_resized_for_save, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, img_resized_bgr)
        print(f"Resized original image saved to: {img_path}")

        # Preprocess for model input
        img_resized = tf.image.resize(img, (192, 288))
        img_resized = img_resized.numpy().astype(np.float32) / 255.0
        input_data = np.expand_dims(img_resized, axis=0)

        # Predict
        prediction = model.predict(input_data)
        predicted_mask = np.argmax(prediction[0], axis=-1)
        unique_classes = np.unique(predicted_mask)
        print(f"Unique classes in predicted mask: {unique_classes}")

        # Convert mask to RGB
        predicted_mask_rgb = label_to_rgb(predicted_mask)
        unique_colors = np.unique(predicted_mask_rgb.reshape(-1, 3), axis=0)
        print(f"Unique colors in predicted_mask_rgb: {unique_colors}")

        # Convert to BGR for OpenCV
        predicted_mask_bgr = cv2.cvtColor(predicted_mask_rgb, cv2.COLOR_RGB2BGR)

        # Save the predicted mask
        timestamp = int(time.time())
        mask_filename = f"mask_{timestamp}_{filename}"
        mask_path = os.path.join(MASK_FOLDER, mask_filename)
        cv2.imwrite(mask_path, predicted_mask_bgr)
        print(f"Mask saved to: {mask_path}")

        # Verify the saved mask
        saved_mask = cv2.imread(mask_path)
        if saved_mask is None:
            print("Error: Failed to save or read the mask image.")
        else:
            unique_saved_colors = np.unique(saved_mask.reshape(-1, 3), axis=0)
            print(f"Unique colors in saved mask: {unique_saved_colors}")

        mask_url = f"/{MASK_FOLDER}/{mask_filename}?t={timestamp}"
        return jsonify({
            'original_image': f"/{UPLOAD_FOLDER}/{filename}?t={timestamp}",
            'predicted_mask': mask_url
        })

    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)