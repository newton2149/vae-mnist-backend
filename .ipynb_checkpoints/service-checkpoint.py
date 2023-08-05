import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

def load_model_weights():
    return (encoder,decoder,vae)

def preprocess(img_path):
    import cv2
    import numpy as np
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # gray = 255 - img

    gray = cv2.resize(gray, (28, 28))
    cv2.imwrite('gray'+ img_path, gray)
    img = gray / 255.0
    img = np.array(img).reshape(28, 28, 1)
    return img

def predict_image(img,encoder,decoder,img_w=28, img_h=28):
    mu, _, _ = encoder.predict(tf.expand_dims(img,axis=0))
    sample_vector = np.array([mu[0]])
    decoded_example = decoder.predict(sample_vector)
    decoded_example_reshaped = decoded_example.reshape(img_w, img_h)
    plt.subplot(1, 2, 2)
    plt.title("Generated")
    plt.imshow(decoded_example_reshaped)
    
    real = img
    plt.subplot(1, 2, 1)
    plt.title("Real")
    plt.imshow(real)
    
    return decoded_example_reshaped