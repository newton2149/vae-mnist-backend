
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import asyncio
import websockets
import base64
from PIL import Image
import io
import matplotlib.pyplot as mp


class CustomLayer(tf.keras.layers.Layer):

    def vae_loss(self, x, z_decoded, z_mu, z_sigma):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        
        # Reconstruction loss (as we used sigmoid activation we can use binarycrossentropy)
        recon_loss = tf.keras.metrics.binary_crossentropy(x, z_decoded)
        
        # KL divergence
        kl_loss = -5e-4 * K.mean(1 + z_sigma - K.square(z_mu) - K.exp(z_sigma), axis=-1)
        return K.mean(recon_loss + kl_loss)

    # add custom loss to the class
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        z_mu = inputs[2]
        z_sigma = inputs[3]
        
        loss = self.vae_loss(x, z_decoded, z_mu, z_sigma)
        self.add_loss(loss, inputs=inputs)
        return x
    
    
def load_model_weights():
    custom_objects = {"CustomLayer": CustomLayer}
    reconstructed_encoder = tf.keras.models.load_model( "./models/encoderModel.keras")
    reconstructed_vae =tf.keras.models.load_model("./models/vaeModel.keras",
                                                  custom_objects={"CustomLayer": CustomLayer},)
    
    reconstructed_decoder = tf.keras.models.load_model(
    "./models/decoderModel.keras", )
    
    
    return (reconstructed_encoder,reconstructed_decoder,reconstructed_vae)

def preprocess(img_path):
    import cv2
    import numpy as np
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    print(gray.shape)
    # gray = 255 - img

    gray = cv2.resize(gray, (28, 28))
    cv2.imwrite('gray'+ img_path, gray)
    img = gray / 255.0
    img = np.array(img).reshape(28, 28, 1)
    return img

def predict_image(img,encoder,decoder,img_w=28, img_h=28):
    import cv2
    import numpy as np
    mu, _, _ = encoder.predict(tf.expand_dims(img,axis=0))
    sample_vector = np.array([mu[0]])
    decoded_example = decoder.predict(sample_vector)
    decoded_example_reshaped = decoded_example.reshape(img_w, img_h)
    img = cv2.resize(decoded_example_reshaped, (1000, 1000))
    return img

encoder,decoder,vae = load_model_weights()


async def websocket_server(websocket, path):
    while True:
        
        


        data = await websocket.recv()

        if data == None:
            continue
        
        base_64_data = data.split(',')[1]
        decoded_data=base64.b64decode((base_64_data))


        # This functional api when invoked the image obtained is not accurate
        # image = Image.open(io.BytesIO(decoded_data))
        # np_array = np.array(image)

        
        img_file = open('image.jpeg', 'wb')
        img_file.write(decoded_data)
        img_file.close()


        img_to_predict = preprocess("./image.jpeg")
        predicted_image = predict_image(img_to_predict,encoder,decoder)
        mp.imsave("Image_from_array.jpeg", predicted_image)
        img_file = open('Image_from_array.jpeg', 'rb')

        image_base64 = base64.b64encode(img_file.read()).decode()

        await websocket.send(image_base64)


start_server = websockets.serve(websocket_server, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
