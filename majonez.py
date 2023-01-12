import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import sys
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

filename = sys.argv[1]

print("File name:", filename)

img = cv2.imread(filename)
resize = tf.image.resize(img, (256,256))

new_model = load_model('majonez.h5')
if new_model.predict(np.expand_dims(resize/255, 0)) > 0.5:
    print("Winiary")
else:
    print("Kielecki")
