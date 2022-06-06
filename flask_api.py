from flask import Flask, request, send_file
from flask_cors import CORS, cross_origin
import os
from PIL import Image
from sr_model.capsule_srgan import generator
import numpy as np
from sr_model import resolve_single
import cv2 as cv
import io

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app, resources={r"/upload": {"origins": "http://localhost:3000"}})
capsule_model = generator()
capsule_model.load_weights(
    'GeneratorToDeploy/caps_gan_generator.h5')


@app.route('/')
def hello_world():
    return 'Welcome to Med-CapsSRGAN API'


@app.route('/upload', methods=['POST'])
@cross_origin(origin='localhost', headers=['Content- Type', 'Authorization'])
def file_upload():
    image = request.files['file']
    img = Image.open(image)
    lr = np.array(img)
    sr = resolve_single(capsule_model, lr)
    is_success, buffer = cv.imencode(".jpeg", sr.numpy())
    io_buf = io.BytesIO(buffer)
    return send_file(io_buf, mimetype="image/jpeg", download_name="test.jpeg")


if __name__ == '__main__':
    app.run(host="0.0.0.0")
