import torch
import flask
import time
from flask import Flask
from flask import request
from flask import Flask, render_template, Response, request, redirect, jsonify, send_from_directory, abort, send_file
from flask_cors import CORS
from models import *
from utils import *
import torch.nn as nn
from PIL import Image
import numpy as np 
import cv2
import base64
import os , io , sys

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
CONFIG_FILENAME = "config.json"
DOWNLOAD_DIRECTORY = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
CORS(app)

MODEL = None


def face_to_cartoon(DOC_FILE, face):

    document_name = DOC_FILE.split('.')[0]
    extension = (DOC_FILE.split('.')[-1]).lower()
    document = Image.open(io.BytesIO(face))

    if not os.path.exists(DOWNLOAD_DIRECTORY):
        os.makedirs(DOWNLOAD_DIRECTORY)
    
    if extension == "png":
        format_image = "PNG"
    else:
        extension = "jpg"
        format_image = "JPEG"

    filename_face = "{}.{}".format(document_name, extension)
    document.save(DOWNLOAD_DIRECTORY + filename_face, format_image, quality=80, optimize=True, progressive=True)


    filename_cartoon = "{}_cartoon.jpg".format(document_name)
    cartoon, cartoon_tensor = MODEL.generate(DOWNLOAD_DIRECTORY + filename_face, DOWNLOAD_DIRECTORY + filename_cartoon)
    
    return filename_cartoon


@app.route('/send_image', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'face_image' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
	
    file = request.files['face_image']

    errors = {}
    success = False

    if file and allowed_file(file.filename):
        filename_cartoon = face_to_cartoon(file.filename, file.read())
        success = True
    else:
        errors[file.filename] = 'File type is not allowed'
	
    if success and errors:
        errors['message'] = 'File(s) successfully uploaded'
        resp = jsonify(errors)
        resp.status_code = 500
        return resp
    if success:
        resp = jsonify({'message' : 'Files successfully processed', 'filename_cartoon': filename_cartoon})
        resp.status_code = 201
        resp.headers.add('Access-Control-Allow-Origin', '*')
        print('headers:: ', resp.headers)
        return resp
    else:
        resp = jsonify(errors)
        resp.status_code = 500
        return resp


@app.route('/predict', methods=['POST'])
def predict():
    doc_name = request.form.get('filename_cartoon')
    try:

        return send_from_directory(DOWNLOAD_DIRECTORY, filename=doc_name, as_attachment=True)
    except FileNotFoundError:
        abort(404)


if __name__ == "__main__":

    use_wandb = False
    config = configure_model(CONFIG_FILENAME,use_wandb=use_wandb)
    DOWNLOAD_DIRECTORY = config.download_directory

    #MODEL = Avatar_Generator_Model(config.model_path, config.device)
    MODEL = Avatar_Generator_Model(config, use_wandb=use_wandb)
    MODEL.load_weights(config.model_path)

    app.run(host="0.0.0.0", port="9999")