import torch
import flask
import time
from flask import Flask
from flask import request
from flask import Flask, render_template, Response, request, redirect, jsonify, send_from_directory, abort, send_file
from flask_cors import CORS
from models import *
import torch.nn as nn
import config
from PIL import Image
import numpy as np 
import cv2
import base64
import os , io , sys

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])

#DOWNLOAD_DIRECTORY = "./"
DOWNLOAD_DIRECTORY = config.DOWNLOAD_DIRECTORY

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
CORS(app)

MODEL = None
DEVICE = config.DEVICE


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
    
    #cartoon.save(filename_cartoon, "JPEG", quality=80, optimize=True, progressive=True)
    #plt.imshow(cartoon_tensor.permute(1, 2, 0))
    #plt.axis('off')
    #plt.savefig(filename_cartoon)

    return filename_cartoon

def serve_pil_image(pil_img):
    img_io = io.BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')    


@app.route('/send_image', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
	
    files = request.files.getlist('files[]')
	
    errors = {}
    success = False
    responsesDocs = []
	
    for file in files:		
        if file and allowed_file(file.filename):
            filename_cartoon = face_to_cartoon(file.filename, file.read())

            #output = serve_pil_image(output)
            dict_dummy = {}
            dict_dummy['file_name'] = filename_cartoon
            responsesDocs.append(dict_dummy)
            success = True
        else:
            errors[file.filename] = 'File type is not allowed'
	
    if success and errors:
        errors['message'] = 'File(s) successfully uploaded'
        resp = jsonify(errors)
        resp.status_code = 500
        return resp
    if success:
        # add service textract responce
        resp = jsonify({'message' : 'Files successfully processed', 'responses_docs': responsesDocs})
        resp.status_code = 201
        resp.headers.add('Access-Control-Allow-Origin', '*')
        print('headers:: ', resp.headers)
        return resp

        #return serve_pil_image(output)
    else:
        resp = jsonify(errors)
        resp.status_code = 500
        return resp


@app.route('/predict', methods=['POST'])
def predict():
    doc_name = request.form.get('cartoon_name')
    try:
        #doc_name = str(doc_name)
        #image = Image.open("/home/stevramos/Documents/personal_projects/xgan/be-app-xgan/{}".format(doc_name))
        #img_io = io.BytesIO()
        #image.save(img_io, 'JPEG', quality=70)
        #img_io.seek(0)
        
        #return send_file(img_io, mimetype='image/jpeg')  

        return send_from_directory(DOWNLOAD_DIRECTORY, filename=doc_name, as_attachment=True)
    except FileNotFoundError:
        abort(404)


if __name__ == "__main__":
    MODEL = Avatar_Generator_Model(config.MODEL_PATH, config.DEVICE)

    app.run(host="0.0.0.0", port="9999")