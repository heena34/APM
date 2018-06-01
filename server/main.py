# -*- coding: utf-8 -*-
"""
Created on Wed May 23 16:08:50 2018

@author: Prathmesh Swaroop
"""

from flask import request,Flask,send_file,jsonify


from flask import flash, redirect

from flask_api import status

#from binary.binary_classification import BinaryClassification

from server.binary.binary_classification import BinaryClassification

import os

from werkzeug.utils import secure_filename

import json

import sys



path = 'D:\\Hack4\\WebApp'
UPLOAD_FOLDER = 'tmp/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'csv'])
executor = BinaryClassification()

if path not in sys.path:
    sys.path.append(path)

app = Flask(__name__, static_folder="www")


generated_graphs = {
  0:'../APM/Output/model_regression_verify.png',
  1:'../APM/Output/model_accuracy.png',
  2:'../APM/Output/datasetSample.png',
  3:'../APM/Output/model_loss.png',
  4:'../APM/Output/model_mae.png',
  5:'../APM/Output/model_r2.png',
  6:'../APM/Output/model_regression_loss.png',
  7:'../APM/Output/model_verify.png'
  
}


@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route('/<path:path>')
def send_js(path):
    if ".js" in path:
        return app.send_static_file(path)
    if ".css" in path:
        return app.send_static_file(path)
    if ".png" in path:
        return app.send_static_file(path)
    if "ttf" in path:
        return app.send_static_file(path)
    else:
        return app.send_static_file('index.html')


@app.route("/<int:key>/")
def get_image(key):
    
    if key not in generated_graphs:
        return '',status.HTTP_404_NOT_FOUND
    return send_file(generated_graphs[key],mimetype='image/png')


@app.route("/train", methods = ['POST'])
def start_training():
    if request.method == 'POST':
     if request.data is None:
       return "",status.HTTP_400_BAD_REQUEST
    request_object = json.loads(request.data)
    
    if executor.startTraining(int(request_object['w1']),int(request_object['w0']),100) is True:
        return '',status.HTTP_200_OK
    else:
        return '',status.HTTP_417_EXPECTATION_FAILED



@app.route("/test", methods = ['POST'])
def execute_testing():
    print("entered test")
    if request.method == 'POST':
        if request.data is None:
            return "",status.HTTP_400_BAD_REQUEST
        print(request.data)
        request_object = json.loads(request.data.decode('utf-8'))
        print("request_object")
        print(request_object)
        print(request_object['w1'])
        
        response = executor.executeAIOnTest(int(request_object['w1']),int(request_object['w0']))
        if response is not None:       
            return jsonify(response),status.HTTP_200_OK
        else:
            return '',status.HTTP_417_EXPECTATION_FAILED


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST','GET'])
def upload_func():
    '''upload with post Method'''
    print('starting file upload')
    try:
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)            
        
        file = request.files['file']
        if file == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            
            filename = secure_filename(file.filename)
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)

        for the_file in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                LOGGER_SERVICE.logger.error(e)

        file.save(os.path.join(UPLOAD_FOLDER, filename))
        return "",status.HTTP_200_OK
    except Exception as EXP:
        print(EXP)
        return "",status.HTTP_406_NOT_ACCEPTABLE


if __name__ == "__main__":    
    app.secret_key = os.urandom(24)
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(host='127.0.0.1',port=80,debug = True)