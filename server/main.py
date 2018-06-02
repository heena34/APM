# -*- coding: utf-8 -*-
"""
Created on Wed May 23 16:08:50 2018

@author: Prathmesh Swaroop
"""

from flask import request,Flask,send_file


from flask import flash, redirect,jsonify, stream_with_context, Response

from flask_api import status

from server.binary.binary_classification import BinaryClassification
from server.ml.lk_regression1 import MLPredictions
from server.ml.stream_main import Live_Streaming

import os, time

from werkzeug.utils import secure_filename

import json

import sys



path = 'D:\\Hack4\\WebApp'
UPLOAD_FOLDER = './server/tmp/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'csv'])
bin_executor = BinaryClassification()


if path not in sys.path:
    sys.path.append(path)

app = Flask(__name__)


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

@app.route("/graph/<int:key>/")
def get_image(key):
    
    if key not in generated_graphs:
        return '',status.HTTP_404_NOT_FOUND
    return send_file(generated_graphs[key],mimetype='image/png')


@app.route("/ml/live_stream")
def get_live_stream():
    
    def generate():
        wait_time = 1
        index = 1
        while True:
            ls = Live_Streaming(10000)    
            print(ls.start_streaming(30,15,index))
            #yield(jsonify(ls.start_streaming(30,15,index)))
            time.sleep(wait_time)
            index = index + 1
    return Response(stream_with_context(generate()))

@app.route("/binary/train", methods = ['POST'])
def start_training():
    if request.method == 'POST':
     if request.data is None:
       return "",status.HTTP_400_BAD_REQUEST
    request_object = json.loads(request.data)
    
    if bin_executor.startTraining(int(request_object['w1']),int(request_object['w0']),100) is True:
        return '',status.HTTP_200_OK
    else:
        return '',status.HTTP_417_EXPECTATION_FAILED



@app.route("/ml/train", methods = ['POST'])
def execute_ml_training():
    if request.method == 'POST':
        if request.data is None:
            return "",status.HTTP_400_BAD_REQUEST
        
        request_object = json.loads(request.data)        
        ml_executor = MLPredictions(str(request_object['algo_name']),str(request_object['approach']),int(request_object['w1']),int(request_object['w0']))                
        response = ml_executor.start_model_training()
        if response is True:
            return '',status.HTTP_200_OK
        else:
            return '',status.HTTP_417_EXPECTATION_FAILED
        




@app.route("/ml/test", methods = ['POST'])
def execute_ml_testing():
    if request.method == 'POST':
        if request.data is None:
            return "",status.HTTP_400_BAD_REQUEST
        
        request_object = json.loads(request.data)        
        ml_executor = MLPredictions(str(request_object['algo_name']),str(request_object['approach']),int(request_object['w1']),int(request_object['w0']))                
        response = ml_executor.start_model_testing()
        if response is not None:            
            return jsonify(response),status.HTTP_200_OK
        else:
            return '',status.HTTP_417_EXPECTATION_FAILED



@app.route("/binary/test", methods = ['POST'])
def execute_bin_testing():
    if request.method == 'POST':
        if request.data is None:
            return "",status.HTTP_400_BAD_REQUEST
        
        request_object = json.loads(request.data)        
        response = bin_executor.executeAIOnTest(int(request_object['w1']),int(request_object['w0']))
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
                print(e)
                file.save(os.path.join(UPLOAD_FOLDER, filename))

        return 200
    except Exception as EXP:
        print(EXP)
        return "",status.HTTP_406_NOT_ACCEPTABLE


if __name__ == "__main__":    
    app.secret_key = os.urandom(24)
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(host='127.0.0.1',port=80,debug = True,threaded = True)