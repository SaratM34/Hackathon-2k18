from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import random
import cv2
#from keras.preprocessing.image import img_to_array
from sklearn.externals import joblib

# Keras
#from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.wsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

model = joblib.load('imgmodel.pkl')



def model_predict(img_path, model):

    image = cv2.imread(img_path)
    image = cv2.resize(image, (28, 28))
    #image = img_to_array(image)
    xdata = image.reshape(1,-1)
    preds = model.predict(xdata)

    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)

        return str(preds)
    return None


@app.route('/part2', methods=['GET'])
def part2index():
    return render_template('test.html')

@app.route('/part2',methods=['POST','GET'])
def get_delay():
    if request.method=='POST':
        result=request.form

        vmake = result['vmake']
        vmodel = result['vmodel']
        year = result['year']

        dict2 = {"Isuzu": 11,
"Land Rover": 13,
"Chevrolet": 5,
"Ford": 8,
"Honda": 10,
"GMC": 9,
"Toyota": 25,
"Mercedes-Benz": 17,
"BMW": 2,
"Dodge": 7,
"Acura": 0,
"Buick": 3,
"Saturn": 23,
"Chrysler": 6,
"smart": 28,
"Nissan": 20,
"Mitsubishi": 19,
"Volkswagen": 26,
"Jeep": 12,
"Pontiac": 22,
"Audi": 1,
"Oldsmobile": 21,
"MINI": 15,
"Subaru": 24,
"Cadillac": 4,
"Mazda": 16,
"Lincoln": 14,
"Volvo": 27,
"Mercury": 18}

        dict = {"NQR":65 ,"Range Rover Sport" : 75 ,"C3500 HD Chassis" :15 ,
"F-350": 40,
"C7500": 16,
"Express": 38,
"F-650": 43,
"Silverado 3500 Chassis": 89,
"Tahoe Hybrid": 95,
"F-450 Chassis": 42,
"CR-V": 17,
"Sierra 3500 Chassis": 84,
"Silverado 1500": 86,
"Avalon": 11,
"C300": 14,
"X5": 106,
"Ram 1500": 73,
"Pilot": 71,
"Acadia": 7,
"Yukon": 107,
"MDX": 59,
"Camry": 19,
"Civic": 24,
"LaCrosse": 57,
"Corvette": 30,
"L200": 55,
"Town & Country": 98,
"LeSabre": 58,
"Accord": 8,
"RAV4": 72,
"PT Cruiser": 70,
"328xi": 0,
"Stratus": 90,
"Avenger": 12,
"TrailBlazer": 100,
"Yukon XL 1500": 108,
"fortwo": 109,
"Taurus": 96,
"Maxima": 62,
"4WD Trucks": 2,
"Odyssey": 67,
"Expedition": 36,
"Traverse": 101,
"Outlander": 69,
"Tahoe": 94,
"Concorde": 27,
"Venture": 104,
"F-350 Chassis": 41,
"Golf": 48,
"Silverado 2500": 87,
"Grand Cherokee": 50,
"Ridgeline": 78,
"G6": 47,
"Wrangler": 105,
"Ranger": 76,
"Tacoma": 93,
"Suburban 1500": 91,
"A6": 6,
"750": 4,
"Montana": 63,
"Alero": 10,
"Rendezvous": 77,
"Cooper S": 28,
"Grand Prix": 51,
"Century": 21,
"L300": 56,
"Cherokee": 23,
"Impala": 52,
"Outback": 68,
"SRX": 81,
"Tracker":99 ,
"Sierra 1500": 83,
"CTS": 18,
"626": 3,
"MKS": 60,
"Tiguan": 97,
"VUE": 103,
"Grand Am": 49,
"Explorer": 37,
"Ram 3500": 74,
"Navigator": 66,
"Frontier": 45,
"Silhouette": 85,
"Uplander": 102,
"Corolla": 29,
"Diamante": 33,
"Cavalier": 20,
"Eclipse": 34,
"S60": 79,
"DeVille": 32,
"A4": 5,
"Cobalt": 26,
"Jetta": 53,
"SL": 80,
"Bonneville": 13,
"Matrix": 61,
"Sable": 82,
"Civic Hybrid": 25,
"T100": 92,
"Forester": 44,
"Accord Crosstour": 9,
"Cutlass Supreme": 31,
"Mustang": 64,
"Fusion": 46,
"4Runner": 1,
"Charger": 22,
"K3500 Chassis": 54,
"F-150": 39,
"Silverado 2500HD": 88,
"Enclave": 35}



        input = np.array([[year,dict[vmodel],dict2[vmake]]])
        clf = joblib.load('model.pkl')
        prediction = clf.predict(input)
        return render_template('result.html',prediction=prediction)

if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
