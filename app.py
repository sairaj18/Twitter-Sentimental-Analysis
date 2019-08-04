from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from flask import jsonify
import re
import loadmodel
from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential
model,token=loadmodel.loading_model()
app=Flask(__name__)

@app.route("/pr",methods=['GET','POST'])		
def index():
    return render_template("index.html")

@app.route("/home",methods=['GET','POST'])		
def home():
    return render_template("home.html")

@app.route("/prediction",methods=['GET','POST'])
def prediction():
    if request.method == 'POST':
    	text=request.form['firstname']
    	out= loadmodel.predcit(model,token,text)
    return render_template('result.html',prediction=out)

if __name__ == '__main__':
	app.run(debug=True)