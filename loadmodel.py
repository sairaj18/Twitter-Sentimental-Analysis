import os
import sys
import numpy as np
from keras.models import load_model
import tensorflow.compat.v1 as tf
import pickle
from keras.preprocessing.sequence import pad_sequences
tf.disable_eager_execution()
from tensorflow.python.keras.backend import set_session

sess = tf.Session()
graph = tf.get_default_graph()

def loading_model():
	global sess
	set_session(sess)
	model=load_model('sentiment.h5')
	token=pickle.load(open('token.pickle','rb'))
	print('model loaded')
	return model,token
def predcit(model,token,text):
	global sess
	global graph
	set_session(sess)
	hh = token.texts_to_sequences([text])
	hh = pad_sequences(hh, maxlen=29)
	hh=np.array(hh[0]).reshape(1,29)
	with graph.as_default():
		out=model.predict(hh)
	label=np.argmax(out)
	print(label)
	if label==0:
		label_name='negative'
	elif label==1:
		label_name='neutral'
	elif label==2:
		label_name='positive'
	return label_name
