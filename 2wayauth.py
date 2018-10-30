import flask
from flask import Flask, request
from werkzeug.utils import secure_filename
import cv2
import os
import numpy as np
import math
from keras.models import load_model
from keras import applications
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.applications.resnet50 import ResNet50
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.config['upload_folder'] = './temp'
app.config['train_imgs'] = './train_imgs'

@app.route('/')
def index():
	return flask.jsonify('Hello, world')

@app.route('/predict', methods=['POST'])
def upload():
	image = request.files['image']
	filename = secure_filename(image.filename)
	image.save(os.path.join(app.config['upload_folder'], filename))
	response = process_image(filename)
	k.clear_session()
	return flask.jsonify(response)

def process_image(image):
	img = cv2.imread('./temp/'+image)	
	img = img[:,:,::-1]
	img = cv2.GaussianBlur(img,(5,5),0)
	lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	lab_planes = cv2.split(lab)
	clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(5,5))
	lab_planes[0] = clahe.apply(lab_planes[0])
	lab = cv2.merge(lab_planes)
	img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
	resizedImg = np.zeros((200, 200))
	img = cv2.normalize(img,  resizedImg, 0, 255, cv2.NORM_MINMAX)
	roi = np.asarray(img)
	roi = np.expand_dims(roi, axis=0)
	response = predict_class(roi)
	return response

def predict_class(roi):	
	model=load_model('face_model.h5')
	y_prob = model.predict(roi)
	#return str(y_prob[0])
	y_classes = y_prob.argmax(axis=-1)
	#print(y_classes)
	if y_classes == 0:
    		return 'Authentication Success'
	else:
    		return 'Try again'

@app.route('/register', methods=['POST'])
def register():
	add_negative_img()
	modelX = compile_model()
	for f in request.files.getlist('photo'):
        	filename = secure_filename(f.filename)
        	f.save(os.path.join(app.config['train_imgs'], filename))
	train_model(modelX)
	return flask.jsonify('test')

def compile_model():
	base_model = ResNet50(weights='imagenet',include_top=False, input_shape=(200,200,3))
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dropout(0.4)(x)
	x = Dense(1024, activation='relu')(x)
	predictions = Dense(2, activation='softmax')(x)
	modelX = Model(inputs=base_model.input, outputs=predictions)
	for layer in base_model.layers:
		layer.trainable = False
	modelX.compile(optimizer='sgd', loss='sparse_categorical_crossentropy')
	layer_num = len(modelX.layers)
	for layer in modelX.layers[:int(layer_num * 0.9)]:
		layer.trainable = False
	for layer in modelX.layers[int(layer_num * 0.9):]:
		layer.trainable = True
	modelX.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
	#modelX.summary()
	return modelX

def train_model():
	iArr1 = []
	y = []
	for i in os.listdir('my_images/'):
    		img = cv2.imread('my_images/'+ i )
    		if img is None:
      			continue
    		img = img[:,:,::-1]
    		img = cv2.GaussianBlur(img,(5,5),0)
    		lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    		lab_planes = cv2.split(lab)
    		clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(5,5))
    		lab_planes[0] = clahe.apply(lab_planes[0])
    		lab = cv2.merge(lab_planes)
    		img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    		resizedImg = np.zeros((200, 200))
    		img = cv2.normalize(img,  resizedImg, 0, 255, cv2.NORM_MINMAX)
    		#print(img.shape)  
    		#start = i.find('_')
    		#end = i.rfind('_')
    		if(i[0:3] == 'img'):
        		y.append(0)
    		else:
        		y.append(1)
    		iArr1.append(img)
	X = np.asarray(iArr1)
	Y = y
	x_train,x_test,y_train,y_test = train_test_split( X , Y, test_size=0.25, shuffle=True, random_state=1)
	y_train = np_utils.to_categorical(y_train)
	y_test = np_utils.to_categorical(y_test)
	modelX.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=30, batch_size=128, verbose=2)
	modelX.save('face_model.h5')

def add_negative_img():
	cnt = 0
	for i in os.listdir('imdb'):
    		if cnt>999:
        		break
   	 	cnt+=1    
    		shutil.copy2('imdb/'+i,'train_imgs')

# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404

@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500
    
if __name__ == '__main__':
	app.run('localhost', 8090, debug=True)
