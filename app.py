import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import load_img
import numpy as np
from flask import Flask,request

app = Flask(__name__)

class_names = ['Bacterial spot','Early blight','healthy','Late blight','Leaf mold','Septoria leaf spot','Spider mites two spotted','Target spot','mosaic virus','Yellow leaf curl virus','Bacterial spot','Early blight','healthy','Late blight','Leaf mold','Septoria leaf spot','Spider mites two spotted','Target spot','mosaic virus','Yellow leaf curl virus']

@app.route("/",methods=["GET","POST"])
def hello():
    test =load_img('a.jpg',target_size=(150,150,3))
    model = keras.models.load_model('test_model.h5')
    test=np.array(test)
    test=test/255
    test=np.expand_dims(test,axis=0)
    result=model.predict(test)
    pred_name = np.argmax(result)
    pred_name = class_names[pred_name] 
    return pred_name

    
if __name__ == "__main__":
    app.run()