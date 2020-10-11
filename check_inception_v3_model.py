from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import cv2
import glob

#image folder
folder_path = './pictures/'
file_path = './pictures/ikea003.526.29(0).jpg'

#dimesions of images
img_width,img_height = 299,299
class inception_retrain(object):
    def __init__(self):
        self.img=None
        self.model=None
        self.InV3model=None
    def _load_image(self,img):
        '''Takes an image
            Returns its proper form to feed into model's predcition '''
        #image = cv2.imread('test/{}'.format(img))
        fd = open(img,'rb')
        img_str = fd.read()
        fd.close()
        nparr = np.fromstring(img_str, np.uint8)
        image = cv2.imdecode(nparr, -1)[:,:,:3] #imdecode flag = -1 Unchanged
        image = cv2.resize(image, (299, 299))
        image = np.expand_dims(image/255, axis=0)
        image = np.vstack([image])
        return image
    def _feature_extraction_inception(self,img):
        image=self._load_image(img)
        self.img=image
        features=self.InV3model.predict(image)
        return features
    def _load_model(self):
        if self.model is None:
            self.model=load_model('./inV3_last_layer_final_ver.h5')
        if self.InV3model is None:
            self.InV3model=load_model("./inception.h5")
    def predict(self,img):
        '''Takes an imagebbb
           Return the predicted probabilities for each class'''
        self._load_model()
        image=self._feature_extraction_inception(img)
        self.img=image
        pred=self.model.predict(image)
        # pred=np.round(pred,3).reshape(4,)
        pred=np.round(pred,2).reshape(2,) #只有兩個class
        # return pred[0],pred[1],pred[2],pred[3]
        return pred[0],pred[1]

def model_check():
    label = ['Not','Yes']
    Predlist = list()
    model = inception_retrain()
    pred1,pred2 = model.predict(file_path)
    Predlist.append(pred1)
    Predlist.append(pred2)
    print(Predlist) #測試用
    return label[Predlist.index(max(Predlist))] #取得a最大的index

a = model_check()
print(a)