import time
import streamlit as st
import numpy as np
from PIL import Image
import urllib.request
from utils import *
import os

labels = gen_labels()

html_temp = '''
    <div style =  padding-bottom: 20px; padding-top: 20px; padding-left: 5px; padding-right: 5px">
    <center><h1>Plant Disease Classification</h1></center>
    
    </div>
    '''

st.markdown(html_temp, unsafe_allow_html=True)
html_temp = '''
    <div>
    <h2></h2>
    <center><h3>Please upload plant leaf image to find its disease category</h3></center>
    </div>
    '''
# st.set_option('deprecation.showfileUploaderEncoding', False)
st.markdown(html_temp, unsafe_allow_html=True)
opt = st.selectbox("How do you want to upload the image for classification?\n", ('Please Select', 'Upload image via link', 'Upload image from device'))
if opt == 'Upload image from device':
    file = st.file_uploader('Select', type = ['jpg', 'png', 'jpeg'])
    # st.set_option('deprecation.showfileUploaderEncoding', False)
    if file is not None:
        image = Image.open(file)

elif opt == 'Upload image via link':

  try:
    img = st.text_input('Enter the Image Address')
    image = Image.open(urllib.request.urlopen(img))
    
  except:
    if st.button('Submit'):
      show = st.error("Please Enter a valid Image Address!")
      time.sleep(4)
      show.empty()

try:
  if image is not None:
    st.image(image, width = 128, caption = 'Uploaded Image')
    if st.button('Predict'):
        prepare_img = preprocess(image)

        label_array = ['scab', 'black rot', 'cedar rust', 'healthy', 'healthy', 'healthy', 'powder mildew', 'spot grey leaf', 'common rust', 'healthy', 'leaf blight', 'black rot', 'black measles', 'healthy', 'leaf blight', 'citrus greening', 'bacterial spot', 'healthy', 'bacterial spot', 'healthy', 'early blight', 'healthy', 'late blight', 'healthy', 'healthy', 'powder mildew', 'healthy', 'leaf scorch', 'bacterial spot', 'early blight', 'healthy', 'late blight', 'leaf mold', 'septoria leaf spot', 'two spotted spider mite', 'target spot', 'mosaic virus', 'yellow leaf curl virus']           

        model = model_arc()
        model.load_weights("../model/plantmodeltype2.h5")
        

        predictions = model.predict(prepare_img[np.newaxis, ...])
        st.info('Hey! The uploaded image has been classified as "{}" '.format(label_array[np.argmax(predictions)]))
        st.info('Information about "{}":'.format(label_array[np.argmax(predictions)]))
       
        new_dir = "../info/"
        os.chdir(new_dir)
        
        with open("{}.txt".format(label_array[np.argmax(predictions)]), "r") as file:
            info = file.read()
        st.info(info)
except Exception as e:
  st.info(e)
  pass
