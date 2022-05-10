from unittest import result
from nbformat import NotebookNode
import streamlit as st
from PIL import Image 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import tensorflow as tf
import keras
import tensorflow_hub as hub



hide_streamlit_style =  """
            #MainMenu {visibility : hidden:}
            footer {visibility : hidden;}
                """


st.markdown(hide_streamlit_style, unsafe_allow_html = True)

st.title("Tomato Leaf Disease Prediction")

def main() :
    file_uploaded = st.file_uploader("Chooose an image...", type = 'jpg')
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        st.write("Uploaded Image.")
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        st.pyplot(figure)
        result, confidence = predict_class(image)
        st.write(f'Prediction : {result}')
        st.write(f'Confidence :{confidence}')


def predict_class(image) :
    with st.spinner('Loading Model...'):
        classifier_model = keras.models.load_model(r'tomatoes.h5', compile = False)



    shape = ((256,256,3))
    model = keras.Sequential([hub.KerasLayer(classifier_model, input_shape = shape)])
    test_image = image.resize((256,256))
    test_image = keras.preprocessing.image.img_to_array(test_image)
    test_image /= 255.0
    test_image = np.expand_dims(test_image, axis =0)
    class_name = ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
                'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite',
                'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
                'Tomato_healthy']

    prediction = model.predict(test_image)
    confidence = round(100 * (np.max(prediction[0])),2)
    final_pred = class_name[np.argmax(prediction)]

    return final_pred, confidence




footer = """
        a:link, a:visited{
            color: white;
            background-color: transparent;
            text-decoration: None;
        }
        
        

        a:hover, a:active {
            color: red;
            background-color: transparent;
            text-decoration: None
        }


        .fotter {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: transparent;
            color: black;
            text-align: centre;
        }
        """

st.markdown(footer, unsafe_allow_html = True)


if __name__=='__main__':
    main()