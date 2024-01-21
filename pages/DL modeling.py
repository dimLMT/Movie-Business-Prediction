import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn import set_config
set_config(transform_output='pandas')
import joblib, json, sys, os
sys.path.append(os.path.abspath("../"))
import functions_for_st as fn
    
# Open the file structure dictionary
with open('config/filepaths.json') as f:
    FPATHS = json.load(f)

st.title("Predicting Movie Review Ratings")
st.subheader('Get predictions')
# Get text to predict from the text input box
X_to_pred = st.text_input(":blue[Enter text to predict here:] :point_down:", 
                          value="I like ...")


# Load the encoder
encoder = fn.load_encoder(FPATHS['data']['ml']['label_encoder'])
# Load the target lookup dictionary
target_lookup = fn.load_lookup(FPATHS['data']['ml']['target_lookup'])
# Create the lime explainer
explainer = fn.get_explainer(class_names = encoder.classes_)


best_network = fn.load_network(FPATHS['models']['deep'])
# Loading train and test ds 
fpath_train_ds = FPATHS['data']['tf']['train_ds']
train_ds = fn.load_tf_dataset(fpath_train_ds)

fpath_test_ds = FPATHS['data']['tf']['test_ds']
test_ds = fn.load_tf_dataset(fpath_test_ds)
st.divider()
if st.button("Get prediction."):
        pred_class_name = fn.predict_decode_deep(X_to_pred, best_network, target_lookup)
        st.markdown(f"##### Neural Network Predicted category:  {pred_class_name}")
else: 
    st.empty()
###############################    
st.divider()
st.subheader('Evaluate Neural Network')

## To place the 3 checkboxes side-by-side
col1,col2,col3 = st.columns(3)
show_train = col1.checkbox("Show training data.", value=True)
show_test = col2.checkbox("Show test data.", value=True)

if st.button("Show Neural Network evaluation."):
    with st.spinner("Please wait while the neural network is evaluated..."):
        if show_train:
            # Display training data results
            report_str, conf_mat = fn.classification_metrics_streamlit_tensorflow(best_network,label='Training Data',
                                                                               X_train=train_ds,
                                                                               )
            st.text(report_str)
            st.pyplot(conf_mat)
            st.text("\n\n")
    
        if show_test: 
            # Display training data results
            report_str, conf_mat = fn.classification_metrics_streamlit_tensorflow(best_network,label='Test Data',
                                                                               X_train=test_ds,
                                                                               )
            st.text(report_str)
            st.pyplot(conf_mat)
            st.text("\n\n")
  
else:
    st.empty()

