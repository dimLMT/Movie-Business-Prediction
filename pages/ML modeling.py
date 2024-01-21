import streamlit as st
import pandas as pd
import numpy as np
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

# Load model from FPATHS
clf_pipe = fn.load_ml_model(FPATHS['models']['ml'])
# Load training data
X_train, y_train = fn.load_Xy_data(fpath=FPATHS['data']['ml']['train'])
# Load testing data
X_test, y_test = fn.load_Xy_data(fpath=FPATHS['data']['ml']['test'])
# Load the encoder
encoder = fn.load_encoder(FPATHS['data']['ml']['label_encoder'])
# Load the target lookup dictionary
target_lookup = fn.load_lookup(FPATHS['data']['ml']['target_lookup'])
# Create the lime explainer
explainer = fn.get_explainer(class_names = encoder.classes_)


# Trigger prediction with a button
if st.button("Get prediction."):
    pred_class = fn.make_prediction(X_to_pred, clf_pipe, lookup_dict=target_lookup)
    st.markdown(f"##### ML Predicted category:  {pred_class}") 
    # Get the Explanation as html and display using the .html component.
    html_explanation = fn.explain_instance(explainer, X_to_pred, clf_pipe.predict_proba)
    components.html(html_explanation, height=400)
else: 
    st.empty()
###############################    
st.divider()
# To place the 3 checkboxes side-by-side
col1,col2,col3 = st.columns(3)
show_train = col1.checkbox("Show training data.", value=True)
show_test = col2.checkbox("Show test data.", value=True)
show_model_params =col3.checkbox("Show model params.", value=False)
if st.button("Show model evaluation."):
    if show_train:
        # Display training data results
        y_pred_train = clf_pipe.predict(X_train)
        report_str, conf_mat = fn.classification_metrics_streamlit(y_train, y_pred_train, label='Training Data')
        st.text(report_str)
        st.pyplot(conf_mat)
        st.text("\n\n")
    if show_test: 
        # Display the trainin data resultsg
        y_pred_test = clf_pipe.predict(X_test)
        report_str, conf_mat = fn.classification_metrics_streamlit(y_test, y_pred_test, cmap='Reds',label='Test Data')
        st.text(report_str)
        st.pyplot(conf_mat)
        st.text("\n\n")
        
    if show_model_params:
        # Display model params
        st.markdown("####  Model Parameters:")
        st.write(clf_pipe.get_params())
else:
    st.empty()
