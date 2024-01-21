import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from lime.lime_tabular import LimeTabularExplainer
from lime.lime_text import LimeTextExplainer
from sklearn import set_config
set_config(transform_output='pandas')
#####################################################
@st.cache_resource
def load_tf_dataset(fpath):
    return tf.data.Dataset.load(fpath)
@st.cache_resource
def load_network(fpath):
    return tf.keras.models.load_model(fpath)
    
def predict_decode_deep(X_to_pred, network, lookup_dict):
    if isinstance(X_to_pred, str):
        X = [X_to_pred]
    else:
        X = X_to_pred
    pred_probs = network.predict(X)
    pred_class = convert_y_to_sklearn_classes(pred_probs)
    # Decode label
    class_name = lookup_dict[pred_class[0]]
    return class_name
#####################################################
# Define the load train or test data function with caching
@st.cache_data
def load_Xy_data(fpath):
    return joblib.load(fpath)

@st.cache_resource
def load_ml_model(fpath):
    return joblib.load(fpath)

# load target lookup dict
@st.cache_data
def load_lookup(fpath):
    return joblib.load(fpath)
    
@st.cache_resource
def load_encoder(fpath):
    return joblib.load(fpath)

# Update the function to decode the prediction
def make_prediction(X_to_pred, clf_pipe, lookup_dict):
    # Get Prediction
    pred_class = clf_pipe.predict([X_to_pred])[0]
    # Decode label
    pred_class = lookup_dict[pred_class]
    return pred_class

@st.cache_resource
def get_explainer(class_names = None):
    lime_explainer = LimeTextExplainer(class_names=class_names)
    return lime_explainer
    
def explain_instance(explainer, X_to_pred, predict_func):
    explanation = explainer.explain_instance(X_to_pred, predict_func)
    return explanation.as_html(predict_proba=False)

from sklearn.metrics import classification_report, ConfusionMatrixDisplay
def classification_metrics_streamlit(y_true, y_pred, label='',
                           figsize=(8,4),
                           normalize='true', cmap='Blues',
                           colorbar=False,values_format=".2f",
                                    class_names=None):

    # Get the classification report
    report = classification_report(y_true, y_pred,target_names=class_names)
    
    ## Save header and report
    header = "-"*70
    final_report = "\n".join([header,f" Classification Metrics: {label}", header,report,"\n"])
        
    ## CONFUSION MATRICES SUBPLOTS
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    
    # Create a confusion matrix  of raw counts (left subplot)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                                            normalize=None, 
                                            cmap='gist_gray_r',# Updated cmap
                                            display_labels = class_names, # Added display labels
                                            values_format="d", 
                                            colorbar=colorbar,
                                            ax = axes[0]);
    axes[0].set_title("Raw Counts")
    
    # Create a confusion matrix with the data with normalize argument 
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                                            normalize=normalize,
                                            cmap=cmap, 
                                            values_format=values_format, #New arg
                                            display_labels = class_names, # Added display labels
                                            colorbar=colorbar,
                                            ax = axes[1]);
    axes[1].set_title("Normalized Confusion Matrix")
    
    # Adjust layout and show figure
    fig.tight_layout()

    return final_report, fig
def get_true_pred_labels(model,ds):

    y_true = []
    y_pred_probs = []
    
    # Loop through the dataset as a numpy iterator
    for images, labels in ds.as_numpy_iterator():
        
        # Get prediction with batch_size=1
        y_probs = model.predict(images, batch_size=1, verbose=0)
        # Combine previous labels/preds with new labels/preds
        y_true.extend(labels)
        y_pred_probs.extend(y_probs)
    ## Convert the lists to arrays
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    
    return y_true, y_pred_probs
    
def convert_y_to_sklearn_classes(y, verbose=False):
    # If already one-dimension
    if np.ndim(y)==1:
        if verbose:
            print("- y is 1D, using it as-is.")
        return y
        
    # If 2 dimensions with more than 1 column:
    elif y.shape[1]>1:
        if verbose:
            print("- y is 2D with >1 column. Using argmax for metrics.")   
        return np.argmax(y, axis=1)
    
    else:
        if verbose:
            print("y is 2D with 1 column. Using round for metrics.")
        return np.round(y).flatten().astype(int)
        
def classification_metrics_streamlit_tensorflow(model,X_train=None, y_train=None, 
                                                label='Training Data',
                                    figsize=(6,4), normalize='true',
                                    output_dict = False,
                                    cmap_train='Blues',
                                    cmap_test="Reds",
                                    values_format=".2f", 
                                                class_names = None,
                                    colorbar=False):
    
    ## Check if X_train is a dataset
    if hasattr(X_train,'shuffle'):
        # If it IS a Datset:
        # extract y_train and y_train_pred with helper function
        y_train, y_train_pred = get_true_pred_labels(model, X_train)
    else:
        # Get predictions for training data
        y_train_pred = model.predict(X_train)


     ## Pass both y-vars through helper compatibility function
    y_train = convert_y_to_sklearn_classes(y_train)
    y_train_pred = convert_y_to_sklearn_classes(y_train_pred)
    
    # Call the helper function to obtain regression metrics for training data
    report, conf_mat = classification_metrics_streamlit(y_train, y_train_pred, 
                                                        figsize=figsize,
                                         colorbar=colorbar, cmap=cmap_train, 
                                                        values_format=values_format,label=label,
                                                       class_names=class_names)
    return report, conf_mat


