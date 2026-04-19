import streamlit as st
import numpy as np
import pickle

with open("iris_dataset.pkl",'rb') as f:
    model = pickle.load(f)

st.title("Iris Flower Predication")


speal_length = st.slider("speal length(cm)",4.0,8.0)
speal_width = st.slider("speal width(cm)",1.0,8.0)
petal_length = st.slider("petal length(cm)",4.0,8.0)
petal_width = st.slider("petal width(cm)",1.0,8.0)


if st.button("prediction"):
    input_data = np.array([[speal_length,speal_width,petal_length,petal_width]])
    prediction = model.predict(input_data)
    species = ['Iris-setosa','Iris-versicolor','Iris-virginica']
    st.success(f"Predicted Iris Species:{species[prediction[0]]}")