import streamlit as st
import pickle
import numpy as np


with open('breast_cancer_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


def predict_cancer(input_data):
    input_data_reshaped = np.asarray(input_data).reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return prediction[0]


def main():
    st.set_page_config(layout="centered")
    st.title('Breast Cancer Prediction')
    st.markdown('Enter the values for the input features:')
    feature_names = [
        'Mean Radius',
        'Mean Texture',
        'Mean Perimeter',
        'Mean Area',
        'Mean Smoothness',
        'Mean Compactness',
        'Mean Concavity',
        'Mean Concave Points',
        'Mean Symmetry',
        'Mean Fractal Dimension',
        'Radius Error',
        'Texture Error',
        'Perimeter Error',
        'Area Error',
        'Smoothness Error',
        'Compactness Error',
        'Concavity Error',
        'Concave Points Error',
        'Symmetry Error',
        'Fractal Dimension Error',
        'Worst Radius',
        'Worst Texture',
        'Worst Perimeter',
        'Worst Area',
        'Worst Smoothness',
        'Worst Compactness',
        'Worst Concavity',
        'Worst Concave Points',
        'Worst Symmetry',
        'Worst Fractal Dimension'
    ]
    col1, col2 = st.columns([2, 1])
    with col1:
        input_data = []
        for feature_name in feature_names:
            input_value = st.number_input(feature_name, step=0.01,)
            input_data.append(float(input_value))

        if st.button('Predict'):
            prediction = predict_cancer(input_data)
            if prediction == 0:
                st.error('The breast cancer is Malignant')
            else:
                st.success('The breast cancer is Benign')
    with col2:
        st.markdown('**MALIGNANT**')
        st.write('Breast cancer is a type of cancer that can spread to other parts of the body. '
                 'It requires prompt medical attention and treatment.')

        st.markdown('**BENIGN**')
        st.write('Benign breast tumors are not cancerous. They do not spread to other parts of the body '
                 'and are generally not life-threatening.')

        st.markdown('**NOTE:** The prediction provided by this app is for informational purposes only. '
                    'It is not a substitute for professional medical advice or diagnosis.')

    st.markdown('<br>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
