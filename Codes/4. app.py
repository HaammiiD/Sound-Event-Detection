
import keras
import librosa
import numpy as np
import streamlit as st

st.title("Voice Detection")
st.header("UrbanSound Dataset")
st.text("Upload a sound file")

def voice_classification(sound, weights_file):
    # Load the model
    model = keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    audio_data, sample_rate = librosa.load(sound, res_type='kaiser_fast') 
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
    mfccsscaled = np.mean(mfccs.T,axis=0)   

    #turn into a numpy array
    sound_array = np.array([mfccsscaled])

    # run the inference
    predicted_vector = model.predict(sound_array)
    
    return np.argmax(predicted_vector)



uploaded_file = st.file_uploader("Choose a sound ...")
if uploaded_file is not None:
    #audio_data, sample_rate = librosa.load(uploaded_file, res_type='kaiser_fast') 
    st.audio(uploaded_file)
    st.write("")
    st.write("Classifying...")
    label = voice_classification(uploaded_file, 'myMLPmodel.hdf5')
    
    if label == 0:
        category = "Air Conditioner"
    elif label == 1:
        category = "car_horn"
    elif label == 2:
        category = "children_playing"
    elif label == 3:
        category = "dog_bark"
    elif label == 4:
        category = "Drilling"
    elif label == 5:
        category = "engine_idling"
    elif label == 6:
        category = "gun_shot"
    elif label == 7:
        category = "jackhammer"
    elif label == 8:
        category = "siren"
    elif label == 9:
        category = "street_music "
    else:
        category = None

        
    st.write("The predicted class is:", category, '\n')
    st.write("The predicted number of class is:", label, '\n') 
        
