import streamlit as st
import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from funasr import AutoModel
import pandas as pd

# Create model
model = AutoModel(
    model="paraformer-zh",  
    vad_model="fsmn-vad",   
    punc_model="ct-punc",   
    spk_model="cam++"       
)

# Title
st.title("Van An Duong Voice Recognition")

# Upload
audio_file = st.file_uploader("Upload your file", type=["wav", "mp3"])

if audio_file is not None:
    # Save audio file
    audio_path = f"/tmp/{audio_file.name}"
    with open(audio_path, "wb") as f:
        f.write(audio_file.getbuffer())
    
    # Processing 
    try:
        st.write("Loading...")
        res = model.generate(input=audio_path, batch_size_s=300)
        
        output_data = []

        # Print result
        for segment in res[0].get('sentence_info', []):
            speaker_id = segment.get('spk', 'N/A')  # Get speaker ID
            text = segment.get('text', 'N/A')  # Text
            start_time = segment.get('start', 'N/A')  
            end_time = segment.get('end', 'N/A')  
            output_data.append({"Start time": start_time, "End time": end_time, "SpeakerID": speaker_id, "Transcript": text})
        
        output_df = pd.DataFrame(output_data)
        st.write("Your result:")
        st.dataframe(output_df)
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.write("Please upload your file")


