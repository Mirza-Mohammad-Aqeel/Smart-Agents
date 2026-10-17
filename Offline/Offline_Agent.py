import io
import os
import streamlit as st
from PIL import Image
import Agent_Widgets_Offline
from transformers import pipeline

st.set_page_config(page_title="Offline Smart Agent", layout="centered")
st.title("Offline Smart Agent")

script_dir = os.path.dirname(os.path.abspath(os.getcwd()))
qwen_model_path = os.path.join(script_dir, "Models", "Qwen Model")
cardiff_model_path = os.path.join(script_dir, "Models", "Cardiff Model")

# Cache

@st.cache_resource
def load_text_pipeline():
    return pipeline("text-generation", model=qwen_model_path)

@st.cache_resource
def load_classifier_pipeline():
    return pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=-1)

@st.cache_resource
def load_tts_pipeline():
    return pipeline("text-to-speech", model="facebook/mms-tts-eng")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "✍️ Text Generation", "🖼️ Image Generation",
    "🔊 Text to Speech", "🌐 Translator Text", "📊 Text Classification"
])

# Text Generation

with tab1:
    st.header("Text Generation")
    system_rules = st.text_input("System Instructions / Personality :", key="off_sys_prompt")
    user_prompt = st.text_area("Your Input Prompt :", placeholder="Type Prompt", key="off_user_prompt")
    text_file_name = st.text_input("Text Output File Name ( Text ) :", placeholder="Text File Name")
    
    if st.button("Submit Prompt", key="offline_text_execution_btn"):
        if user_prompt.strip():
            with st.spinner("Model is Generating Response ..."):
                text_pipe = load_text_pipeline()
                st.session_state.off_text_res = Agent_Widgets_Offline.Text(user_prompt, system_rules, text_pipe)
        else: st.warning("Please Enter a Valid Prompt")
        
    if "off_text_res" in st.session_state:
        st.subheader("Response Output :")
        st.write(st.session_state.off_text_res)
        f_name = text_file_name.strip() if text_file_name.strip() else "local_inference_output"
        st.download_button(label="Export your Response", data=st.session_state.off_text_res, file_name=f"{f_name}.txt", mime="text/plain", key="off_text_dl_btn")

# Image Generation

with tab2:
    st.header("Image Generation | In Development")
    image_prompt = st.text_area("Your Image Description :", placeholder="Describe Details", key="off_img_box")
    img_format = st.selectbox("Output Format", ["PNG", "JPEG"], key="off_img_fmt_select")
    img_file_name = st.text_input("Image Output File Name ( Image ) :", placeholder="Image File Name")
    
    if st.button("Generate Image", key="offline_img_execution_btn"):
        if image_prompt.strip():
            img_output = Agent_Widgets_Offline.Picture(image_prompt, img_format.lower())
            if isinstance(img_output, Image.Image):
                st.session_state.off_img_res = img_output
                st.session_state.off_img_format_type = img_format.lower()
                st.success("Successfully Completed your Image!")
                
    if "off_img_res" in st.session_state:
        st.markdown("<style>img { border: 3px solid #000000; border-radius: 5px; }</style>", unsafe_allow_html=True)
        st.image(st.session_state.off_img_res, caption=image_prompt, use_container_width=True)
        img_buffer = io.BytesIO()
        save_format = "JPEG" if st.session_state.off_img_format_type in ["jpeg", "jpg"] else "PNG"
        st.session_state.off_img_res.save(img_buffer, format=save_format)
        f_name = img_file_name.strip() if img_file_name.strip() else "local_rendered_art"
        st.download_button(label="Export your Image", data=img_buffer.getvalue(), file_name=f"{f_name}.{save_format.lower()}", mime=f"image/{save_format.lower()}", key="off_img_dl_btn")

# Text to Speech Converter

with tab3:
    st.header("Text Speech Converter")
    tts_prompt = st.text_area("Your Target String :", placeholder="Type your String", key="off_tts_input_box")
    tts_file_name = st.text_input("Speech Output File Name ( Voice Audio ) :", placeholder="Speech File Name")
    
    if st.button("Generate Speech File", key="offline_tts_execution_btn"):
        if tts_prompt.strip():
            with st.spinner("Model is Generating Speech File ..."):
                tts_pipe = load_tts_pipeline()
                st.session_state.off_tts_res = Agent_Widgets_Offline.Text_To_Speech(tts_prompt, tts_pipe)
                st.success("Successfully Generated your Speech File!")
                
    if "off_tts_res" in st.session_state and not isinstance(st.session_state.off_tts_res, str):
        st.subheader("Listen to your Speech File :")
        st.audio(st.session_state.off_tts_res, format="audio/wav")
        f_name = tts_file_name.strip() if tts_file_name.strip() else "local_voice_audio"
        st.download_button(label="Export your Speech File", data=st.session_state.off_tts_res, file_name=f"{f_name}.wav", mime="audio/wav", key="off_tts_dl_btn")
    elif "off_tts_res" in st.session_state and isinstance(st.session_state.off_tts_res, str):
        st.error(st.session_state.off_tts_res)

# Text Translation

with tab4:
    st.header("Text Translation")
    col1, col2 = st.columns(2)
    with col1: src_lang = st.selectbox("Your Source Language :", ["English", "French", "Spanish", "German"], key="off_src_dropdown")
    with col2: target_lang = st.selectbox("Your Target Language :", ["German", "French", "Spanish", "English"], key="off_dest_dropdown")
    translation_input = st.text_area("Your Target Text :", placeholder="Enter your Target Text", key="off_trans_text_area")
    trans_file_name = st.text_input("Translated Output File Name ( Translation ) :", placeholder="Translated Text File Name")
    
    if st.button("Translate Text", key="offline_trans_execution_btn"):
        if translation_input.strip() and src_lang != target_lang:
            with st.spinner("Model is Translating Text ..."):
                text_pipe = load_text_pipeline()
                st.session_state.off_trans_res = Agent_Widgets_Offline.Translator_Text(translation_input, src_lang, target_lang, text_pipe)
                
    if "off_trans_res" in st.session_state:
        st.subheader("Translated Output :")
        st.write(st.session_state.off_trans_res)
        f_name = trans_file_name.strip() if trans_file_name.strip() else "local_translated_text"
        st.download_button(label="Export your Translated Text", data=st.session_state.off_trans_res, file_name=f"{f_name}.txt", mime="text/plain", key="off_trans_dl_btn")

# Text Classification

with tab5:
    st.header("Text Classification")
    classify_input = st.text_area("Your Target Sentence :", placeholder="Type your Sentence", key="off_classify_input_box")
    classify_file_name = st.text_input("Classified Output File Name ( Classification ) :", placeholder="Classification Output File Name")
    
    if st.button("Classify Text", key="offline_classify_execution_btn"):
        if classify_input.strip():
            with st.spinner("Model is Classifying Text ..."):
                classifier_pipe = load_classifier_pipeline()
                st.session_state.off_class_res = Agent_Widgets_Offline.Classify_Text(classify_input, classifier_pipe)
                st.success("Successfully Classified your Text!")
                
    if "off_class_res" in st.session_state:
        st.subheader("Classification Output :")
        st.write(st.session_state.off_class_res)
        f_name = classify_file_name.strip() if classify_file_name.strip() else "local_sentiment_analysis"
        st.download_button(label="Export your Classification Analysis File", data=st.session_state.off_class_res, file_name=f"{f_name}.txt", mime="text/plain", key="off_class_dl_btn")
