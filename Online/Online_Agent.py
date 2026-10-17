import io
import streamlit as st
from PIL import Image
import Agent_Widgets_Online

st.set_page_config(page_title="Online Smart Agent", layout="centered")
st.title("Online Smart Agent")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "✍️ Text Generation", "🖼️ Image Generation",
    "🔊 Text to Speech", "🌐 Translator Text", "📊 Text Classification"
])

# Text Generation

with tab1:
    st.header("Text Generation")
    system_rules = st.text_input("System Instructions / Personality :", key="on_sys_prompt")
    user_prompt = st.text_area("Your Input Prompt :", placeholder="Type Prompt", key="on_user_prompt")
    text_file_name = st.text_input("Text Output File Name ( Text ) :", placeholder="Text File Name")
    
    if st.button("Submit Prompt", key="online_text_execution_btn"):
        if user_prompt.strip():
            with st.spinner("Gemini is Generating Response ..."):
                st.session_state.online_text_res = Agent_Widgets_Online.Text(user_prompt, system_rules)
        else: st.warning("Please Enter a Valid Prompt")
    
    if "online_text_res" in st.session_state:
        st.subheader("Response Output :")
        st.write(st.session_state.online_text_res)
        f_name = text_file_name.strip() if text_file_name.strip() else "cloud_inference_output"
        st.download_button("Export your Response", data=st.session_state.online_text_res, file_name=f"{f_name}.txt", mime="text/plain", key="on_txt_dl_btn")

# Image Generation

with tab2:
    st.header("Image Generation")
    image_prompt = st.text_area("Your Image Descriptipn :", placeholder="Describe Details", key="on_img_prompt")
    img_format = st.selectbox("Output Format", ["PNG", "JPEG"], key="on_img_fmt_select")
    img_file_name = st.text_input("Image Output File Name ( Image ) :", placeholder="Image File Name")
    
    if st.button("Generate Image", key="online_img_execution_btn"):
        if image_prompt.strip():
            with st.spinner("Gemini is Generating Image ..."):
                img_output = Agent_Widgets_Online.Picture(image_prompt, img_format.lower())
                if isinstance(img_output, Image.Image):
                    st.session_state.online_img_res = img_output
                    st.session_state.online_img_fmt = img_format.lower()
                    st.success("Successfully Completed your Image!")
                else: st.error(img_output)

    if "online_img_res" in st.session_state:
        st.markdown("<style>img { border: 3px solid #000000; border-radius: 5px; }</style>", unsafe_allow_html=True)
        st.image(st.online_img_res, caption=image_prompt, use_container_width=True)
        img_buffer = io.BytesIO()
        save_format = "JPEG" if st.online_img_fmt in ["jpeg", "jpg"] else "PNG"
        st.session_state.online_img_res.save(img_buffer, format=save_format)
        f_name = img_file_name.strip() if img_file_name.strip() else "cloud_rendered_art"
        st.download_button(f"Export your Image", data=img_buffer.getvalue(), file_name=f"{f_name}.{save_format.lower()}", mime=f"image/{save_format.lower()}", key="on_img_dl_btn")

# Text to Speech Converter 

with tab3:
    st.header("Text Speech Converter")
    tts_text_prompt = st.text_area("Your Target String :", placeholder="Type your String", key="on_tts_input_box")
    tts_file_name = st.text_input("Speech Output File Name ( Voice Audio ) :", placeholder="Speech File Name")
    if st.button("Generate Speech File", key="online_tts_execution_btn"):
        if tts_text_prompt.strip():
            with st.spinner("Cloud engine initializing vocal matrices..."):
                st.session_state.online_tts_res = Agent_Widgets_Online.Text_To_Speech(tts_text_prompt)
    
    if "online_tts_res" in st.session_state:
        st.subheader("Listen to your Speech File :")
        st.write(st.session_state.online_tts_res)
        f_name = tts_file_name.strip() if tts_file_name.strip() else "cloud_vocal_script"
        st.download_button("Export your Speech File", data=st.session_state.online_tts_res, file_name=f"{f_name}.txt", mime="text/plain", key="on_tts_dl_btn")

# Text Translation

with tab4:
    st.header("Text Translation")
    col1, col2 = st.columns(2)
    with col1: src_lang = st.selectbox("Your Source Language :", ["English", "French", "Spanish", "German"], key="on_src_dropdown")
    with col2: target_lang = st.selectbox("Your Target Language :", ["German", "French", "Spanish", "English"], key="on_dest_dropdown")
    translation_input = st.text_area("Your Target Text :", placeholder="Enter your Target Text", key="on_trans_text_area")
    trans_file_name = st.text_input("Translated Output File Name ( Translation ) :", placeholder="Translated Text File Name")
    
    if st.button("Translate Text", key="online_trans_execution_btn"):
        if translation_input.strip() and src_lang != target_lang:
            with st.spinner("Gemini is Translating Text ..."):
                st.session_state.online_trans_res = Agent_Widgets_Online.Translator_Text(translation_input, src_lang, target_lang)
    
    if "online_trans_res" in st.session_state:
        st.subheader("Translated Output :")
        st.write(st.session_state.online_trans_res)
        f_name = trans_file_name.strip() if trans_file_name.strip() else "cloud_translated_text"
        st.download_button("Export your Translated Text", data=st.session_state.online_trans_res, file_name=f"{f_name}.txt", mime="text/plain", key="on_trans_dl_btn")

with tab5:
    st.header("Text Classification")
    classify_input = st.text_area("Your Target Sentence :", placeholder="Type your Sentence", key="on_classify_input_box")
    classify_file_name = st.text_input("Classification Output File Name ( Classification ) :", placeholder="Classification Output File Name")
    
    if st.button("Classify Text", key="online_classify_execution_btn"):
        if classify_input.strip():
            with st.spinner("Gemini is Classifying Text ..."):
                st.session_state.online_class_res = Agent_Widgets_Online.Classify_Text(classify_input)
    
    if "online_class_res" in st.session_state:
        st.subheader("Classification Output :")
        st.markdown(st.session_state.online_class_res)
        f_name = classify_file_name.strip() if classify_file_name.strip() else "cloud_sentiment_analysis"
        st.download_button("Export your Classification Analysis File", data=st.session_state.online_class_res.replace("**", ""), file_name=f"{f_name}.txt", mime="text/plain", key="on_class_dl_btn")
