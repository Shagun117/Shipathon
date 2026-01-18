#first project
import streamlit as st
import subprocess
import os
import json 
import tempfile
import pandas as pd
import sys
import time


WA_TXT_PATH = "chat_W.txt"
# WA_TRANSLATED_CSV = "whatsapp_translated.csv"
# MAPPING_JSON = "character_user_mapping.json"

st.set_page_config(
    page_title="Zindagi na milegei dubara",
    layout="centered"
)

st.title("🎬 Zindagi na milegei dubara × WhatsApp Character Mapper ")

st.markdown("""
<style>
.stApp {
    background-image: url("https://wallpapercave.com/wp/wp7115055.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
Upload a WhatsApp chat export (`.txt`). 
Pipeline:
1. Parse WhatsApp chat  
2. Translate (if enabled)  
3. Embed users  
4. Match movie characters  
""")

uploaded_file = st.file_uploader(
    "Upload WhatsApp chat (.txt)",
    type=["txt"]


)

# 🔑 PLACEHOLDER — THIS IS THE KEY
output_placeholder = st.empty()

import subprocess
if uploaded_file:
    with open(WA_TXT_PATH, "wb") as f:
        f.write(uploaded_file.read())

    st.success("Chat uploaded")
    st.subheader("WAIT WAIT...AA RH HAI ANSWER:)")
    with st.spinner("Running wa_parse.py..."):
        #ubprocess.run(
         #   [sys.executable, "first.py"],
          #  check=True
        
        #)

        try:
            result = subprocess.run(
                ["python", "first.py"],
                check=True,
                capture_output=True,
                text=True
            )
            

            st.success("Parsing completed")
            st.subheader("Script Output")
            st.code(result.stdout)

            
            
            
            
            
            output = st.empty()
            with output.container():
                st.success("Parsing completed ✅")
                st.subheader("Script Output")
                st.code(result.stdout)
                st.balloons()   # 🎈 SAFE 🎈

           

            
            

            # Optional celebration
            st.balloons()
            time.sleep(1.5)
            st.markdown(""")
            <script>
            window.scrollTo({
                top: document.body.scrollHeight,
                behavior: 'smooth'
            });
            </script>
            """, unsafe_allow_html=True)

        except subprocess.CalledProcessError as e:
            st.error("Parsing failed")
            st.subheader("Error Output")
            st.code(e.stderr)
   



        


