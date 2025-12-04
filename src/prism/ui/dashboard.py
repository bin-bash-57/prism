import streamlit as st
import requests
import pandas as pd
from PIL import Image


API_URL = "http://localhost:8080/search/similar"

st.set_page_config(page_title="Prism Vision Search", layout="wide")

st.title("Prism: Smart Video Search")
st.markdown("""
**System Status:** Online | **Infrastructure:** Kafka + Triton + Milvus
""")
st.divider()


with st.sidebar:
    st.header("Search Query")
    uploaded_file = st.file_uploader("Upload an object to find...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Query Object", use_column_width=True)


if uploaded_file:
    if st.button("🔍 Run Semantic Search", type="primary"):
        with st.spinner("Analyzing video history (Triton Inference)..."):
            try:
                uploaded_file.seek(0)
                files = {"file": uploaded_file}
                params = {"top_k": 10}

                response = requests.post(API_URL, files=files, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    matches = data.get("matches", [])
                    
                    if not matches:
                        st.warning("No matches found.")
                    else:
                        st.success(f"Found {len(matches)} occurrences!")

                        df = pd.DataFrame(matches)
                        df['timestamp'] = df['timestamp'].apply(lambda x: f"{x:.2f} seconds")
                        df['score'] = df['score'].apply(lambda x: f"{x:.4f} (L2 Dist)")
                        df = df.rename(columns={
                            "timestamp": "Time in Video",
                            "frame_id": "Frame ID",
                            "score": "Confidence Score"
                        })
                        
                        st.dataframe(
                            df, 
                            hide_index=True, 
                            use_container_width=True,
                            column_config={
                                "Time in Video": st.column_config.TextColumn(help="Exact moment the object appeared"),
                            }
                        )
                        
                else:
                    st.error(f"API Error: {response.text}")

            except Exception as e:
                st.error(f"Connection Failed. Is the API running? Error: {e}")

else:
    st.info("Upload an image in the sidebar to start searching your video.")