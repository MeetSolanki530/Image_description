import os
import pandas as pd
import streamlit as st
import base64
import ollama  # Import the Ollama client
import subprocess


# Path to check if model is downloaded
MODEL_PATH = "llava_model_downloaded.txt"

def download_model():
    if not os.path.exists(MODEL_PATH):
        try:
            subprocess.run(['ollama', 'download', 'llava:7b'], check=True)
            # Create a file to indicate the model has been downloaded
            with open(MODEL_PATH, 'w') as f:
                f.write("Model downloaded.")
        except subprocess.CalledProcessError as e:
            st.error(f"Failed to download model: {e}")
            st.stop()

# Ensure the llava:7b model is downloaded
download_model()

st.set_page_config(
    page_title="Image to Text",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("AI Image Description Generator 🤖✍️")


def to_base64(uploaded_file):
    file_buffer = uploaded_file.read()
    b64 = base64.b64encode(file_buffer).decode()
    return f"data:image/png;base64,{b64}"


with st.sidebar:
    st.title("Upload Your Images")
    st.session_state.images = st.file_uploader(label=" ", accept_multiple_files=True)


def generate_df():
    current_df = pd.DataFrame(
        {
            "image_id": [img.name for img in st.session_state.images],  # Use name as unique identifier
            "image": [to_base64(img) for img in st.session_state.images],
            "name": [img.name for img in st.session_state.images],
            "description": [""] * len(st.session_state.images),
        }
    )

    if "df" not in st.session_state:
        st.session_state.df = current_df
        return

    new_df = pd.merge(current_df, st.session_state.df, on=["image_id"], how="outer", indicator=True)
    new_df = new_df[new_df["_merge"] != "right_only"].drop(columns=["_merge", "name_y", "image_y", "description_x"])
    new_df = new_df.rename(columns={"name_x": "name", "image_x": "image", "description_y": "description"})
    new_df["description"] = new_df["description"].fillna("")

    st.session_state.df = new_df


def render_df():
    st.data_editor(
        st.session_state.df,
        column_config={
            "image": st.column_config.ImageColumn(
                "Preview Image", help="Image preview", width=100
            ),
            "name": st.column_config.Column("Name", help="Image name", width=200),
            "description": st.column_config.Column(
                "Description", help="Image description", width=800
            ),
        },
        hide_index=True,
        height=500,
        column_order=["image", "name", "description"],
        use_container_width=True,
    )


def generate_description(image_base64):
    response = ollama.chat(model='llava:7b', messages=[
        {
            'role': 'user',
            'content': 'Tell me about the image?',
        },
    ])
    return response['message']['content']


def update_df():
    indexes = st.session_state.df[st.session_state.df["description"] == ""].index
    for idx in indexes:
        description = generate_description(st.session_state.df.loc[idx, "image"])
        st.session_state.df.loc[idx, "description"] = description


if st.session_state.images:
    generate_df()

    st.text_input("Prompt", value="What's in this image?", key="text_prompt")

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Generate Image Descriptions", use_container_width=True):
            update_df()
    
    with col2:    
        st.download_button(
            "Download descriptions as CSV",
            st.session_state.df.drop(['image', "image_id"], axis=1).to_csv(index=False),
            "descriptions.csv",
            "text/csv",
            use_container_width=True
        )

    render_df()
