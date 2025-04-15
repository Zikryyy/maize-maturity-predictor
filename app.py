import streamlit as st
import requests
import numpy as np
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt
from fastapi import FastAPI, Request
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import joblib
from threading import Thread
import nest_asyncio
from datetime import datetime

# --- FastAPI Backend ---
app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
try:
    model = joblib.load("rf_model_maize_maturity.pkl")
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    model = None


@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    try:
        r = float(data["R"])
        g = float(data["G"])
        b = float(data["B"])
        temp = float(data["temperature"])
        hum = float(data["humidity"])

        features = np.array([[r, g, b, temp, hum]])
        prediction = model.predict(features)
        result = "Mature" if prediction[0] == 1 else "Immature"
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}


def run_fastapi():
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)


# --- Streamlit Frontend ---
def main():
    st.set_page_config(
        page_title="Blue Maize Maturity Predictor",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # High-contrast minimalist theme
    st.markdown("""
        <style>
            /* Base styles */
            body, .stApp {
                font-family: 'Arial', sans-serif;
                background-color: white !important;
                color: black !important;
            }

            /* Title */
            h1 {
                color: #1a56db !important;
                text-align: center;
                margin-bottom: 1.5rem;
                font-size: 2rem;
                font-weight: 700;
            }

            /* All text elements - force black */
            .stRadio, .stRadio label, 
            .stFileUploader, .stFileUploader label, 
            .stFileUploader small,
            .stNumberInput label, 
            .stSlider label,
            .stSuccess, .stSuccess div {
                color: black !important;
                font-weight: 500 !important;
            }

            /* Headings */
            h2, h3, h4 {
                color: #1e40af !important;
                font-weight: 600;
            }

            /* File uploader box */
            .stFileUploader>section {
                border: 2px dashed #1a56db !important;
                background: white !important;
            }

            /* Success message box */
            .stSuccess {
                background-color: #f0f4f8 !important;
                border-left: 4px solid #1a56db !important;
                font-weight: 600;
            }

            /* Make sure all text is visible */
            div[data-testid="stMarkdownContainer"] p,
            div[data-testid="stMarkdownContainer"] div {
                color: black !important;
            }

            /* Button styling */
            .stButton>button {
                background-color: #2563eb !important;
                color: white !important;
                border: none;
                border-radius: 8px;
                padding: 0.75rem 1.5rem;
                width: 100%;
                font-weight: 500;
            }

            /* History cards */
            .history-card {
                padding: 1rem;
                margin-bottom: 1rem;
                border-radius: 8px;
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-left: 4px solid #2563eb;
            }
        </style>
    """, unsafe_allow_html=True)

    # App title
    st.markdown("# Blue Maize Maturity Predictor")

    HISTORY_FILE = "prediction_history.csv"

    # Load history
    if "history" not in st.session_state:
        if os.path.exists(HISTORY_FILE):
            st.session_state.history = pd.read_csv(HISTORY_FILE).to_dict(orient="records")
        else:
            st.session_state.history = []

    # Input mode
    st.markdown("## Input Method")
    mode = st.radio("", ["Manual RGB Entry", "Upload Image for RGB"],
                    horizontal=True, label_visibility="collapsed")

    if mode == "Manual RGB Entry":
        col1, col2, col3 = st.columns(3)
        with col1:
            r = st.number_input("Red (R)", min_value=0, max_value=255, value=100)
        with col2:
            g = st.number_input("Green (G)", min_value=0, max_value=255, value=100)
        with col3:
            b = st.number_input("Blue (B)", min_value=0, max_value=255, value=100)
    else:
        uploaded_file = st.file_uploader(
            "Drag and drop file here (Limit 200MB - JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Process image and show RGB values
            resized = image.resize((100, 100))
            img_np = np.array(resized)
            avg_color = img_np.mean(axis=(0, 1)).astype(int)
            r, g, b = int(avg_color[0]), int(avg_color[1]), int(avg_color[2])

            # High-contrast success message
            st.success(f"**Extracted RGB values → R: {r}, G: {g}, B: {b}**")

            # RGB heatmap
            st.markdown("## Color Channels Analysis")
            fig, axs = plt.subplots(1, 3, figsize=(12, 3))
            cmap_labels = ['Red', 'Green', 'Blue']
            for i, ax in enumerate(axs):
                ax.imshow(img_np[:, :, i], cmap='Reds' if i == 0 else 'Greens' if i == 1 else 'Blues')
                ax.set_title(cmap_labels[i], color='#1e40af', fontsize=12)
                ax.axis("off")
            plt.tight_layout()
            st.pyplot(fig)

    # Environmental data
    st.markdown("## Environmental Conditions")
    col1, col2 = st.columns(2)
    with col1:
        temp = st.slider("Temperature (°C)", 20.0, 35.0, 25.0)
    with col2:
        hum = st.slider("Humidity (%)", 30.0, 80.0, 50.0)

    # Predict button
    if st.button("Predict Maturity", type="primary"):
        data = {
            "R": r, "G": g, "B": b,
            "temperature": temp,
            "humidity": hum
        }
        try:
            with st.spinner("Analyzing..."):
                response = requests.post("http://localhost:8000/predict", json=data)
                result = response.json()

            if "prediction" in result:
                prediction = result["prediction"]
                st.success(f"Prediction: **{prediction}**")

                entry = {
                    "R": r, "G": g, "B": b,
                    "Temp": temp, "Humidity": hum,
                    "Prediction": prediction,
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Add timestamp
                }
                st.session_state.history.append(entry)
                pd.DataFrame(st.session_state.history).to_csv(HISTORY_FILE, index=False)
            else:
                st.error(f"Error: {result.get('error', 'Unknown error')}")
        except Exception as e:
            st.error(f"Connection failed: {str(e)}")

    # History section
    if st.session_state.history:
        st.markdown("## Prediction History")
        for i, entry in enumerate(reversed(st.session_state.history), 1):
            with st.container():
                st.markdown(f"""
                <div class="history-card">
                    <div style='font-weight: 600; color: #1e40af;'>Prediction #{i}</div>
                    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-top: 0.5rem;'>
                        <div><b>Time:</b> {entry['Timestamp']}</div>
                        <div><b>Result:</b> {entry['Prediction']}</div>
                        <div><b>RGB:</b> {entry['R']}, {entry['G']}, {entry['B']}</div>
                        <div><b>Temp:</b> {entry['Temp']}°C</div>
                        <div><b>Humidity:</b> {entry['Humidity']}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Export CSV
        df = pd.DataFrame(st.session_state.history)
        st.download_button(
            label="Download History as CSV",
            data=df.to_csv(index=False),
            file_name="maize_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )

if __name__ == "__main__":
    # Start FastAPI in a separate thread
    Thread(target=run_fastapi, daemon=True).start()

    # Start Streamlit
    main()