import streamlit as st
import requests
import numpy as np
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt
from fastapi import FastAPI, Request, JSONResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import joblib
from threading import Thread
import nest_asyncio
from socket import socket, AF_INET, SOCK_STREAM
import random
import signal
import atexit

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
@st.cache_resource
def load_model():
    try:
        return joblib.load("rf_model_maize_maturity.pkl")
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None


model = load_model()


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
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )


def find_available_port(start_port=8000, end_port=9000):
    """Find random available port in range"""
    for port in range(start_port, end_port):
        with socket(AF_INET, SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except:
                continue
    raise OSError("No available ports found")


def run_fastapi():
    port = find_available_port()
    st.session_state['api_port'] = port
    print(f"🌐 API running on port {port}")
    nest_asyncio.apply()

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        timeout_keep_alive=30
    )
    server = uvicorn.Server(config)
    server.run()


# --- Streamlit Frontend ---
def main():
    st.set_page_config(
        page_title="Blue Maize Maturity Predictor",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # Initialize session state
    if 'api_port' not in st.session_state:
        st.session_state.api_port = 8001  # Default fallback

    # High-contrast minimalist theme
    st.markdown("""
        <style>
            /* Your existing styles */
            body, .stApp {
                font-family: 'Arial', sans-serif;
                background-color: white !important;
                color: black !important;
            }
            /* ... (keep all your existing styles) ... */
        </style>
    """, unsafe_allow_html=True)

    # App title with port info
    st.markdown(f"# Blue Maize Maturity Predictor [Port:{st.session_state.api_port}]")

    HISTORY_FILE = "prediction_history.csv"

    # Load history with caching
    @st.cache_data
    def load_history():
        if os.path.exists(HISTORY_FILE):
            return pd.read_csv(HISTORY_FILE).to_dict(orient="records")
        return []

    if "history" not in st.session_state:
        st.session_state.history = load_history()

    # Rest of your existing Streamlit UI code remains the same
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
            # Optimized image processing
            @st.cache_data
            def process_image(file):
                image = Image.open(file).convert("RGB")
                resized = image.resize((100, 100))
                img_np = np.array(resized)
                avg_color = img_np.mean(axis=(0, 1)).astype(int)
                return image, img_np, avg_color

            image, img_np, avg_color = process_image(uploaded_file)
            r, g, b = avg_color
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.success(f"**Extracted RGB values → R: {r}, G: {g}, B: {b}**")

            # RGB heatmap
            st.markdown("## Color Channels Analysis")
            fig, axs = plt.subplots(1, 3, figsize=(12, 3))
            for i, ax in enumerate(axs):
                ax.imshow(img_np[:, :, i], cmap='Reds' if i == 0 else 'Greens' if i == 1 else 'Blues')
                ax.set_title(['Red', 'Green', 'Blue'][i], color='#1e40af', fontsize=12)
                ax.axis("off")
            st.pyplot(fig)

    # Environmental data
    st.markdown("## Environmental Conditions")
    col1, col2 = st.columns(2)
    with col1:
        temp = st.slider("Temperature (°C)", 20.0, 35.0, 25.0)
    with col2:
        hum = st.slider("Humidity (%)", 30.0, 80.0, 50.0)

    # Predict button with improved error handling
    if st.button("Predict Maturity", type="primary"):
        data = {
            "R": r, "G": g, "B": b,
            "temperature": temp,
            "humidity": hum
        }

        try:
            with st.spinner("Analyzing..."):
                response = requests.post(
                    f"http://localhost:{st.session_state.api_port}/predict",
                    json=data,
                    timeout=5
                )

                if response.status_code == 200:
                    result = response.json()
                    prediction = result["prediction"]
                    st.success(f"Prediction: **{prediction}**")

                    entry = {
                        "R": r, "G": g, "B": b,
                        "Temp": temp, "Humidity": hum,
                        "Prediction": prediction
                    }
                    st.session_state.history.append(entry)
                    pd.DataFrame(st.session_state.history).to_csv(HISTORY_FILE, index=False)
                else:
                    st.error(f"API Error: {response.text}")

        except requests.exceptions.RequestException as e:
            st.error(f"Connection failed: {str(e)}")
            st.error("Please ensure the backend service is running")

    # History section with pagination
    if st.session_state.history:
        st.markdown("## Prediction History")

        # Pagination
        items_per_page = 5
        total_pages = (len(st.session_state.history) // items_per_page + 1
                       page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)

        start_idx = (page - 1) * items_per_page
        end_idx = min(page * items_per_page, len(st.session_state.history))

        for i, entry in enumerate(reversed(st.session_state.history[start_idx:end_idx]), start_idx + 1):
            with
        st.container():
        st.markdown(f"""
                <div class="history-card">
                    <div style='font-weight: 600; color: #1e40af;'>Prediction #{i}</div>
                    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-top: 0.5rem;'>
                        <div><b>RGB:</b> {entry['R']}, {entry['G']}, {entry['B']}</div>
                        <div><b>Temp:</b> {entry['Temp']}°C</div>
                        <div><b>Humidity:</b> {entry['Humidity']}%</div>
                        <div><b>Result:</b> {entry['Prediction']}</div>
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

        # Cleanup handler


def cleanup():
    print("🛑 Cleaning up resources...")
    if 'api_port' in st.session_state:
        try:
            requests.post(f"http://localhost:{st.session_state.api_port}/shutdown", timeout=1)
        except:
            pass


if __name__ == "__main__":
    # Register cleanup
    atexit.register(cleanup)

    # Start FastAPI in a separate thread
    try:
        Thread(target=run_fastapi, daemon=True).start()
    except Exception as e:
        st.error(f"Failed to start API: {str(e)}")

    # Start Streamlit
    main()