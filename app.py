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


# --- Chatbot Responses ---
CHATBOT_RESPONSES = {
    "hello": "Hi there! I'm your Maize Maturity Assistant. How can I help?",
    "hi": "Hello! Ready to predict maize maturity?",
    "how to use": """
    1. Choose input method (Manual RGB or Image Upload)
    2. Set environmental conditions
    3. Click 'Predict Maturity'
    """,
    "what is this": "This app predicts maize maturity using RGB values and environmental data",
    "temperature": "Optimal temperature range: 20-35°C",
    "humidity": "Ideal humidity levels: 30-80%",
    "rgb": "For mature maize: R(29-35), G(35-51), B(51-60)",
    "default": "I can answer about: 'how to use', 'temperature', 'humidity', or 'RGB values'"
}


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
            /* Your existing styles... */
            .stChatInput textarea {
                background-color: #f0f4f8 !important;
            }
            .stChatMessage {
                border-radius: 12px;
                padding: 1rem;
                margin: 0.5rem 0;
            }
        </style>
    """, unsafe_allow_html=True)

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # App title with tabs
    tab1, tab2 = st.tabs(["🌽 Maize Predictor", "💬 Chat Assistant"])

    with tab1:
        # --- Your existing prediction UI ---
        st.markdown("# Blue Maize Maturity Predictor")

        HISTORY_FILE = "prediction_history.csv"
        if "history" not in st.session_state:
            if os.path.exists(HISTORY_FILE):
                st.session_state.history = pd.read_csv(HISTORY_FILE).to_dict(orient="records")
            else:
                st.session_state.history = []

        # Input mode
        st.markdown("## Input Method")
        mode = st.radio("", ["Manual RGB Entry", "Upload Image for RGB"],
                        horizontal=True, key="input_mode", label_visibility="collapsed")

        if mode == "Manual RGB Entry":
            col1, col2, col3 = st.columns(3)
            with col1:
                r = st.number_input("Red (R)", min_value=0, max_value=255, value=100, key="r_value")
            with col2:
                g = st.number_input("Green (G)", min_value=0, max_value=255, value=100, key="g_value")
            with col3:
                b = st.number_input("Blue (B)", min_value=0, max_value=255, value=100, key="b_value")
        else:
            uploaded_file = st.file_uploader(
                "Drag and drop file here (Limit 200MB - JPG, JPEG, PNG)",
                type=["jpg", "jpeg", "png"], key="image_uploader"
            )

            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Image", use_column_width=True)
                resized = image.resize((100, 100))
                img_np = np.array(resized)
                avg_color = img_np.mean(axis=(0, 1)).astype(int)
                r, g, b = int(avg_color[0]), int(avg_color[1]), int(avg_color[2])
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
            temp = st.slider("Temperature (°C)", 20.0, 35.0, 25.0, key="temp_slider")
        with col2:
            hum = st.slider("Humidity (%)", 30.0, 80.0, 50.0, key="hum_slider")

        # Predict button
        if st.button("Predict Maturity", type="primary", key="predict_btn"):
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
                        "Prediction": prediction
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
                use_container_width=True,
                key="download_btn"
            )

    with tab2:
        # --- Chatbot UI ---
        st.markdown("# 💬 Maize Assistant")
        st.caption("Ask me about maize maturity prediction")

        # Display chat messages
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])

        # Chat input
        if prompt := st.chat_input("Type your question here..."):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            # Get bot response (case-insensitive check)
            response = CHATBOT_RESPONSES.get(
                prompt.lower(),
                CHATBOT_RESPONSES["default"]
            )

            # Special dynamic responses
            if "current temp" in prompt.lower():
                response = f"Current temperature setting: {temp}°C"
            elif "current humidity" in prompt.lower():
                response = f"Current humidity setting: {hum}%"
            elif "current rgb" in prompt.lower():
                response = f"Current RGB values: R={r}, G={g}, B={b}"

            # Add bot response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})

            # Rerun to update chat display
            st.rerun()


if __name__ == "__main__":
    # Start FastAPI in a separate thread
    Thread(target=run_fastapi, daemon=True).start()

    # Start Streamlit
    main()