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


# --- Chatbot Configuration ---
CHATBOT_RESPONSES = {
    "hello": "Hi there! I'm your Maize Maturity Assistant. How can I help?",
    "hi": "Hello! Ready to predict maize maturity?",
    "help": "Here's what I can help with:\n\n"
            "• How to use the predictor\n"
            "• Understanding RGB values\n"
            "• Optimal temperature/humidity\n"
            "• Interpreting results",
    "how to use": "1. Choose input method (Manual or Image)\n"
                  "2. Set environmental conditions\n"
                  "3. Click 'Predict Maturity'",
    "what is this": "This app predicts maize maturity using:\n\n"
                    "- Plant color (RGB values)\n"
                    "- Temperature (20-35°C optimal)\n"
                    "- Humidity (30-80% optimal)",
    "temperature": "Optimal temperature range: 20-35°C\n\n"
                   "Higher temps may indicate faster maturation",
    "humidity": "Ideal humidity levels: 30-80%\n\n"
                "High humidity can affect disease risk",
    "rgb": "Typical mature maize RGB ranges:\n\n"
           "• Red: 29-35\n"
           "• Green: 35-51\n"
           "• Blue: 51-60",
    "history": "Your prediction history shows past analyses with:\n\n"
               "- Input values\n"
               "- Environmental conditions\n"
               "- Maturity predictions",
    "default": "I can help with:\n\n"
               "• 'how to use' the app\n"
               "• 'temperature' ranges\n"
               "• 'humidity' effects\n"
               "• 'rgb' value guidance\n"
               "• 'history' explanation"
}


# --- Streamlit Frontend ---
def main():
    st.set_page_config(
        page_title="Blue Maize Maturity Predictor",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS for both components
    st.markdown("""
        <style>
            /* Base styles */
            body, .stApp {
                font-family: 'Arial', sans-serif;
                background-color: white !important;
                color: black !important;
            }

            /* Title styles */
            h1 {
                color: #1a56db !important;
                text-align: center;
                margin-bottom: 1.5rem;
                font-size: 2rem;
                font-weight: 700;
            }

            /* Radio Button Labels - Updated for better specificity */
            [data-baseweb="radio"] {
                color: black !important; /* Ensure radio button labels are black */
            }

            /* More specific selectors to ensure radio button text is visible */
            [data-testid="stRadio"] label {
                color: black !important;
                font-weight: 500;
            }

            /* Target specifically the text inside radio buttons */
            [data-testid="stRadio"] [data-baseweb="radio"] label div {
                color: black !important;
            }

            /* Even more specific selector for the label text */
            div[role="radiogroup"] label span {
                color: black !important;
            }

            /* Fallback: Target all labels directly */
            label {
                color: black !important;
            }

            /* Chat-specific styles */
            .stChatInput textarea {
                background-color: #f0f4f8 !important;
                border: 1px solid #1a56db !important;
            }

            .stChatMessage {
                border-radius: 12px;
                padding: 1rem;
                margin: 0.5rem 0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }

            [data-testid="stChatMessage"] > div:first-child {
                background-color: #f0f4f8 !important;
                border-left: 4px solid #1a56db !important;
            }

            /* Ensure chat response text is black */
            [data-testid="stChatMessage"] * {
                color: black !important;
            }

            /* Prediction card styles */
            .history-card {
                padding: 1rem;
                margin-bottom: 1rem;
                border-radius: 8px;
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-left: 4px solid #2563eb;
            }

            /* Tab styling */
            .stTabs [role="tablist"] {
                justify-content: center;
            }

            .stTabs [role="tab"] {
                padding: 0.5rem 1.5rem;
                border-bottom: 3px solid transparent;
                transition: all 0.3s;
            }

            .stTabs [aria-selected="true"] {
                color: #1a56db;
                border-bottom: 3px solid #1a56db;
                font-weight: 600;
            }
        </style>
    """, unsafe_allow_html=True)

    # Initialize sessions
    if "history" not in st.session_state:
        st.session_state.history = pd.read_csv("prediction_history.csv").to_dict(orient="records") if os.path.exists(
            "prediction_history.csv") else []

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Main tabs
    tab1, tab2 = st.tabs(["🌽 Maize Predictor", "💬 Chat Assistant"])

    with tab1:
        # --- Original Prediction UI ---
        st.markdown("# Blue Maize Maturity Predictor")

        # Input mode selector
        st.markdown("## Input Method")
        mode = st.radio("", ["Manual RGB Entry", "Upload Image for RGB"],
                        horizontal=True, label_visibility="collapsed", key="mode_selector")

        # RGB input handling
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
                type=["jpg", "jpeg", "png"], key="image_uploader"
            )

            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Image", use_column_width=True)

                # Process image
                resized = image.resize((100, 100))
                img_np = np.array(resized)
                avg_color = img_np.mean(axis=(0, 1)).astype(int)
                r, g, b = int(avg_color[0]), int(avg_color[1]), int(avg_color[2])
                st.success(f"**Extracted RGB values → R: {r}, G: {g}, B: {b}**")

                # RGB heatmap
                st.markdown("## Color Channels Analysis")
                fig, axs = plt.subplots(1, 3, figsize=(12, 3))
                for i, ax in enumerate(axs):
                    ax.imshow(img_np[:, :, i], cmap='Reds' if i == 0 else 'Greens' if i == 1 else 'Blues')
                    ax.set_title(['Red', 'Green', 'Blue'][i], color='#1e40af', fontsize=12)
                    ax.axis("off")
                plt.tight_layout()
                st.pyplot(fig)

        # Environmental conditions
        st.markdown("## Environmental Conditions")
        col1, col2 = st.columns(2)
        with col1:
            temp = st.slider("Temperature (°C)", 20.0, 35.0, 25.0, key="temp_slider")
        with col2:
            hum = st.slider("Humidity (%)", 30.0, 80.0, 50.0, key="hum_slider")

        # Prediction button
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

                    # Save to history
                    entry = {
                        "R": r, "G": g, "B": b,
                        "Temp": temp, "Humidity": hum,
                        "Prediction": prediction
                    }
                    st.session_state.history.append(entry)
                    pd.DataFrame(st.session_state.history).to_csv("prediction_history.csv", index=False)
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
            st.download_button(
                label="Download History as CSV",
                data=pd.DataFrame(st.session_state.history).to_csv(index=False),
                file_name="maize_prediction_history.csv",
                mime="text/csv",
                use_container_width=True
            )

    with tab2:
        # --- Chatbot UI ---
        st.markdown("# Maize Assistant")
        st.caption("Ask me about maize maturity prediction")

        # Display chat history
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])

        # Chat input
        if prompt := st.chat_input("Type your question here...", key="chat_input"):
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            # Get response (case-insensitive)
            clean_prompt = prompt.lower().strip()
            response = CHATBOT_RESPONSES.get(clean_prompt, CHATBOT_RESPONSES["default"])

            # Dynamic responses
            if "current temp" in clean_prompt:
                response = f"Current temperature setting: {temp}°C (optimal range: 20-35°C)"
            elif "current humidity" in clean_prompt:
                response = f"Current humidity setting: {hum}% (optimal range: 30-80%)"
            elif "current rgb" in clean_prompt:
                response = f"Current RGB values:\n\n- Red: {r}\n- Green: {g}\n- Blue: {b}\n\n(Typical mature ranges: R29-35, G35-51, B51-60)"
            elif "last prediction" in clean_prompt and st.session_state.history:
                last = st.session_state.history[-1]
                response = f"Last prediction:\n\n- RGB: {last['R']}, {last['G']}, {last['B']}\n- Temp: {last['Temp']}°C\n- Humidity: {last['Humidity']}%\n- Result: {last['Prediction']}"

            # Add bot response
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()


if __name__ == "__main__":
    # Start FastAPI in a separate thread
    Thread(target=run_fastapi, daemon=True).start()

    # Start Streamlit
    main()