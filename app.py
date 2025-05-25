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
    "hello": "Hi there! I'm your MaizeMaturity Assistant. How can I help you today?",
    "hi": "Hello! Ready to predict maize maturity? Try asking about 'how to use' or 'what is this'.",
    "help": "Here's what I can help with:\n\n"
            "• How to use the predictor\n"
            "• Understanding RGB values\n"
            "• Optimal temperature/humidity\n"
            "• Interpreting results\n"
            "• Image upload tips\n"
            "• Prediction accuracy\n"
            "• Model information\n"
            "• Troubleshooting errors",
    "how to use": "To use MaizeMaturity:\n\n"
                  "1. Select an input method (Manual RGB or Upload Image)\n"
                  "2. Enter RGB values or upload a clear image of the maize\n"
                  "3. Set temperature (20-35°C) and humidity (30-80%)\n"
                  "4. Click 'Predict Maturity' to see the result\n"
                  "5. Check prediction history for past results",
    "what is this": "MaizeMaturity predicts maize maturity using:\n\n"
                    "• Plant color (RGB values from manual entry or image)\n"
                    "• Temperature (optimal: 20-35°C)\n"
                    "• Humidity (optimal: 30-80%)\n\n"
                    "It uses a machine learning model to determine if maize is Mature or Immature.",
    "temperature": "Optimal temperature range: 20-35°C\n\n"
                   "• Higher temps (above 30°C) may speed up maturation\n"
                   "• Lower temps (below 20°C) may slow it down",
    "humidity": "Ideal humidity levels: 30-80%\n\n"
                "• High humidity (>80%) may increase disease risk\n"
                "• Low humidity (<30%) can stress the plant",
    "rgb": "Typical mature maize RGB ranges:\n\n"
           "• Red: 29-35\n"
           "• Green: 35-51\n"
           "• Blue: 51-60\n\n"
           "These values reflect the color of mature maize kernels or leaves.",
    "history": "Your prediction history shows past analyses with:\n\n"
               "• RGB values\n"
               "• Temperature and humidity\n"
               "• Maturity predictions\n\n"
               "You can download it as a CSV file.",
    "image upload": "To upload an image:\n\n"
                    "• Use a clear, well-lit photo of the maize (JPG, JPEG, or PNG)\n"
                    "• Ensure the image is under 200MB\n"
                    "• The app will extract average RGB values from the image",
    "prediction accuracy": "The prediction is based on a Random Forest model trained on maize data. Accuracy depends on:\n\n"
                          "• Quality of RGB input (clear images or accurate manual values)\n"
                          "• Environmental conditions within optimal ranges\n"
                          "• Model training data relevance\n\n"
                          "For best results, use typical RGB ranges and realistic environmental data.",
    "model info": "The model is a Random Forest classifier trained on:\n\n"
                  "• RGB color values of maize\n"
                  "• Temperature and humidity data\n\n"
                  "It predicts whether maize is 'Mature' or 'Immature' based on these inputs.",
    "error": "Common errors and fixes:\n\n"
             "• 'Connection failed': Ensure the app is running and try again\n"
             "• 'Invalid RGB': Use values between 0-255\n"
             "• 'Image processing error': Upload a valid JPG, JPEG, or PNG under 200MB\n"
             "• 'Model not loaded': Check if 'rf_model_maize_maturity.pkl' exists",
    "clear history": "To clear prediction history:\n\n"
                     "• The app doesn't support direct history clearing\n"
                     "• You can manually delete 'prediction_history.csv' from your system\n"
                     "• Restart the app to reset the history",
    "default": "I can help with:\n\n"
               "• 'how to use' the app\n"
               "• 'temperature' ranges\n"
               "• 'humidity' effects\n"
               "• 'rgb' value guidance\n"
               "• 'image upload' tips\n"
               "• 'prediction accuracy' details\n"
               "• 'model info' about the predictor\n"
               "• 'history' explanation\n"
               "• 'error' troubleshooting"
}


# --- Streamlit Frontend ---
def main():
    st.set_page_config(
        page_title="MaizeMaturity",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS for professional design with gradient animation
    st.markdown("""
        <style>
            /* Import modern font */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

            /* Base styles */
            body, .stApp {
                font-family: 'Inter', sans-serif !important;
                background-color: #F8FAFC !important;
                color: #1F2937 !important;
            }

            /* Container padding */
            .stApp > div:first-child {
                padding: 2rem;
                max-width: 800px;
                margin: 0 auto;
            }

            /* Title styles with gradient animation */
            h1 {
                text-align: center;
                font-size: 2.25rem !important;
                font-weight: 700 !important;
                margin-bottom: 2rem !important;
                background: linear-gradient(90deg, #1E3A8A, #2DD4BF, #1E3A8A);
                background-size: 200% 100%;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                animation: gradientFlow 5s ease-in-out infinite;
            }

            @keyframes gradientFlow {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }

            h2 {
                color: #1F2937 !important;
                font-size: 1.5rem !important;
                font-weight: 600 !important;
                margin-bottom: 1rem !important;
            }

            /* Instruction box */
            .instruction-box {
                background-color: #FFFFFF !important;
                border: 1px solid #D1D5DB !important;
                border-radius: 8px !important;
                padding: 1rem !important;
                margin-bottom: 2rem !important;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
            }

            /* Force text color */
            * {
                color: #1F2937 !important;
            }

            /* Radio button labels */
            .st-bq, div[role="radiogroup"] * {
                color: #1F2937 !important;
            }

            /* Input fields */
            .stNumberInput input, .stSlider div[role="slider"] {
                background-color: #FFFFFF !important;
                border: 1px solid #D1D5DB !important;
                border-radius: 8px !important;
                padding: 0.5rem !important;
                transition: border-color 0.2s ease !important;
            }

            .stNumberInput input:focus, .stSlider div[role="slider"]:focus {
                border-color: #1E3A8A !important;
                box-shadow: 0 0 0 3px rgba(30, 58, 138, 0.1) !important;
            }

            /* Buttons */
            .stButton button {
                background-color: #1E3A8A !important;
                color: #FFFFFF !important;
                border: none !important;
                border-radius: 8px !important;
                padding: 0.75rem 1.5rem !important;
                font-weight: 500 !important;
                transition: background-color 0.2s ease, transform 0.1s ease !important;
            }

            .stButton button:hover {
                background-color: #3B82F6 !important;
                transform: translateY(-1px) !important;
            }

            .stButton button:active {
                transform: translateY(0) !important;
            }

            .stButton button[kind="secondary"] {
                background-color: #FFFFFF !important;
                color: #2DD4BF !important;
                border: 1px solid #D1D5DB !important;
            }

            .stButton button[kind="secondary"]:hover {
                background-color: #F8FAFC !important;
                border-color: #2DD4BF !important;
            }

            /* File uploader */
            .stFileUploader {
                background-color: #FFFFFF !important;
                border: 1px solid #D1D5DB !important;
                border-radius: 8px !important;
                padding: 1rem !important;
            }

            /* Chat-specific styles */
            .stChatInput textarea {
                background-color: #FFFFFF !important;
                border: 1px solid #D1D5DB !important;
                border-radius: 8px !important;
                padding: 0.75rem !important;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
            }

            .stChatInput textarea:focus {
                border-color: #1E3A8A !important;
                box-shadow: 0 0 0 3px rgba(30, 58, 138, 0.1) !important;
            }

            .stChatMessage {
                border-radius: 8px !important;
                padding: 1rem !important;
                margin: 0.5rem 0 !important;
                background-color: #FFFFFF !important;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
            }

            [data-testid="stChatMessage"] > div:first-child {
                background-color: #F8FAFC !important;
                border-left: 3px solid #2DD4BF !important;
            }

            /* Prediction card styles */
            .history-card {
                padding: 1rem !important;
                margin-bottom: 1rem !important;
                border-radius: 8px !important;
                background-color: #FFFFFF !important;
                border: 1px solid #D1D5DB !important;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
            }

            /* Tab styling */
            .stTabs [role="tablist"] {
                justify-content: center !important;
                border-bottom: 1px solid #D1D5DB !important;
                margin-bottom: 2rem !important;
            }

            .stTabs [role="tab"] {
                padding: 0.75rem 1.5rem !important;
                font-weight: 500 !important;
                color: #6B7280 !important;
                transition: color 0.2s ease, border-color 0.2s ease !important;
            }

            .stTabs [aria-selected="true"] {
                color: #1E3A8A !important;
                border-bottom: 2px solid #1E3A8A !important;
            }

            .stTabs [role="tab"]:hover {
                color: #1E3A8A !important;
            }

            /* Download button */
            .stDownloadButton button {
                background-color: #FFFFFF !important;
                color: #2DD4BF !important;
                border: 1px solid #D1D5DB !important;
                border-radius: 8px !important;
                padding: 0.75rem 1.5rem !important;
                font-weight: 500 !important;
            }

            .stDownloadButton button:hover {
                background-color: #F8FAFC !important;
                border-color: #2DD4BF !important;
            }

            /* Success and error messages */
            .stSuccess {
                background-color: #10B981 !important;
                color: #FFFFFF !important;
            }

            .stError {
                background-color: #EF4444 !important;
                color: #FFFFFF !important;
            }

            /* Spinner */
            .stSpinner {
                color: #1E3A8A !important;
            }

            /* Remove default Streamlit decorations */
            [data-testid="stDecoration"] {
                display: none !important;
            }

            /* Ensure consistent column spacing */
            .stColumn > div {
                padding: 0 0.5rem !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # Initialize sessions
    if "history" not in st.session_state:
        st.session_state.history = pd.read_csv("prediction_history.csv").to_dict(orient="records") if os.path.exists(
            "prediction_history.csv") else []

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialize input mode session state
    if "input_mode" not in st.session_state:
        st.session_state.input_mode = "Manual RGB Entry"

    # Main tabs
    tab1, tab2 = st.tabs(["🌽 Maize Predictor", "💬 Chat Assistant"])

    with tab1:
        # --- Prediction UI with Instructions ---
        st.markdown("# MaizeMaturity")

        # Instructions for users
        st.markdown("""
        <div class="instruction-box">
            <h3 style="color: #1F2937; font-size: 1.25rem; font-weight: 600; margin-bottom: 0.5rem;">How to Use MaizeMaturity</h3>
            <p style="color: #1F2937;">
                1. Choose how to input maize color: enter RGB values manually or upload an image.<br>
                2. Adjust temperature (20-35°C) and humidity (30-80%) sliders.<br>
                3. Click "Predict Maturity" to get the result.<br>
                4. View past predictions in the history section or ask the Chat Assistant for help.<br>
                <i>Tip: For images, use clear, well-lit photos of maize for accurate RGB extraction.</i>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Input mode selector
        st.markdown("## Input Method")

        # Create radio buttons with proper styling using custom components
        col1, col2 = st.columns(2)

        def handle_manual_click():
            st.session_state.input_mode = "Manual RGB Entry"

        def handle_upload_click():
            st.session_state.input_mode = "Upload Image for RGB"

        with col1:
            if st.button("Manual RGB Entry",
                         key="manual_btn",
                         use_container_width=True,
                         type="primary" if st.session_state.input_mode == "Manual RGB Entry" else "secondary"):
                handle_manual_click()
                st.rerun()

        with col2:
            if st.button("Upload Image for RGB",
                         key="upload_btn",
                         use_container_width=True,
                         type="primary" if st.session_state.input_mode == "Upload Image for RGB" else "secondary"):
                handle_upload_click()
                st.rerun()

        # Set the mode based on session state
        mode = st.session_state.input_mode

        # RGB input handling
        if mode == "Manual RGB Entry":
            col1, col2, col3 = st.columns(3)
            with col1:
                r = st.number_input("Red (R)", min_value=0, max_value=255, value=100, help="Enter value between 0-255. Typical for mature maize: 29-35")
            with col2:
                g = st.number_input("Green (G)", min_value=0, max_value=255, value=100, help="Enter value between 0-255. Typical for mature maize: 35-51")
            with col3:
                b = st.number_input("Blue (B)", min_value=0, max_value=255, value=100, help="Enter value between 0-255. Typical for mature maize: 51-60")
        else:
            uploaded_file = st.file_uploader(
                "Drag and drop file here (Limit 200MB - JPG, JPEG, PNG)",
                type=["jpg", "jpeg", "png"], key="image_uploader",
                help="Upload a clear image of maize. The app will extract average RGB values."
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
                    ax.set_title(['Red', 'Green', 'Blue'][i], color='#1E3A8A', fontsize=12)
                    ax.axis("off")
                plt.tight_layout()
                st.pyplot(fig)

        # Environmental conditions
        st.markdown("## Environmental Conditions")
        col1, col2 = st.columns(2)
        with col1:
            temp = st.slider("Temperature (°C)", 20.0, 35.0, 25.0, key="temp_slider", help="Optimal range: 20-35°C")
        with col2:
            hum = st.slider("Humidity (%)", 30.0, 80.0, 50.0, key="hum_slider", help="Optimal range: 30-80%")

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
                        <div style='font-weight: 600; color: #1E3A8A;'>Prediction #{i}</div>
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
        st.markdown("# MaizeMaturity Assistant")
        st.caption("Ask me about maize maturity prediction or troubleshooting")

        # Display chat history
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])

        # Chat input
        if prompt := st.chat_input("Type your question here (e.g., 'how to use', 'rgb', 'error')...", key="chat_input"):
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