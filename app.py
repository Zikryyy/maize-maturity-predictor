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
        page_icon="🌽",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Modern UI styling
    st.markdown("""
        <style>
            /* Global styles and reset */
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }

            body, .stApp {
                background-color: #f8f9fc !important;
                color: #374151 !important;
            }

            /* Header styling */
            h1 {
                color: #1e40af !important;
                font-weight: 800 !important;
                font-size: 2rem !important;
                margin-bottom: 0.5rem !important;
                letter-spacing: -0.025em !important;
            }

            h2 {
                color: #2563eb !important;
                font-weight: 600 !important;
                font-size: 1.25rem !important;
                margin: 1.5rem 0 1rem 0 !important;
                letter-spacing: -0.01em !important;
            }

            h3 {
                color: #4b5563 !important;
                font-weight: 600 !important;
                font-size: 1.1rem !important;
                margin-top: 1rem !important;
            }

            /* Container cards styling */
            .stTabs [data-testid="stVerticalBlock"] {
                background-color: white;
                padding: 1.5rem;
                border-radius: 0.75rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                margin-bottom: 1rem;
                border: 1px solid #f1f5f9;
            }

            /* Tab styling */
            .stTabs [role="tablist"] {
                gap: 0.5rem;
                border-bottom: 1px solid #e2e8f0;
                padding-bottom: 0px;
            }

            .stTabs [role="tab"] {
                background-color: transparent !important;
                color: #4b5563 !important;
                padding: 0.5rem 1rem !important;
                font-size: 1rem !important;
                font-weight: 500 !important;
                border-radius: 0.375rem 0.375rem 0 0 !important;
                border: none !important;
                border-bottom: 3px solid transparent !important;
                margin-right: 0.25rem !important;
                transition: all 0.2s ease !important;
            }

            .stTabs [aria-selected="true"] {
                background-color: transparent !important;
                color: #2563eb !important;
                border-bottom: 3px solid #2563eb !important;
                font-weight: 600 !important;
            }

            .stTabs [role="tab"]:hover:not([aria-selected="true"]) {
                background-color: #f8fafc !important;
                color: #1e40af !important;
            }

            /* Input fields styling */
            [data-testid="stNumberInput"] input {
                border-radius: 0.375rem !important;
                border: 1px solid #d1d5db !important;
                padding: 0.5rem !important;
                transition: all 0.15s ease-in-out !important;
            }

            [data-testid="stNumberInput"] input:focus {
                border-color: #2563eb !important;
                box-shadow: 0 0 0 1px rgba(37, 99, 235, 0.3) !important;
            }

            /* Sliders styling */
            [data-testid="stSlider"] {
                padding-top: 1rem !important;
                padding-bottom: 1.5rem !important;
            }

            .stSlider [data-testid="stThumbValue"] {
                background-color: #2563eb !important;
                color: white !important;
                font-weight: 600 !important;
            }

            /* Button styling */
            .stButton > button {
                width: 100%;
                border-radius: 0.375rem !important;
                font-weight: 500 !important;
                transition: all 0.15s ease-in-out !important;
                padding: 0.5rem 1rem !important;
                border: none !important;
            }

            .stButton > button[data-baseweb="button"] {
                background-color: #2563eb !important;
                color: white !important;
            }

            .stButton > button:hover {
                opacity: 0.9 !important;
                transform: translateY(-1px) !important;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
            }

            /* Input method button styling */
            button[data-testid="baseButton-secondary"] {
                background-color: #f1f5f9 !important;
                color: #475569 !important;
                border: 1px solid #e2e8f0 !important;
            }

            button[data-testid="baseButton-primary"] {
                background-color: #2563eb !important;
                color: white !important;
            }

            /* Chat styling */
            .stChatMessage {
                padding: 0.75rem 1rem !important;
                margin-bottom: 0.5rem !important;
                border-radius: 0.75rem !important;
                box-shadow: none !important;
            }

            [data-testid="stChatMessage"] [data-testid="stVerticalBlock"] {
                background-color: transparent !important;
                padding: 0 !important;
                border-radius: 0 !important;
                box-shadow: none !important;
                margin-bottom: 0 !important;
                border: none !important;
            }

            .stChatInput textarea, .stChatInput [data-testid="textarea"] {
                border-radius: 1.5rem !important;
                border: 1px solid #d1d5db !important;
                padding: 0.75rem 1.25rem !important;
                background-color: white !important;
                font-size: 0.95rem !important;
            }

            .stChatInput textarea:focus {
                border-color: #2563eb !important;
                box-shadow: 0 0 0 1px rgba(37, 99, 235, 0.3) !important;
            }

            /* Prediction history card styling */
            .history-card {
                background-color: white;
                border-radius: 0.5rem;
                padding: 1rem;
                margin-bottom: 0.75rem;
                border-left: 4px solid #2563eb;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            }

            /* Download button styling */
            [data-testid="stDownloadButton"] button {
                background-color: #f1f5f9 !important;
                color: #475569 !important;
                border: 1px solid #e2e8f0 !important;
                border-radius: 0.375rem !important;
                padding: 0.5rem 1rem !important;
                transition: all 0.15s ease-in-out !important;
            }

            [data-testid="stDownloadButton"] button:hover {
                background-color: #e2e8f0 !important;
                color: #1e40af !important;
            }

            /* Status indicator styling */
            .stAlert {
                background-color: #f0f9ff !important;
                color: #0369a1 !important;
                border-radius: 0.5rem !important;
                padding: 0.75rem 1rem !important;
                margin-bottom: 1rem !important;
            }

            .stAlert [data-testid="stMarkdownContainer"] p {
                color: #0369a1 !important;
                font-weight: 500 !important;
            }

            /* Image uploader styling */
            [data-testid="stFileUploader"] {
                border: 2px dashed #d1d5db !important;
                padding: 1.5rem !important;
                border-radius: 0.5rem !important;
                background-color: #f8fafc !important;
                transition: all 0.15s ease-in-out !important;
            }

            [data-testid="stFileUploader"]:hover {
                border-color: #2563eb !important;
                background-color: #f0f7ff !important;
            }

            [data-testid="stFileUploader"] small {
                color: #64748b !important;
            }

            /* Success message styling */
            [data-testid="stAlert"] {
                background-color: #f0fdf4 !important;
                border-color: #22c55e !important;
                color: #166534 !important;
                border-radius: 0.5rem !important;
                padding: 0.75rem 1rem !important;
            }

            /* Layout spacing */
            .block-container {
                padding-top: 2rem !important;
                padding-bottom: 2rem !important;
                max-width: 1200px !important;
            }

            /* Caption text */
            [data-testid="caption"] {
                color: #64748b !important;
                font-size: 0.875rem !important;
                margin-bottom: 1rem !important;
            }

            /* For grid layout with clean backgrounds */
            .grid-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 1rem;
            }

            .content-card {
                background-color: white;
                border-radius: 0.75rem;
                padding: 1.5rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                border: 1px solid #f1f5f9;
            }

            /* Plot styling */
            .matplotlib-figure {
                margin-top: 1rem !important;
                margin-bottom: 1rem !important;
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

    # App container
    with st.container():
        # App header
        col1, col2 = st.columns([5, 1])
        with col1:
            st.title("Blue Maize Maturity Predictor")
            st.caption("Advanced analysis tool for maize maturity prediction using RGB values and environmental data")

        with col2:
            st.write("")
            st.write("")
            st.image("https://api/placeholder/64/64", width=64)  # Replace with your logo

        # Main tabs
        tab1, tab2 = st.tabs(["🌽 Predictor", "💬 Assistant"])

        with tab1:
            # Create two columns for layout
            main_col, side_col = st.columns([3, 2])

            with main_col:
                # Input method selector - cleaner design
                st.markdown("### Input Method")

                input_col1, input_col2 = st.columns(2)
                with input_col1:
                    if st.button("Manual RGB Entry",
                                 use_container_width=True,
                                 type="primary" if st.session_state.input_mode == "Manual RGB Entry" else "secondary"):
                        st.session_state.input_mode = "Manual RGB Entry"
                        st.rerun()

                with input_col2:
                    if st.button("Upload Image",
                                 use_container_width=True,
                                 type="primary" if st.session_state.input_mode == "Upload Image for RGB" else "secondary"):
                        st.session_state.input_mode = "Upload Image for RGB"
                        st.rerun()

                # Set the mode based on session state
                mode = st.session_state.input_mode

                # Create content card
                st.markdown('<div class="content-card">', unsafe_allow_html=True)

                # RGB input handling with improved layout
                if mode == "Manual RGB Entry":
                    st.markdown("### RGB Values")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        r = st.number_input("Red (R)", min_value=0, max_value=255, value=100)
                    with col2:
                        g = st.number_input("Green (G)", min_value=0, max_value=255, value=100)
                    with col3:
                        b = st.number_input("Blue (B)", min_value=0, max_value=255, value=100)

                    # Color preview
                    st.markdown(f"""
                        <div style="
                            background-color: rgb({r}, {g}, {b}); 
                            height: 40px; 
                            width: 100%; 
                            border-radius: 0.375rem;
                            margin: 1rem 0;
                            border: 1px solid #e2e8f0;">
                        </div>
                    """, unsafe_allow_html=True)

                else:
                    st.markdown("### Upload Maize Image")
                    uploaded_file = st.file_uploader(
                        "Drop image here",
                        type=["jpg", "jpeg", "png"],
                        key="image_uploader"
                    )

                    if uploaded_file is not None:
                        # Process image
                        image = Image.open(uploaded_file).convert("RGB")

                        # Show image with clean styling
                        st.image(image, use_column_width=True)

                        # Process image
                        resized = image.resize((100, 100))
                        img_np = np.array(resized)
                        avg_color = img_np.mean(axis=(0, 1)).astype(int)
                        r, g, b = int(avg_color[0]), int(avg_color[1]), int(avg_color[2])

                        # Display extracted RGB
                        st.markdown("#### Extracted RGB Values")

                        rgb_cols = st.columns(3)
                        with rgb_cols[0]:
                            st.metric("R", r)
                        with rgb_cols[1]:
                            st.metric("G", g)
                        with rgb_cols[2]:
                            st.metric("B", b)

                        # Color preview
                        st.markdown(f"""
                            <div style="
                                background-color: rgb({r}, {g}, {b}); 
                                height: 40px; 
                                width: 100%; 
                                border-radius: 0.375rem;
                                margin: 1rem 0;
                                border: 1px solid #e2e8f0;">
                            </div>
                        """, unsafe_allow_html=True)

                        # RGB channels visualization (modern style)
                        st.markdown("#### Color Channels Analysis")

                        # Create a figure with a clean, modern style
                        fig, axs = plt.subplots(1, 3, figsize=(10, 2.5))
                        fig.patch.set_facecolor('#ffffff')

                        channel_names = ['Red', 'Green', 'Blue']
                        cmaps = ['Reds', 'Greens', 'Blues']

                        for i, ax in enumerate(axs):
                            ax.imshow(img_np[:, :, i], cmap=cmaps[i])
                            ax.set_title(channel_names[i], color='#1e40af', fontsize=10, fontweight='bold')
                            ax.axis('off')

                        plt.tight_layout()
                        st.pyplot(fig)

                # Environmental conditions
                st.markdown("### Environmental Conditions")

                env_col1, env_col2 = st.columns(2)
                with env_col1:
                    temp = st.slider("Temperature (°C)",
                                     min_value=20.0,
                                     max_value=35.0,
                                     value=25.0,
                                     key="temp_slider")
                    st.caption("Optimal range: 20-35°C")

                with env_col2:
                    hum = st.slider("Humidity (%)",
                                    min_value=30.0,
                                    max_value=80.0,
                                    value=50.0,
                                    key="hum_slider")
                    st.caption("Optimal range: 30-80%")

                # End content card
                st.markdown('</div>', unsafe_allow_html=True)

                # Prediction button
                if st.button("Predict Maturity",
                             type="primary",
                             key="predict_btn",
                             use_container_width=True):

                    data = {
                        "R": r, "G": g, "B": b,
                        "temperature": temp,
                        "humidity": hum
                    }

                    try:
                        with st.spinner("Analyzing plant maturity..."):
                            response = requests.post("http://localhost:8000/predict", json=data)
                            result = response.json()

                        if "prediction" in result:
                            prediction = result["prediction"]

                            # Create a visually appealing result card
                            st.markdown(f"""
                                <div style="
                                    background-color: {'#dcfce7' if prediction == 'Mature' else '#f3f4f6'}; 
                                    padding: 1.25rem; 
                                    border-radius: 0.5rem; 
                                    margin: 1rem 0; 
                                    text-align: center;
                                    border-left: 4px solid {'#16a34a' if prediction == 'Mature' else '#6b7280'};
                                    box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                                    <p style="font-size: 0.9rem; color: {'#166534' if prediction == 'Mature' else '#374151'}; margin-bottom: 0.25rem;">
                                        Prediction Result
                                    </p>
                                    <h2 style="font-size: 1.75rem; font-weight: 700; color: {'#16a34a' if prediction == 'Mature' else '#4b5563'}; margin: 0;">
                                        {prediction}
                                    </h2>
                                </div>
                            """, unsafe_allow_html=True)

                            # Save to history with timestamp
                            entry = {
                                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "R": r, "G": g, "B": b,
                                "Temp": temp, "Humidity": hum,
                                "Prediction": prediction
                            }
                            st.session_state.history.append(entry)
                            pd.DataFrame(st.session_state.history).to_csv("prediction_history.csv", index=False)
                        else:
                            st.error(f"Error: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")

            # Side column for history
            with side_col:
                st.markdown("### Prediction History")

                if not st.session_state.history:
                    st.info("No predictions yet. Start by making a prediction!")
                else:
                    # Create a scrollable history area with fixed height
                    with st.container():
                        # Display recent predictions with timestamp
                        for i, entry in enumerate(reversed(st.session_state.history[-5:]), 1):
                            timestamp = entry.get('Timestamp', 'No date')

                            # Formatted history card
                            st.markdown(f"""
                                <div class="history-card">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                        <div style="font-weight: 600; color: #1e40af;">#{i}</div>
                                        <div style="font-size: 0.75rem; color: #64748b;">{timestamp}</div>
                                    </div>

                                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                                        <div><span style="font-weight: 500;">RGB:</span> {entry['R']}, {entry['G']}, {entry['B']}</div>
                                        <div><span style="font-weight: 500;">Temp:</span> {entry['Temp']}°C</div>
                                        <div><span style="font-weight: 500;">Humidity:</span> {entry['Humidity']}%</div>
                                        <div>
                                            <span style="font-weight: 500;">Result:</span> 
                                            <span style="color: {'#16a34a' if entry['Prediction'] == 'Mature' else '#4b5563'}; font-weight: 500;">
                                                {entry['Prediction']}
                                            </span>
                                        </div>
                                    </div>

                                    <div style="
                                        background-color: rgb({entry['R']}, {entry['G']}, {entry['B']}); 
                                        height: 10px; 
                                        width: 100%; 
                                        border-radius: 0.25rem;
                                        margin-top: 0.75rem;">
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

                    # Export button at the bottom
                    st.download_button(
                        label="📊 Export History (CSV)",
                        data=pd.DataFrame(st.session_state.history).to_csv(index=False),
                        file_name=f"maize_prediction_history_{datetime.now().strftime('%Y-%m-%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

        with tab2:
            # Assistant tab with modern chat interface
            st.markdown("### Maize Assistant")
            st.caption("Ask questions about maize maturity, RGB values, or how to use this tool")

            # Create a cleaner chat container
            chat_container = st.container()

            with chat_container:
                # Display chat history with improved styling
                for msg in st.session_state.chat_history:
                    role = msg["role"]
                    content = msg["content"]

                    if role == "user":
                        st.markdown(f"""
                            <div style="
                                display: flex;
                                justify-content: flex-end;
                                margin-bottom: 0.75rem;">
                                <div style="
                                    background-color: #e0f2fe;
                                    border-radius: 1rem 1rem 0 1rem;
                                    padding: 0.75rem 1rem;
                                    max-width: 80%;
                                    color: #0369a1;">
                                    {content}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:  # assistant
                        st.markdown(f"""
                            <div style="
                                display: flex;
                                justify-content: flex-start;
                                margin-bottom: 0.75rem;">
                                <div style="
                                    background-color: #f1f5f9;
                                    border-radius: 1rem 1rem 1rem 0;
                                    padding: 0.75rem 1rem;
                                    max-width: 80%;
                                    color: #334155;">
                                    {content.replace('\n', '<br>')}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

            # Chat input with modern styling
            prompt = st.chat_input("Ask me about maize maturity prediction...", key="chat_input")

            if prompt:
                # Add user message
                st.session_state.chat_history.append({"role": "user", "content": prompt})

                # Get response (case-insensitive)
                clean_prompt = prompt.lower().strip()
                response = CHATBOT_RESPONSES.get(clean_prompt, CHATBOT_RESPONSES["default"])

                # Dynamic responses based on current settings
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