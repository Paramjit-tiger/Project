# import streamlit as st
# import joblib
# import os
# import cv2
# import numpy as np
# import mediapipe as mp
# from PIL import Image
# import base64
# import pyttsx3
# import time
# import threading
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
# import av

# # CONFIG
# MODEL_DIR = "saved_models"
# DATASET_DIR = "dataset"
# NUM_LANDMARKS = 21 * 2 * 3

# # Background
# def set_background(image_file):
#     with open(image_file, "rb") as f:
#         encoded = base64.b64encode(f.read()).decode()
#     css = f"""
#     <style>
#     .stApp {{
#         background-image: url("data:image/jpg;base64,{encoded}");
#         background-size: cover;
#         background-position: center;
#         background-repeat: no-repeat;
#     }}
#     </style>
#     """
#     st.markdown(css, unsafe_allow_html=True)

# # Load dataset folder names as class labels
# CLASS_LABELS = sorted([folder for folder in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, folder))])

# @st.cache_resource
# def load_model(model_name):
#     model_path = os.path.join(MODEL_DIR, model_name)
#     if model_name.endswith(".pkl"):
#         model_data = joblib.load(model_path)
#         if isinstance(model_data, tuple) and len(model_data) == 2:
#             model, label_encoder = model_data
#         else:
#             model = model_data
#             label_encoder = None
#     else:
#         from tensorflow.keras.models import load_model as keras_load_model
#         model = keras_load_model(model_path)
#         label_encoder = None
#     return model, label_encoder

# def get_combined_hand_box(frame, hand_landmarks_list):
#     h, w, _ = frame.shape
#     boxes = []
#     for hand_landmarks in hand_landmarks_list:
#         x_list = [lm.x * w for lm in hand_landmarks.landmark]
#         y_list = [lm.y * h for lm in hand_landmarks.landmark]
#         xmin, xmax = int(min(x_list)) - 20, int(max(x_list)) + 20
#         ymin, ymax = int(min(y_list)) - 20, int(max(y_list)) + 20
#         xmin, ymin = max(0, xmin), max(0, ymin)
#         xmax, ymax = min(w, xmax), min(h, ymax)
#         boxes.append((xmin, ymin, xmax, ymax))
#     x1 = min(box[0] for box in boxes)
#     y1 = min(box[1] for box in boxes)
#     x2 = max(box[2] for box in boxes)
#     y2 = max(box[3] for box in boxes)
#     return (x1, y1, x2, y2)

# def extract_landmarks(results):
#     landmarks = np.zeros((2, 21, 3), dtype=np.float32)
#     for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
#         if i >= 2:
#             break
#         for j, lm in enumerate(hand_landmarks.landmark):
#             landmarks[i, j] = [lm.x, lm.y, lm.z]
#     return landmarks.flatten().reshape(1, -1)

# # Init
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils

# # UI
# st.set_page_config(page_title="ISL Detection App", layout="centered")
# set_background("Assets/Background.jpg")
# st.title("🧠 Indian Sign Language Recognition")
# st.markdown("Uses MediaPipe for hand detection and ML models trained for ISL (A-Z, 1–9).")


# # Model
# model_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl") or f.endswith(".h5")])
# model_name = st.selectbox("Select a Trained Model", model_files)
# model, label_encoder = load_model(model_name)
# st.success(f"✅ Loaded model: {model_name}")


# # Streamlit WebRTC integration
# class ISLTransformer(VideoTransformerBase):
#     def __init__(self):
#         self.model, _ = load_model(model_name)
#         self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
#         self.last_prediction_time = 0
#         self.current_sentence = st.session_state.get("sentence", "")

#     def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
#         img = frame.to_ndarray(format="bgr24")
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         results = self.hands.process(img_rgb)

#         display_text = "No hands detected."
#         prediction = ""
#         sentence=""
        
#         if results.multi_hand_landmarks:
#             try:
#                 x1, y1, x2, y2 = get_combined_hand_box(img, results.multi_hand_landmarks)
#                 input_data = extract_landmarks(results)

#                 if input_data.shape[1] == NUM_LANDMARKS:
#                     if model_name.endswith(".h5"):
#                         raw_pred = np.argmax(self.model.predict(input_data), axis=1)[0]
#                     else:
#                         raw_pred = self.model.predict(input_data)[0]
#                         if hasattr(raw_pred, "__iter__"):
#                             raw_pred = raw_pred[0]

#                     if isinstance(raw_pred, (int, np.integer)) and raw_pred < len(CLASS_LABELS):
#                         prediction = CLASS_LABELS[raw_pred]
#                         current_time = time.time()
#                         sentence+=prediction

#                         display_text = f"Prediction: {prediction}"
#                         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     else:
#                         display_text = f"Unknown class ({raw_pred})"
#                 else:
#                     display_text = "Landmark data incomplete."

#             except Exception as e:
#                 display_text = f"Prediction error: {str(e)}"

#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#         st.markdown(f"{sentence}")
#         cv2.putText(img, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#         return av.VideoFrame.from_ndarray(img, format="bgr24")
    

# st.markdown("🖐️ Show your ISL sign in front of the webcam.")

# webrtc_streamer(
#     key="isl-stream",
#     video_transformer_factory=ISLTransformer,
#     media_stream_constraints={"video": True, "audio": False},
# )

# # Reference
# st.markdown("---")
# st.subheader("🧾 Reference Image")
# st.image("Assets/Reference.png", caption="Use this as a guide for signs", use_container_width=True)






import streamlit as st
import joblib
import os
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import base64
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
from streamlit_autorefresh import st_autorefresh
import google.generativeai as genai
import os
from dotenv import load_dotenv

# CONFIG
MODEL_DIR = "saved_models"
DATASET_DIR = "dataset"
NUM_LANDMARKS = 21 * 2 * 3

# Background
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Load dataset folder names as class labels
CLASS_LABELS = sorted([folder for folder in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, folder))])

@st.cache_resource
def load_model(model_name):
    model_path = os.path.join(MODEL_DIR, model_name)
    if model_name.endswith(".pkl"):
        model_data = joblib.load(model_path)
        if isinstance(model_data, tuple) and len(model_data) == 2:
            model, label_encoder = model_data
        else:
            model = model_data
            label_encoder = None
    else:
        from tensorflow.keras.models import load_model as keras_load_model
        model = keras_load_model(model_path)
        label_encoder = None
    return model, label_encoder

def get_combined_hand_box(frame, hand_landmarks_list):
    h, w, _ = frame.shape
    boxes = []
    for hand_landmarks in hand_landmarks_list:
        x_list = [lm.x * w for lm in hand_landmarks.landmark]
        y_list = [lm.y * h for lm in hand_landmarks.landmark]
        xmin, xmax = int(min(x_list)) - 20, int(max(x_list)) + 20
        ymin, ymax = int(min(y_list)) - 20, int(max(y_list)) + 20
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(w, xmax), min(h, ymax)
        boxes.append((xmin, ymin, xmax, ymax))
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    return (x1, y1, x2, y2)

def extract_landmarks(results):
    landmarks = np.zeros((2, 21, 3), dtype=np.float32)
    for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
        if i >= 2:
            break
        for j, lm in enumerate(hand_landmarks.landmark):
            landmarks[i, j] = [lm.x, lm.y, lm.z]
    return landmarks.flatten().reshape(1, -1)

# Init
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# UI
st.set_page_config(page_title="ISL Detection App", layout="centered")
set_background("Assets/Background.jpg")
st.title("🧠 Indian Sign Language Recognition")


# Model
model_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl") or f.endswith(".h5")])
model_name = st.selectbox("Select a Trained Model", model_files)
model, label_encoder = load_model(model_name)
st.success(f"✅ Loaded model: {model_name}")

# Streamlit WebRTC integration
class ISLTransformer(VideoTransformerBase):
    def __init__(self):
        self.model, _ = load_model(model_name)
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
        self.sentence = ""
        self.last_prediction = ""
        self.last_confident_prediction = ""
        self.prediction_added = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        display_text = "No hands detected."

        if results.multi_hand_landmarks:
            try:
                x1, y1, x2, y2 = get_combined_hand_box(img, results.multi_hand_landmarks)
                input_data = extract_landmarks(results)

                if input_data.shape[1] == NUM_LANDMARKS:
                    if model_name.endswith(".h5"):
                        preds = self.model.predict(input_data)[0]
                        pred_index = np.argmax(preds)
                        confidence = preds[pred_index]
                    else:
                        preds = self.model.predict_proba(input_data)[0]
                        pred_index = np.argmax(preds)
                        confidence = preds[pred_index]

                    if pred_index < len(CLASS_LABELS):
                        prediction = CLASS_LABELS[pred_index]
                        self.last_prediction = prediction

                        if confidence >= 0.9:
                            display_text = f"Prediction: {prediction} ({confidence:.2f})"
                            if not self.prediction_added or self.last_confident_prediction != prediction:
                                self.sentence += prediction
                                self.last_confident_prediction = prediction
                                self.prediction_added = True
                        else:
                            display_text = f"Confidence: {confidence:.2f} | Last: {self.last_confident_prediction}"

                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    else:
                        display_text = f"Unknown class ({pred_index})"
                else:
                    display_text = "Landmark data incomplete."

            except Exception as e:
                display_text = f"Prediction error: {str(e)}"

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            # Reset flag so next confident prediction can update sentence
            self.prediction_added = False
            display_text = "No hands detected."

        cv2.putText(img, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")



st.markdown("🖐️ Show your ISL sign in front of the webcam.")

webrtc_ctx = webrtc_streamer(
    key="isl-stream",
    video_transformer_factory=ISLTransformer,
    media_stream_constraints={"video": True, "audio": False},
)



# After importing necessary libraries and initializing app...

st.markdown("### Detected Sentence:")
sentence_placeholder = st.empty()

# Initialize session state
if "saved_sentence" not in st.session_state:
    st.session_state.saved_sentence = ""

# Periodically refresh while webcam is on
if webrtc_ctx.video_transformer:
    current_sentence = webrtc_ctx.video_transformer.sentence
    if current_sentence != st.session_state.saved_sentence:
        st.session_state.saved_sentence = current_sentence
    st_autorefresh(interval=2000, limit=None, key="sentence_refresh")

# Show the sentence
sentence_placeholder.markdown(f"**{st.session_state.saved_sentence}**")

ss=st.session_state.saved_sentence


# Gemini API setup
load_dotenv()
genai.configure(api_key=os.getenv("GENAI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

def correct_sentence(s: str) -> str:
    prompt = f"Given a sequence of predicted letters from an Indian Sign Language (ISL) fingerspelling recognition system, where the letters (A-Z, 0-9) represent individual hand gestures captured in real-time, construct a meaningful sentence or phrase. The sequence may contain errors due to mispredictions, and some letters might be part of words like 'HELLO', 'HOW', 'ARE', 'YOU', or numbers like '123'. Add context based on common conversational patterns, correct minor errors if possible, and form a coherent sentence or phrase. If the sequence is incomplete or ambiguous, make a reasonable interpretation. Here is the sequence of predicted letters: {s}. Give response to the point.Only give the word or sentence nothing else, just the answer"
    response = model.generate_content(prompt)
    return response.text.strip()

languages = [
    "English", "Hindi", "Bengali", "Telugu", "Marathi", "Tamil", "Gujarati", 
    "Urdu", "Kannada", "Odia", "Punjabi", "Malayalam", "Assamese", "Maithili", 
    "Santali", "Kashmiri", "Nepali", "Konkani", "Dogri", "Manipuri", "Bodo", 
    "Sindhi", "Sanskrit"
]


def translate_text(text: str, language: str) -> str:
    prompt = f"Translate this sentence to {language}. Only give translation:\n{text}"
    response = model.generate_content(prompt)
    return response.text.strip()



# Buttons
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🧹 Clear"):
        st.session_state.saved_sentence = ""
        st.session_state.corrected_sentence = ""
        st.session_state.translated_sentence = ""
        st.rerun()

with col2:
    if st.button("🔊 Speak"):
        js_code = f"""
        <script>
        var msg = new SpeechSynthesisUtterance("{ss}");
        window.speechSynthesis.speak(msg);
        </script>
        """
        st.components.v1.html(js_code)

with col3:
    if st.button("✍️ Correct"):
        corrected = correct_sentence(ss)
        st.session_state.corrected_sentence = corrected
        st.write(f"{corrected}")
selected_language = st.selectbox("Select language for translation:", languages)
with col4:
    
    if st.button("🌐 Translate"):
        if st.session_state.corrected_sentence:
            to_translate = st.session_state.corrected_sentence
        else:
            to_translate = ss
        translated = translate_text(to_translate, selected_language)
        st.session_state.translated_sentence = translated
        st.write(f"{translated}")



# Reference
st.markdown("---")
st.subheader("🧾 Reference Image")
st.image("Assets/Reference.png", caption="Use this as a guide for signs", use_container_width=True)






