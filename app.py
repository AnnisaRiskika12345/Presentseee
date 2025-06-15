import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import csv
import random
import glob

# =======================
# Page Configuration
# =======================
st.set_page_config(
    page_title="PresentSee",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');

/* Base Theme */
html, body, [class*="css"] {
    background: radial-gradient(circle at top left, #0e0f1a, #12131f);
    color: #ffffff !important;
    font-family: 'Orbitron', sans-serif;
}

/* Headings */
h1, h2, h3, h4 {
    color: #00ffff;
    text-align: center;
    letter-spacing: 1.2px;
    text-transform: uppercase;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #00c9ff, #92fe9d);
    color: #000000;
    font-weight: 700;
    font-size: 16px;
    border: none;
    border-radius: 12px;
    padding: 12px 26px;
    transition: 0.3s ease-in-out;
    box-shadow: 0 0 12px rgba(0, 201, 255, 0.3);
}

.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 18px rgba(0, 201, 255, 0.6);
    background: linear-gradient(90deg, #92fe9d, #00c9ff);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1a1d2b;
    color: #ffffff !important;
    font-family: 'Orbitron', sans-serif;
    border-right: 1px solid #2ec4f1;
}

section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Inputs */
input, textarea, .stTextInput>div>div>input {
    background-color: #222430 !important;
    color: #ffffff !important;
    border: 1px solid #00ffff !important;
    border-radius: 6px;
}

/* Selectbox and Dropdown */
div[data-baseweb="select"] > div {
    background-color: #1e202b !important;
    color: #ffffff !important;
}

/* Slider and other widgets */
.css-1cpxqw2 {
    background-color: #00c9ff !important;
}

/* Optional scroll glow */
::-webkit-scrollbar {
    width: 10px;
}
::-webkit-scrollbar-track {
    background: #12131f;
}
::-webkit-scrollbar-thumb {
    background: #00ffff;
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Welcome to PresentSee")
st.subheader("Futuristic Face & Gesture Attendance System")

# =======================
# Utility Functions
# =======================
def init_directories():
    """Initialize required directories"""
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    if not os.path.exists("karyawan.csv"):
        with open("karyawan.csv", mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "age", "position", "department"])

def get_next_id():
    """Get next employee ID"""
    if os.path.exists("karyawan.csv"):
        df = pd.read_csv("karyawan.csv")
        if not df.empty and 'id' in df.columns:
            return df['id'].max() + 1
    return 1

def load_employee_names():
    """Load ID to name mapping"""
    if os.path.exists("karyawan.csv"):
        df = pd.read_csv("karyawan.csv")
        if not df.empty and 'id' in df.columns and 'name' in df.columns:
            return {str(row['id']): row['name'] for _, row in df.iterrows()}
    return {}

def get_all_models():
    """Load all face recognition models"""
    models = []
    model_files = glob.glob("models/face-model*.yml")
    
    for model_file in model_files:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(model_file)
        models.append(recognizer)
    
    return models

def predict_face(roi_gray, models, names):
    """Predict face using all available models"""
    predictions = []
    
    for recognizer in models:
        id, conf = recognizer.predict(roi_gray)
        predictions.append((str(id), conf))
    
    if not predictions:
        return "Unknown", 0
    
    best_id, best_conf = min(predictions, key=lambda x: x[1])
    name = names.get(best_id, "Unknown")
    confidence = max(0, 100 - best_conf)
    
    return name if confidence >= 40 else "Unknown", confidence

def detect_and_recognize_faces(frame, gray, recognizers, names, faceCascade):
    """Enhanced face detection and recognition"""
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(80, 80),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    results = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (200, 200))
        roi_gray = cv2.equalizeHist(roi_gray)
        
        name, confidence = predict_face(roi_gray, recognizers, names)
        
        results.append({
            'x': x, 'y': y, 'w': w, 'h': h,
            'name': name,
            'confidence': confidence,
            'roi': roi_gray
        })
    
    return results

# =======================
# Initialize
# =======================
init_directories()

# =======================
# Add New Employee
# =======================

with st.expander("➕ Tambahkan Data Wajah Baru"):
    with st.form("form_tambah_data"):
        nama_baru = st.text_input("Nama Lengkap")
        umur_baru = st.text_input("Umur")
        posisi_baru = st.text_input("Posisi")
        dept_baru = st.text_input("Departemen")
        submit_data = st.form_submit_button("Tambahkan dan Ambil Wajah")

def tambah_data_dan_train(nama, umur, posisi, departemen):
    """Add new employee and train model"""
    init_directories()
    new_id = get_next_id()
    data_baru = pd.DataFrame([[new_id, nama, umur, posisi, departemen]], 
                           columns=["id", "name", "age", "position", "department"])

    if os.path.exists("karyawan.csv"):
        df = pd.read_csv("karyawan.csv")
        if not df.empty and 'name' in df.columns and nama.lower() in df['name'].str.lower().values:
            return f"⚠️ Nama '{nama}' sudah terdaftar!"
        
        if df.empty or 'id' not in df.columns:
            df = data_baru
        else:
            df = pd.concat([df, data_baru], ignore_index=True)
    else:
        df = data_baru
    
    df.to_csv("karyawan.csv", index=False)

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    count = 0
    st.info("📸 Kamera aktif. Tekan Q untuk berhenti...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = faceCascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))
            face_img = cv2.equalizeHist(face_img)
            cv2.imwrite(f"dataset/Person-{new_id}-{count}.jpg", face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Ambil Wajah", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 100:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Train model with augmentation
    imagePaths = [os.path.join("dataset", f) for f in os.listdir("dataset") if f.startswith("Person-")]
    faces, ids = [], []
    
    for path in imagePaths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        id = int(os.path.basename(path).split("-")[1])
        
        # Original
        faces.append(img)
        ids.append(id)
        
        # Flipped
        faces.append(cv2.flip(img, 1))
        ids.append(id)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))

    model_number = len(glob.glob("models/face-model*.yml")) + 1
    model_filename = f"models/face-model{model_number}.yml"
    recognizer.write(model_filename)

    return f"✅ Data wajah '{nama}' (ID: {new_id}) berhasil ditambahkan. Model disimpan sebagai {model_filename}"

if submit_data:
    if not all([nama_baru, umur_baru, posisi_baru, dept_baru]):
        st.warning("❗ Semua kolom harus diisi.")
    else:
        with st.spinner("Sedang memproses data dan mengambil wajah..."):
            hasil = tambah_data_dan_train(nama_baru, umur_baru, posisi_baru, dept_baru)
        st.success(hasil)

# =======================
# Load Employee Data
# =======================
@st.cache_data
def load_employee_data():
    if os.path.exists("karyawan.csv"):
        return pd.read_csv("karyawan.csv")
    return pd.DataFrame(columns=["id", "name", "age", "position", "department"])

employee_df = load_employee_data()

# =======================
# Sidebar
# =======================
st.sidebar.title("\U0001F530 Halaman Utama")
department_options = sorted(employee_df['department'].unique()) if not employee_df.empty else []
selected_dept = st.sidebar.selectbox("Pilih Departemen", department_options) if department_options else ""
st.sidebar.markdown("---")

if st.sidebar.checkbox("📅 Lihat Absensi Hari Ini"):
    if os.path.isfile("absensi.csv"):
        df_absen = pd.read_csv("absensi.csv")
        today = datetime.now().strftime("%Y-%m-%d")
        st.sidebar.dataframe(df_absen[df_absen['Waktu'].str.startswith(today)])
    else:
        st.sidebar.info("Belum ada data absensi hari ini.")

if st.sidebar.checkbox("👥 Tampilkan Pegawai") and selected_dept:
    st.sidebar.dataframe(employee_df[employee_df['department'] == selected_dept])

# =======================
# Attendance Functions
# =======================
def is_authorized(name, department):
    row = employee_df[(employee_df['name'].str.lower() == name.lower()) &
                      (employee_df['department'].str.lower() == department.lower())]
    return not row.empty

def log_attendance(name):
    row = employee_df[employee_df['name'].str.lower() == name.lower()].iloc[0]
    filename = "absensi.csv"
    waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = [row['name'], row['age'], row['position'], row['department'], waktu, "Hadir"]

    if not os.path.isfile(filename):
        with open(filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Nama", "Usia", "Posisi", "Departemen", "Waktu", "Status"])

    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(data)

def has_attended_today(name):
    if not os.path.isfile("absensi.csv"):
        return False
    
    df = pd.read_csv("absensi.csv")
    today = datetime.now().strftime("%Y-%m-%d")
    return not df[(df['Nama'].str.lower() == name.lower()) & (df['Waktu'].str.startswith(today))].empty

# =======================
# Load Models
# =======================
@st.cache_resource
def load_system_models():
    models = get_all_models()
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    smileCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    draw_utils = mp.solutions.drawing_utils
    names = load_employee_names()
    
    return models, faceCascade, smileCascade, hands, draw_utils, names

recognizers, faceCascade, smileCascade, hands, draw_utils, names = load_system_models()

# =======================
# Gesture Functions
# =======================
def count_fingers(hand_landmarks, hand_label):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    if hand_label == "Right":
        fingers.append(hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0] - 1].x)
    else:
        fingers.append(hand_landmarks.landmark[tips[0]].x > hand_landmarks.landmark[tips[0] - 1].x)

    for tip in tips[1:]:
        fingers.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y)

    return fingers.count(True)

# =======================
# Session State
# =======================
if 'attendance_done' not in st.session_state:
    st.session_state.attendance_done = {}

if 'unknown_start_time' not in st.session_state:
    st.session_state.unknown_start_time = None

if 'target_fingers' not in st.session_state:
    st.session_state.target_fingers = {
        'Right': random.randint(1, 5),
        'Left': random.randint(1, 5)
    }

if st.button("🔁 Reset Gesture Target"):
    st.session_state.target_fingers = {
        'Right': random.randint(1, 5),
        'Left': random.randint(1, 5)
    }
    st.session_state.attendance_done = {}
    st.success("Target gesture berhasil direset.")

# =======================
# Camera Interface
# =======================
FRAME_WINDOW = st.image([])
run = st.checkbox("▶️ Mulai Kamera (Mirror Mode)")
debug_mode = st.checkbox("Enable Debug Mode")
success_message = ""

st.info(f"Tunjukkan tangan kanan {st.session_state.target_fingers['Right']} jari dan tangan kiri {st.session_state.target_fingers['Left']} jari untuk absen.")

if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Gagal membuka kamera.")
    else:
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Gagal membaca frame dari kamera.")
                break

            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            # Hand detection
            hand_results = hands.process(frameRGB)
            finger_counts = {'Left': 0, 'Right': 0}

            if hand_results.multi_hand_landmarks:
                for landmark, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                    label = handedness.classification[0].label
                    draw_utils.draw_landmarks(frame, landmark, mp.solutions.hands.HAND_CONNECTIONS)
                    finger_counts[label] = count_fingers(landmark, label)
                    cv2.putText(frame, f"{label} Fingers: {finger_counts[label]}",
                                (10, 60 if label == "Right" else 90),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)

            # Face detection and recognition
            face_results = detect_and_recognize_faces(frame, gray, recognizers, names, faceCascade)
            
            if debug_mode:
                debug_cols = st.columns(len(face_results))
            
            for i, face in enumerate(face_results):
                x, y, w, h = face['x'], face['y'], face['w'], face['h']
                name = face['name']
                confidence = face['confidence']
                
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                if debug_mode and i < len(debug_cols):
                    with debug_cols[i]:
                        st.image(face['roi'], caption=f"{name} ({confidence}%)")
                        st.write(f"Position: ({x},{y}) Size: {w}x{h}")
                        st.write(f"Fingers: {finger_counts}")

                if name != "Unknown":
                    st.session_state.unknown_start_time = None
                    row = employee_df[employee_df['name'].str.lower() == name.lower()]
                    
                    if not row.empty:
                        age = row.iloc[0]['age']
                        position = row.iloc[0]['position']
                        department = row.iloc[0]['department']

                        cv2.putText(frame, f"{name} ({confidence}%)", (x, y - 60), 
                                   cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 0), 2)
                        cv2.putText(frame, f"Umur: {age}", (x, y - 40), 
                                   cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 255, 255), 1)
                        cv2.putText(frame, f"{position}, {department}", (x, y - 20), 
                                   cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 255, 255), 1)

                        smile = smileCascade.detectMultiScale(face['roi'], 
                                                            scaleFactor=1.8, 
                                                            minNeighbors=20)
                        smiling = len(smile) > 0
                        if smiling:
                            cv2.putText(frame, "Smiling", (x, y + h + 30), 
                                       cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 2)

                        if (smiling and
                            finger_counts["Right"] == st.session_state.target_fingers['Right'] and
                            finger_counts["Left"] == st.session_state.target_fingers['Left'] and
                            name not in st.session_state.attendance_done):

                            if has_attended_today(name):
                                success_message = f"⚠ Anda sudah absen hari ini, {name}."
                            elif is_authorized(name, selected_dept):
                                log_attendance(name)
                                st.session_state.attendance_done[name] = True
                                success_message = f"✅ Absensi berhasil untuk {name}."
                            else:
                                st.session_state.attendance_done[name] = True
                                success_message = f"❌ Absensi ditolak untuk {name}, bukan dari departemen {selected_dept}."

                            run = False
                            break
                else:
                    current_time = datetime.now()
                    if st.session_state.unknown_start_time is None:
                        st.session_state.unknown_start_time = current_time
                        cv2.putText(frame, "Menganalisis wajah...", (x, y - 10), 
                                   cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 255), 2)
                    else:
                        elapsed = (current_time - st.session_state.unknown_start_time).total_seconds()
                        if elapsed >= 5:
                            cv2.putText(frame, "Unknown", (x, y - 10), 
                                       cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
                            if "Unknown" not in st.session_state.attendance_done:
                                st.session_state.attendance_done["Unknown"] = True
                                success_message = "❌ Absensi gagal: Wajah tidak dikenali (Unknown) setelah 5 detik."
                                run = False
                                break
                        else:
                            countdown = int(5 - elapsed)
                            cv2.putText(frame, f"Menunggu wajah valid... ({countdown}s)", 
                                       (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (50, 100, 255), 2)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()

    if "✅" in success_message:
        st.success(success_message)
    elif "⚠" in success_message:
        st.warning(success_message)
    else:
        st.error(success_message)
else:
    st.info("Klik '\u25B6\ufe0f Mulai Kamera' untuk memulai absensi.")
