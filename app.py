import os
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import mediapipe as mp
from datetime import datetime
import pandas as pd
import av
import pickle
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Configuration for WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Initialize MediaPipe solutions
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Constants
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_LOG = "attendance.csv"
MODEL_PATH = "face_recognition_model.pkl"
THRESHOLD = 0.6  # Confidence threshold for face recognition

# Create necessary directories and files if they don't exist
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
if not os.path.exists(ATTENDANCE_LOG):
    pd.DataFrame(columns=["Name", "Time", "Status"]).to_csv(ATTENDANCE_LOG, index=False)

class AttendanceTracker(VideoTransformerBase):
    def __init__(self):
        # Initialize models
        self.face_detector = mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        self.hands_detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        
        # Load or initialize face recognition model
        if os.path.exists(MODEL_PATH):
            self.face_recognizer = joblib.load(MODEL_PATH)
            with open(os.path.join(KNOWN_FACES_DIR, "encodings.pkl"), "rb") as f:
                self.known_face_encodings, self.known_face_names = pickle.load(f)
        else:
            self.face_recognizer = KNeighborsClassifier(n_neighbors=3)
            self.known_face_encodings = []
            self.known_face_names = []
        
        # Attendance tracking
        self.last_detected = {}
        self.cooldown = 30  # seconds between attendance records for the same person
        
        # Mode flags
        self.face_recognition_mode = True
        self.hand_tracking_mode = True
    
    def set_modes(self, face_mode, hand_mode):
        self.face_recognition_mode = face_mode
        self.hand_tracking_mode = hand_mode
    
    def register_face(self, name, frame):
        """Register a new face in the system"""
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.face_detector.process(rgb_frame)
        
        if results.detections:
            # Get the first face (assuming one face for registration)
            face = results.detections[0]
            
            # Extract face bounding box
            ih, iw, _ = frame.shape
            bbox = face.location_data.relative_bounding_box
            x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), \
                          int(bbox.width * iw), int(bbox.height * ih)
            
            # Extract face ROI and compute encoding
            face_roi = rgb_frame[y:y+h, x:x+w]
            encoding = self._compute_face_encoding(face_roi)
            
            if encoding is not None:
                # Add to known faces
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(name)
                
                # Save the face image for reference
                cv2.imwrite(os.path.join(KNOWN_FACES_DIR, f"{name}.jpg"), frame[y:y+h, x:x+w])
                
                # Retrain the model
                self.face_recognizer.fit(self.known_face_encodings, self.known_face_names)
                
                # Save the updated model and encodings
                joblib.dump(self.face_recognizer, MODEL_PATH)
                with open(os.path.join(KNOWN_FACES_DIR, "encodings.pkl"), "wb") as f:
                    pickle.dump((self.known_face_encodings, self.known_face_names), f)
                
                return True
        
        return False
    
    def _compute_face_encoding(self, face_roi):
        """Compute face encoding using a simple feature extraction method"""
        # In a production app, you'd use a more sophisticated method like FaceNet
        # Here we use a simplified approach for demonstration
        
        # Resize and flatten the face ROI
        try:
            resized = cv2.resize(face_roi, (64, 64))
            return resized.flatten()
        except:
            return None
    
    def recognize_face(self, face_roi):
        """Recognize a face from the given ROI"""
        if not self.known_face_encodings:
            return None, 0
        
        # Compute encoding for the detected face
        encoding = self._compute_face_encoding(face_roi)
        if encoding is None:
            return None, 0
        
        # Predict using KNN
        distances, indices = self.face_recognizer.kneighbors([encoding], n_neighbors=1)
        
        # Check if the distance is below threshold
        if distances[0][0] < THRESHOLD:
            return self.known_face_names[indices[0][0]], distances[0][0]
        else:
            return None, distances[0][0]
    
    def record_attendance(self, name):
        """Record attendance for the recognized person"""
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        
        # Check if this person was recently detected
        if name in self.last_detected:
            last_time = self.last_detected[name]
            if (now - last_time).seconds < self.cooldown:
                return False
        
        # Record attendance
        new_entry = pd.DataFrame([[name, timestamp, "Present"]], 
                                columns=["Name", "Time", "Status"])
        
        # Append to log file
        existing_log = pd.read_csv(ATTENDANCE_LOG)
        updated_log = pd.concat([existing_log, new_entry], ignore_index=True)
        updated_log.to_csv(ATTENDANCE_LOG, index=False)
        
        # Update last detected time
        self.last_detected[name] = now
        
        return True
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Convert to RGB for processing
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.face_recognition_mode:
            # Face detection and recognition
            face_results = self.face_detector.process(rgb_img)
            
            if face_results.detections:
                for detection in face_results.detections:
                    # Get face bounding box
                    ih, iw, _ = img.shape
                    bbox = detection.location_data.relative_bounding_box
                    x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), \
                                  int(bbox.width * iw), int(bbox.height * ih)
                    
                    # Draw face bounding box
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Extract face ROI for recognition
                    face_roi = rgb_img[y:y+h, x:x+w]
                    
                    # Recognize face
                    name, confidence = self.recognize_face(face_roi)
                    
                    if name:
                        # Display name and confidence
                        label = f"{name} ({confidence:.2f})"
                        cv2.putText(img, label, (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                        # Record attendance
                        self.record_attendance(name)
                    else:
                        cv2.putText(img, "Unknown", (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        if self.hand_tracking_mode:
            # Hand tracking
            hand_results = self.hands_detector.process(rgb_img)
            
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    # Draw hand landmarks and connections
                    mp_drawing.draw_landmarks(
                        img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                    )
        
        return img

def main():
    st.title("Advanced Attendance System")
    st.markdown("""
    This application combines face recognition for attendance tracking with real-time hand skeleton visualization.
    - **Face Recognition**: Detects and recognizes faces, recording attendance automatically
    - **Hand Tracking**: Visualizes hand landmarks in real-time
    """)
    
    # Initialize tracker
    if 'tracker' not in st.session_state:
        st.session_state.tracker = AttendanceTracker()
    
    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        
        # Mode selection
        face_mode = st.checkbox("Enable Face Recognition", value=True)
        hand_mode = st.checkbox("Enable Hand Tracking", value=True)
        st.session_state.tracker.set_modes(face_mode, hand_mode)
        
        # Face registration
        st.subheader("Register New Face")
        reg_name = st.text_input("Enter name for registration")
        if st.button("Register from Camera"):
            # This would need a separate capture mechanism in a real app
            st.warning("Face registration requires implementation of a capture mechanism")
    
    # Main content
    tab1, tab2 = st.tabs(["Live Camera", "Attendance Log"])
    
    with tab1:
        st.header("Live Camera Feed")
        st.info("Allow camera access when prompted")
        
        # WebRTC streamer
        ctx = webrtc_streamer(
            key="attendance",
            video_processor_factory=AttendanceTracker,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        if ctx.video_processor:
            ctx.video_processor.set_modes(face_mode, hand_mode)
    
    with tab2:
        st.header("Attendance Records")
        
        if os.path.exists(ATTENDANCE_LOG):
            attendance_df = pd.read_csv(ATTENDANCE_LOG)
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                name_filter = st.selectbox(
                    "Filter by name",
                    ["All"] + sorted(attendance_df["Name"].unique().tolist())
                )
            
            with col2:
                date_filter = st.date_input("Filter by date")
            
            # Apply filters
            if name_filter != "All":
                attendance_df = attendance_df[attendance_df["Name"] == name_filter]
            
            if date_filter:
                attendance_df = attendance_df[
                    pd.to_datetime(attendance_df["Time"]).dt.date == date_filter
                ]
            
            # Display table
            st.dataframe(attendance_df, height=400)
            
            # Export button
            if st.button("Export to CSV"):
                st.download_button(
                    label="Download CSV",
                    data=attendance_df.to_csv(index=False),
                    file_name=f"attendance_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )
        else:
            st.warning("No attendance records found")

if __name__ == "__main__":
    main()
    
