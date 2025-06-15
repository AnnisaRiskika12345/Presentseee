import streamlit as st
import pandas as pd

st.set_page_config(page_title="Homepage", layout="centered")

st.title("👋 Selamat Datang di Sistem Absensi")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');

/* === Base Theme === */
html, body, [class*="css"] {
    background: radial-gradient(circle at top left, #0e0f1a, #12131f);
    color: #ffffff !important;
    font-family: 'Orbitron', sans-serif;
}

/* === Headings === */
h1, h2, h3, h4 {
    color: #00ffff;
    text-align: center;
    letter-spacing: 1.2px;
    text-transform: uppercase;
}

/* === Buttons === */
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

/* === Sidebar === */
section[data-testid="stSidebar"] {
    background-color: #1a1d2b;
    color: #ffffff !important;
    font-family: 'Orbitron', sans-serif;
    border-right: 1px solid #2ec4f1;
}

section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* === Inputs === */
input, textarea, .stTextInput>div>div>input {
    background-color: #222430 !important;
    color: #ffffff !important;
    border: 1px solid #00ffff !important;
    border-radius: 6px;
}

/* === Selectbox and Dropdown === */
div[data-baseweb="select"] > div {
    background-color: #1e202b !important;
    color: #ffffff !important;
}

/* === Slider and other widgets === */
.css-1cpxqw2 {
    background-color: #00c9ff !important;
}

/* === Optional scroll glow === */
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

st.markdown("""

Kami percaya bahwa kehadiran bukan sekadar angka, tapi bentuk komitmen dan kebersamaan.  
**Presentsee** hadir sebagai solusi **absensi pintar** yang mengutamakan kemudahan, kecepatan, dan kenyamanan.

✨ Dengan teknologi **pengenalan wajah dan gesture**, Anda cukup hadir di depan kamera—tanpa repot menulis atau menekan tombol.

Beberapa fitur utama yang kami siapkan untuk Anda:

🔹 **Absensi berbasis Wajah & Gesture** – Karena kehadiran bisa lebih alami.  
🔹 **Pemilihan Departemen** – Untuk data yang lebih rapi dan sesuai tim Anda.  
🔹 **Pencatatan waktu otomatis** – Tak perlu lagi mencatat manual.  

---

Silakan pilih departemen Anda di bawah untuk memulai.  
**Terima kasih telah hadir bersama Presentsee.** 🌱
""")


# Cek apakah file karyawan.csv tersedia
def load_departemen():
    try:
        df = pd.read_csv("karyawan.csv")
        return sorted(df["department"].unique())
    except Exception as e:
        st.warning("⚠️ Tidak bisa memuat daftar departemen. Pastikan file 'karyawan.csv' tersedia dan benar.")
        return []

departemen_list = load_departemen()

selected_dept = st.selectbox("📁 Pilih Departemen", [""] + departemen_list)

if selected_dept:
    st.success(f"✅ Anda memilih departemen: **{selected_dept}**")
    st.markdown("👉 Lanjutkan ke halaman **Absensi** untuk mulai menggunakan kamera.")

    st.session_state["selected_department"] = selected_dept

# Pindah ke halaman app.py (halaman absensi)
    st.switch_page("app.py")