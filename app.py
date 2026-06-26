import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. KONFIGURASI HALAMAN UTAMA (UI/UX)
# ==========================================
st.set_page_config(
    page_title="ChurnLens | AI Prediction",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk membuat tampilan ala Startup (Tombol lebih modern, kartu, dsb)
st.markdown("""
    <style>
    /* Styling untuk header utama */
    .main-title {
        font-size: 3rem !important;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px !important;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    /* Styling untuk tombol prediksi */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: black;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
    /* Styling untuk kartu hasil */
    .result-card {
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 2. SIDEBAR EKSKLUSIF
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100) # Placeholder logo
    st.markdown("### 🚀 ChurnLens AI")
    st.caption("v1.0.0 - Project Bengkel Koding")
    st.divider()
    st.markdown("""
        **Tentang Aplikasi:**
        Sistem ini menggunakan *Machine Learning* untuk mendeteksi sinyal *churn* dari perilaku pelanggan secara *real-time*.
        
        **Cara Penggunaan:**
        1. Isi form karakteristik di layar utama.
        2. Klik tombol **Analisis Sekarang**.
        3. Dapatkan *insight* & rekomendasi.
    """)
    st.divider()
    st.caption("© 2026 Data Science Team")


# ==========================================
# 3. MEMUAT ASET MODEL (CACHING)
# ==========================================
@st.cache_resource
def load_assets():
    model = joblib.load('model_churn_terbaik.pkl')
    preprocessor = joblib.load('preprocessor_churn.pkl')
    scaler = joblib.load('scaler_churn.pkl')
    features = joblib.load('X_columns.pkl')
    return model, preprocessor, scaler, features

try:
    model, preprocessor, scaler, features_columns = load_assets()
    model_loaded = True
except Exception as e:
    st.error(f"⚠️ Sistem gagal memuat *engine* AI. Pastikan file `.pkl` tersedia. Error: {e}")
    model_loaded = False


# ==========================================
# 4. HEADER APLIKASI
# ==========================================
st.markdown('<p class="main-title">Customer Intelligence Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Kenali perilaku pelangganmu sebelum mereka pergi. Masukkan data di bawah ini untuk memulai analisis.</p>', unsafe_allow_html=True)


# ==========================================
# 5. FORM INPUT MODERN MENGGUNAKAN TABS
# ==========================================
# Memecah 25 input menjadi 3 kategori agar tidak menumpuk dan terlihat bersih
tab1, tab2, tab3 = st.tabs(["👤 Demografi & Akuisisi", "📱 Perilaku Platform", "💳 Finansial & Kepuasan"])

with tab1:
    st.markdown("#### Profil Dasar Pelanggan")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        age = st.number_input("Usia", min_value=1, max_value=120, value=28, step=1)
        country = st.text_input("Negara Domisili", value="Indonesia")
    with col2:
        city = st.text_input("Kota Domisili", value="Jakarta")
        acquisition_channel = st.selectbox("Sumber Akuisisi", ["Email", "Ads", "Referral", "Organic"])
        device_type = st.selectbox("Perangkat Utama", ["Mobile", "Desktop", "Tablet"])

with tab2:
    st.markdown("#### Aktivitas & Penggunaan Layanan")
    col3, col4 = st.columns(2)
    with col3:
        subscription_type = st.selectbox("Tipe Langganan", ["Basic", "Standard", "Premium"])
        is_premium_user = st.radio("Status Pengguna Premium?", [1, 0], format_func=lambda x: "👑 Ya" if x == 1 else "Member Biasa", horizontal=True)
        total_visits = st.slider("Total Kunjungan Platform", 0, 100, 15)
        pages_per_session = st.slider("Rata-rata Halaman per Sesi", 0.0, 20.0, 5.0)
    with col4:
        avg_session_time = st.number_input("Durasi Sesi Rata-rata (Menit)", min_value=0.0, value=20.5)
        email_open_rate = st.number_input("Email Open Rate (%)", min_value=0.0, max_value=100.0, value=65.0)
        email_click_rate = st.number_input("Email Click Rate (%)", min_value=0.0, max_value=100.0, value=15.0)

with tab3:
    st.markdown("#### Transaksi & Feedback Pelanggan")
    col5, col6, col7 = st.columns(3)
    with col5:
        total_spent = st.number_input("Total Pengeluaran", min_value=0.0, value=1500000.0, step=50000.0)
        avg_order_value = st.number_input("AOV (Rata-rata Transaksi)", min_value=0.0, value=250000.0, step=10000.0)
        lifetime_value = st.number_input("Customer Lifetime Value (CLV)", min_value=0.0, value=3000000.0, step=100000.0)
        marketing_spend_per_user = st.number_input("Biaya Marketing per User", min_value=0.0, value=50000.0, step=5000.0)
    with col6:
        payment_method = st.selectbox("Metode Pembayaran", ["Credit Card", "Bank Transfer", "E-Wallet"])
        discount_used = st.selectbox("Sering Pakai Diskon?", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")
        refund_requested = st.selectbox("Pernah Minta Refund?", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")
        last_3_month_purchase_freq = st.number_input("Frekuensi Beli (3 Bln Terakhir)", min_value=0, value=4, step=1)
    with col7:
        satisfaction_score = st.slider("Skor Kepuasan (CSAT)", 1.0, 5.0, 4.5, step=0.1)
        nps_score = st.slider("Skor NPS", 0, 10, 9)
        support_tickets = st.number_input("Jumlah Komplain/Tiket", min_value=0, value=0, step=1)
        delivery_delay_days = st.number_input("Keterlambatan Layanan/Kirim (Hari)", min_value=0, value=0, step=1)

st.write("---")

# ==========================================
# 6. LOGIKA PREDIKSI & HASIL TAMPILAN
# ==========================================
# Membungkus tombol prediksi di tengah
_, center_btn, _ = st.columns([1, 2, 1])

with center_btn:
    analyze_button = st.button("✨ Analisis Sekarang")

if analyze_button and model_loaded:
    with st.spinner("Menganalisis pola pelanggan..."):
        # Mapping nama kolom harus SANGAT SESUAI dengan model sebelumnya
        input_data = pd.DataFrame([{
            'gender': gender, 'age': age, 'country': country, 'city': city,
            'acquisition_channel': acquisition_channel, 'device_type': device_type,
            'subscription_type': subscription_type, 'is_premium_user': is_premium_user,
            'total_visits': total_visits, 'avg_session_time': avg_session_time,
            'pages_per_session': pages_per_session, 'email_open_rate': email_open_rate,
            'email_click_rate': email_click_rate, 'total_spent': total_spent,
            'avg_order_value': avg_order_value, 'discount_used': discount_used,
            'support_tickets': support_tickets, 'refund_requested': refund_requested,
            'delivery_delay_days': delivery_delay_days, 'payment_method': payment_method,
            'satisfaction_score': satisfaction_score, 'nps_score': nps_score,
            'marketing_spend_per_user': marketing_spend_per_user, 'lifetime_value': lifetime_value,
            'last_3_month_purchase_freq': last_3_month_purchase_freq
        }])
        
        # Preprocessing Data User
        input_preprocessed = preprocessor.transform(input_data)
        input_preprocessed_df = pd.DataFrame(input_preprocessed, columns=features_columns)
        input_scaled = scaler.transform(input_preprocessed_df)
        
        # Eksekusi Model
        prediksi = model.predict(input_scaled)
        probabilitas = model.predict_proba(input_scaled)[0][1]
        
        # Menampilkan Hasil dengan UI Card ala Startup
        st.markdown("### 📊 Hasil Intelijensi")
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.metric(label="Probabilitas Churn", value=f"{probabilitas * 100:.1f}%")
        
        with res_col2:
            if prediksi[0] == 1:
                st.markdown("""
                <div style="background-color: #ffe6e6; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4b4b;">
                    <h4 style="color: #cc0000; margin-top: 0;">⚠️ Pelanggan Berisiko Tinggi (CHURN)</h4>
                    <p style="color: black;">Algoritma mendeteksi pola yang mengindikasikan pelanggan ini akan meninggalkan layanan. Segera lakukan tindakan retensi!</p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("💡 Lihat Rekomendasi Tindakan (Actionable Insight)"):
                    st.write("- Kirimkan kampanye *win-back* atau email personalisasi dengan diskon khusus.")
                    st.write("- Tugaskan tim *Customer Success* untuk menghubungi pelanggan dan menanyakan kendala layanan.")
                    st.write("- Tawarkan perpanjangan layanan dengan *upgrade* fitur secara gratis selama 1 bulan.")
            else:
                st.markdown("""
                <div style="background-color: #e6ffe6; padding: 20px; border-radius: 10px; border-left: 5px solid #00cc66;">
                    <h4 style="color: #008040; margin-top: 0;">✅ Pelanggan Setia (RETAINED)</h4>
                    <p style="color: black;">Pelanggan ini menunjukkan pola loyalitas yang kuat dan cenderung bertahan menggunakan platform.</p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("💡 Lihat Rekomendasi Tindakan (Actionable Insight)"):
                    st.write("- Pertahankan tingkat layanan (*Service Level Agreement*) saat ini.")
                    st.write("- Targetkan pelanggan ini untuk kampanye *Up-Selling* atau *Cross-Selling* produk premium.")
                    st.write("- Tawarkan program *Referral* agar mereka bisa mengajak rekan-rekannya bergabung.")