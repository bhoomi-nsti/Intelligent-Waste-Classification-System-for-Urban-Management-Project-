import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import tensorflow as tf
import numpy as np
import cv2
import av
import pandas as pd
import time
from PIL import Image, ImageOps
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="EcoVision Enterprise",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State for Theme ---
if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = True  # Dark mode default

# --- Theme-based CSS ---
def get_theme_css():
    if st.session_state['dark_mode']:
        # Dark theme
        return """
        <style>
        .main {
            background-color: #0e1117;
            color: #ffffff;
        }
        
        .stMetric {
            background-color: #1e1e1e;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #00d4aa;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            color: white;
        }
        
        .metric-value {
            color: #00d4aa !important;
        }
        
        .metric-label {
            color: #b0b0b0 !important;
        }
        
        div[data-testid="stSidebar"] {
            background-color: #1a1d23;
            color: white;
        }
        
        div[data-testid="stSidebarUserContent"] {
            padding-top: 2rem;
        }
        
        .stButton button {
            background-color: #00d4aa;
            color: black;
            font-weight: bold;
            border-radius: 8px;
            border: none;
            padding: 10px 20px;
        }
        
        .stButton button:hover {
            background-color: #00b894;
        }
        
        .stRadio > div {
            background-color: #2d2d2d;
            padding: 10px;
            border-radius: 10px;
            color: white;
        }
        
        .stProgress > div > div {
            background-color: #00d4aa;
        }
        
        .stAlert {
            background-color: #2d2d2d;
            color: white;
        }
        
        h1, h2, h3, h4 {
            color: #00d4aa;
        }
        
        .info-card {
            background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
            padding: 20px;
            border-radius: 15px;
            border-left: 4px solid #00d4aa;
            margin-bottom: 20px;
            color: white;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #1a1d23 0%, #252930 100%);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            border: 1px solid #333;
            color: white;
        }
        
        .sidebar .stRadio label, .sidebar .stMarkdown, .sidebar .stMetric label {
            color: #e0e0e0 !important;
        }
        
        /* Footer captions */
        .stCaption {
            color: #b0b0b0 !important;
        }
        </style>
        """
    else:
        # Light theme - fully readable, high contrast
        return """
        <style>
        /* Main background and text */
        .stApp, .main {
            background-color: #f8f9fa;
            color: #212529;
        }
        
        /* Broadly override default light text elements to dark gray */
        .stApp p, .stApp span, .stApp label, .stApp li {
            color: #212529;
        }
        
        /* Force widget labels (like the radio button header) to be dark */
        div[data-testid="stWidgetLabel"] p, 
        div[data-testid="stWidgetLabel"] h1, 
        div[data-testid="stWidgetLabel"] h2, 
        div[data-testid="stWidgetLabel"] h3,
        legend {
            color: #212529 !important;
        }
        
        /* Metric cards */
        div[data-testid="stMetric"] {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #00d4aa;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }
        
        div[data-testid="stMetricValue"] > div {
            color: #00b894 !important;
        }
        
        div[data-testid="stMetricLabel"] > div > p {
            color: #212529 !important;
            font-weight: 500;
        }
        
        /* Sidebar */
        div[data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: 1px solid #dee2e6;
        }
        
        /* Radio button container */
        div[data-testid="stRadio"] > div {
            background-color: #e9ecef;
            padding: 10px;
            border-radius: 10px;
        }
        
        /* Buttons */
        .stButton button {
            background-color: #00d4aa;
            border-radius: 8px;
            border: none;
            padding: 10px 20px;
        }
        
        .stButton button p {
            color: #212529 !important;
            font-weight: bold;
        }
        
        .stButton button:hover {
            background-color: #00b894;
        }
        
        /* Progress bar */
        .stProgress > div > div > div > div {
            background-color: #00d4aa;
        }
        
        /* Alert/info boxes */
        div[data-testid="stAlert"] {
            background-color: #f1f3f5 !important;
        }
        div[data-testid="stAlert"] p {
            color: #212529 !important;
        }
        
        /* Headers */
        h1, h2, h3, h4 {
            color: #00b894 !important;
        }
        
        /* Info cards (custom) */
        .info-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            padding: 20px;
            border-radius: 15px;
            border-left: 4px solid #00d4aa;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .info-card p, .info-card h4 {
            color: #212529 !important;
        }
        
        /* Stat cards */
        .stat-card {
            background: #ffffff;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            border: 1px solid #dee2e6;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        
        .stat-card p {
            color: #212529 !important;
        }
        
        /* Captions (footer, etc.) */
        div[data-testid="stCaptionContainer"] p {
            color: #495057 !important;
        }
        
        /* Expander headers */
        [data-testid="stExpander"] summary p {
            color: #212529 !important;
        }
        
        /* Tabs */
        [data-baseweb="tab-list"] button p {
            color: #212529 !important;
        }
        
        /* File Uploader & Camera explicit overrides */
        [data-testid="stFileUploader"] p,
        [data-testid="stFileUploader"] span,
        [data-testid="stCameraInput"] p,
        [data-testid="stCameraInput"] span {
            color: #212529 !important;
        }
        </style>
        """

# Apply current theme CSS
st.markdown(get_theme_css(), unsafe_allow_html=True)

# --- Global Constants ---
IMG_HEIGHT = 150
IMG_WIDTH = 150
MODEL_PATH = 'model.h5'
CLASS_NAMES = ['Organic', 'Recyclable']

# Waste categories with examples and disposal guidelines
WASTE_INFO = {
    'Organic': {
        'examples': ['Food scraps', 'Vegetable peels', 'Coffee grounds', 'Leaves', 'Paper towels'],
        'disposal': 'Green Bin (Compost)',
        'benefits': 'Reduces landfill methane, creates nutrient-rich compost',
        'color': '#2ecc71'
    },
    'Recyclable': {
        'examples': ['Plastic bottles', 'Aluminum cans', 'Glass jars', 'Cardboard', 'Metal containers'],
        'disposal': 'Blue Bin (Recycling)',
        'benefits': 'Conserves resources, saves energy, reduces pollution',
        'color': '#3498db'
    }
}

# --- Session State for History ---
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'stats' not in st.session_state:
    st.session_state['stats'] = {
        'total_scans': 0,
        'organic_count': 0,
        'recyclable_count': 0,
        'avg_confidence': 0
    }

# --- Model Loading ---
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        st.info("Using mock predictions for demonstration")
        return None

model = load_model()

# --- Feature 1: Live Video Processor Class ---
class WasteDetectionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Preprocess
        img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)

        # 2. Predict (only if model is loaded)
        if model is not None:
            prediction = model.predict(img_batch)
            probability = prediction[0][0]

            if probability > 0.5:
                label = "Recyclable"
                confidence = probability
                color = (52, 152, 219)  # Blue
                bgr_color = (219, 152, 52)
            else:
                label = "Organic"
                confidence = 1 - probability
                color = (46, 204, 113)  # Green
                bgr_color = (113, 204, 46)

            # 3. Overlay Graphics with better visual
            text = f"{label}: {confidence*100:.1f}%"
            
            # Create semi-transparent overlay
            overlay = img.copy()
            cv2.rectangle(overlay, (10, 10), (400, 70), (30, 30, 30), -1)
            img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
            
            # Add text with shadow
            cv2.putText(img, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.2, (0, 0, 0), 4)
            cv2.putText(img, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.2, bgr_color, 2)
            
            # Add colored indicator dot
            cv2.circle(img, (25, 25), 8, bgr_color, -1)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Feature 2: Shared Analysis Logic ---
def analyze_and_display(image_input, source="Unknown"):
    """
    Common logic to process an image, display results, and save to history.
    """
    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col1:
        st.image(image_input, caption="Input Source", use_column_width=True)
    
    with col2:
        with st.spinner("🔍 Analyzing waste composition..."):
            time.sleep(0.5)
            
            # Preprocess
            img = ImageOps.fit(image_input, (IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
            img_array = np.asarray(img)
            normalized_image_array = (img_array.astype(np.float32) / 255.0)
            data = np.expand_dims(normalized_image_array, axis=0)
            
            # Predict
            if model is not None:
                prediction = model.predict(data, verbose=0)
                probability = prediction[0][0]
            else:
                # Mock prediction for demo
                probability = np.random.uniform(0, 1)
            
            if probability > 0.5:
                label = "Recyclable"
                conf = probability
            else:
                label = "Organic"
                conf = 1 - probability

            # Update statistics
            st.session_state['stats']['total_scans'] += 1
            if label == "Organic":
                st.session_state['stats']['organic_count'] += 1
            else:
                st.session_state['stats']['recyclable_count'] += 1
            
            # Calculate average confidence
            total_conf = st.session_state['stats']['avg_confidence'] * (st.session_state['stats']['total_scans'] - 1)
            st.session_state['stats']['avg_confidence'] = (total_conf + conf) / st.session_state['stats']['total_scans']

            # Display Metrics
            st.markdown(f"### 📊 Analysis Results")
            
            # Confidence meter
            st.markdown(f"**Confidence:** {conf*100:.1f}%")
            st.progress(float(conf))
            
            # Type indicator
            col_a, col_b = st.columns(2)
            with col_a:
                color = WASTE_INFO[label]['color']
                st.markdown(f"""
                <div style='background-color:{color}20; padding:15px; border-radius:10px; border-left:4px solid {color}'>
                    <h4 style='color:{color}; margin:0;'>🔍 Detected Type</h4>
                    <h2 style='color:{color}; margin:5px 0;'>{label}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                st.markdown(f"""
                <div class='info-card'>
                    <h4>📝 Disposal Guide</h4>
                    <p><strong>Bin:</strong> {WASTE_INFO[label]['disposal']}</p>
                    <p><strong>Benefits:</strong> {WASTE_INFO[label]['benefits']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Examples
            with st.expander(f"📋 Common {label} Waste Examples"):
                for example in WASTE_INFO[label]['examples']:
                    st.markdown(f"• {example}")
            
            # Update History
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_entry = {
                "Time": timestamp,
                "Type": label,
                "Confidence": f"{conf:.3f}",
                "Source": source
            }
            
            # Only add if it's a new entry
            if not st.session_state['history'] or st.session_state['history'][-1] != new_entry:
                st.session_state['history'].append(new_entry)

# --- Knowledge Section ---
def display_knowledge_section():
    st.markdown("## 🌱 Waste Management Knowledge Hub")
    
    tab1, tab2, tab3 = st.tabs(["📚 Classification Guide", "📈 Environmental Impact", "💡 Best Practices"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ♻️ Recyclable Waste")
            st.markdown("""
            - **Plastics:** Bottles, containers, packaging (check local codes)
            - **Metals:** Aluminum cans, steel containers
            - **Glass:** Jars, bottles (clear, green, brown)
            - **Paper:** Newspapers, magazines, cardboard
            - **Rinse before recycling to avoid contamination**
            """)
        
        with col2:
            st.markdown("### 🍃 Organic Waste")
            st.markdown("""
            - **Food Scraps:** Fruits, vegetables, grains
            - **Yard Waste:** Leaves, grass clippings
            - **Paper Products:** Soiled paper, napkins
            - **Avoid:** Meat, dairy, oils (in home compost)
            - **Benefits:** Reduces landfill methane by 95%
            """)
    
    with tab2:
        st.markdown("### Environmental Impact Statistics")
        
        # Environmental impact data
        impact_data = pd.DataFrame({
            'Metric': ['CO2 Reduction', 'Energy Saved', 'Landfill Space', 'Water Saved'],
            'Value': [85, 65, 75, 40],
            'Unit': ['%', '%', '%', '%']
        })
        
        # Use Plotly template based on current theme
        template = 'plotly_dark' if st.session_state['dark_mode'] else 'plotly_white'
        fig = px.bar(impact_data, x='Metric', y='Value',
                     color='Value', color_continuous_scale='Viridis',
                     title='Environmental Benefits of Proper Waste Sorting',
                     labels={'Value': 'Percentage Improvement (%)'},
                     template=template)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Key Statistics:**
        - Recycling 1 aluminum can saves enough energy to power a TV for 3 hours
        - Composting food waste reduces methane emissions by 21 times compared to landfills
        - Proper waste sorting can reduce overall waste by up to 85%
        """)
    
    with tab3:
        st.markdown("### 💡 Sustainable Practices")
        
        tips = [
            "Rinse containers before recycling",
            "Break down cardboard boxes to save space",
            "Start a home compost bin for organic waste",
            "Avoid plastic bags in recycling bins",
            "Check local recycling guidelines regularly",
            "Use reusable containers instead of disposable ones",
            "Donate usable items instead of throwing them away"
        ]
        
        for i, tip in enumerate(tips, 1):
            st.markdown(f"{i}. {tip}")

# --- Statistics Dashboard ---
def display_statistics():
    st.markdown("## 📊 Performance Dashboard")
    
    # Calculate stats
    stats = st.session_state['stats']
    
    # Create metric columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='stat-card'>
            <h3 style='color:#00d4aa;'>{stats['total_scans']}</h3>
            <p>Total Scans</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        organic_pct = (stats['organic_count'] / max(stats['total_scans'], 1)) * 100
        st.markdown(f"""
        <div class='stat-card'>
            <h3 style='color:#2ecc71;'>{stats['organic_count']}</h3>
            <p>Organic ({organic_pct:.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        recyclable_pct = (stats['recyclable_count'] / max(stats['total_scans'], 1)) * 100
        st.markdown(f"""
        <div class='stat-card'>
            <h3 style='color:#3498db;'>{stats['recyclable_count']}</h3>
            <p>Recyclable ({recyclable_pct:.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='stat-card'>
            <h3 style='color:#f1c40f;'>{stats['avg_confidence']:.1%}</h3>
            <p>Avg Confidence</p>
        </div>
        """, unsafe_allow_html=True)
    
    # History visualization
    if st.session_state['history']:
        st.markdown("### 📈 Historical Analysis")
        
        df = pd.DataFrame(st.session_state['history'])
        df['Time'] = pd.to_datetime(df['Time'])
        df['Confidence'] = df['Confidence'].astype(float)
        
        col_chart1, col_chart2 = st.columns(2)
        
        # Determine plotly template based on current theme
        template = 'plotly_dark' if st.session_state['dark_mode'] else 'plotly_white'
        
        with col_chart1:
            # Pie chart of waste types
            type_counts = df['Type'].value_counts()
            fig_pie = px.pie(values=type_counts.values, 
                            names=type_counts.index,
                            color=type_counts.index,
                            color_discrete_map={'Organic': '#2ecc71', 'Recyclable': '#3498db'},
                            title='Waste Type Distribution',
                            template=template)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_chart2:
            # Confidence over time
            fig_line = px.line(df, x='Time', y='Confidence', color='Type',
                              color_discrete_map={'Organic': '#2ecc71', 'Recyclable': '#3498db'},
                              title='Confidence Trends Over Time',
                              markers=True,
                              template=template)
            fig_line.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig_line, use_container_width=True)

# --- Main Application Interface ---
def main():
    
    # --- Sidebar: Navigation & Stats ---
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; margin-bottom: 30px;'>
            <h1 style='color:#00d4aa;'>♻️</h1>
            <h3>EcoVision Control</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Theme Toggle Button
        theme_label = "🌙 Dark Mode" if st.session_state['dark_mode'] else "☀️ Light Mode"
        if st.button(theme_label, use_container_width=True):
            st.session_state['dark_mode'] = not st.session_state['dark_mode']
            st.rerun()
        
        st.divider()
        
        # Operation Mode
        mode = st.radio("🎯", 
                        ["🔴 Live Stream", 
                         "📷 Camera", 
                         "📁 File Upload",
                         "📊 Dashboard"])
        
        st.divider()
        
        # Quick Stats
        st.markdown("### ⚡ Quick Stats")
        stats = st.session_state['stats']
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Total Scans", stats['total_scans'])
        with col_s2:
            st.metric("Accuracy", f"{stats['avg_confidence']:.1%}")
        
        st.divider()
        
        # Session History
        st.markdown("### 📋 Recent History")
        if st.session_state['history']:
            df_recent = pd.DataFrame(st.session_state['history'][-5:])  # Last 5 entries
            for _, row in df_recent.iterrows():
                color = '#2ecc71' if row['Type'] == 'Organic' else '#3498db'
                st.markdown(f"""
                <div style='background-color:{color}20; padding:10px; border-radius:5px; margin:5px 0; border-left:3px solid {color}'>
                    <strong>{row['Time'].split()[1]}</strong> - {row['Type']} ({float(row['Confidence'])*100:.0f}%)
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("🗑️ Clear All Data", use_container_width=True):
                st.session_state['history'] = []
                st.session_state['stats'] = {
                    'total_scans': 0,
                    'organic_count': 0,
                    'recyclable_count': 0,
                    'avg_confidence': 0
                }
                st.rerun()
        else:
            st.caption("No analysis records yet")

    # --- Main Page Layout ---
    # Dynamic subheading color based on theme
    subheading_color = "#b0b0b0" if st.session_state['dark_mode'] else "#495057"
    st.markdown(f"""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='color:#00d4aa; font-size: 3em;'>♻️ EcoVision Enterprise</h1>
        <h3 style='color:{subheading_color};'>Intelligent Waste Classification System for Sustainable Urban Management</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # --- Display Mode Content ---
    if mode == "🔴 Live Stream":
        st.markdown("## 🎥 Real-time Waste Analysis")
        st.info("🎯 **Live Mode**: Point your camera at waste items for instant classification. Results display in real-time.")
        
        col_video, col_guide = st.columns([2, 1])
        
        with col_video:
            webrtc_streamer(
                key="waste-live", 
                video_processor_factory=WasteDetectionProcessor,
                media_stream_constraints={"video": True, "audio": False},
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )
        
        with col_guide:
            st.markdown("#### 🎯 Real-time Guide")
            st.markdown("""
            <div class='info-card'>
                <h4>🟢 Organic Waste</h4>
                <p>Food scraps, yard waste, paper products</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class='info-card'>
                <h4>🔵 Recyclable Waste</h4>
                <p>Plastics, metals, glass, clean paper</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### ⚡ Tips for Best Results")
            st.markdown("""
            - Ensure good lighting
            - Focus on one item at a time
            - Keep camera steady
            - Avoid reflective surfaces
            """)

    elif mode == "📷 Camera":
        st.markdown("## 📸 Smart Camera Analysis")
        st.info("📷 **Snapshot Mode**: Take photos to analyze waste items. Results are saved to your history.")
        
        camera_file = st.camera_input("📸 Position waste item and click to capture")
        
        if camera_file:
            user_image = Image.open(camera_file).convert("RGB")
            analyze_and_display(user_image, source="Camera")
            
            # Show statistics after analysis
            display_statistics()

    elif mode == "📁 File Upload":
        st.markdown("## 📁 Batch Image Analysis")
        st.info("💾 **Upload Mode**: Analyze existing images from your device. Supports JPG, PNG formats.")
        
        upload_file = st.file_uploader("📤 Drag and drop or click to upload", 
                                      type=['jpg', 'jpeg', 'png'])
        
        if upload_file:
            user_image = Image.open(upload_file).convert("RGB")
            analyze_and_display(user_image, source="Upload")
            
            # Show statistics after analysis
            display_statistics()
        
        # Multiple file upload option
        uploaded_files = st.file_uploader("📦 Batch Upload (Multiple Images)", 
                                         type=['jpg', 'jpeg', 'png'],
                                         accept_multiple_files=True)
        
        if uploaded_files:
            progress_bar = st.progress(0)
            for i, file in enumerate(uploaded_files):
                image = Image.open(file).convert("RGB")
                st.markdown(f"**Analyzing:** {file.name}")
                analyze_and_display(image, source=f"Batch:{file.name}")
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            st.success(f"✅ Completed analysis of {len(uploaded_files)} images!")

    elif mode == "📊 Dashboard":
        display_statistics()
        display_knowledge_section()
        
        # Export data option
        if st.session_state['history']:
            st.divider()
            col_exp1, col_exp2 = st.columns([1, 3])
            with col_exp1:
                if st.button("📥 Export History to CSV"):
                    df_export = pd.DataFrame(st.session_state['history'])
                    csv = df_export.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"ecovision_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            with col_exp2:
                st.caption("Export your analysis history for reporting and compliance purposes.")

    # --- Footer ---
    st.divider()
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.markdown("**🌍 Environmental Impact**")
        st.caption("Every proper classification helps reduce landfill waste")
    with col_f2:
        st.markdown("**🤖 AI-Powered**")
        st.caption("Using deep learning for accurate waste recognition")
    with col_f3:
        st.markdown("**📈 Real-time Analytics**")
        st.caption("Monitor your waste management performance")

if __name__ == "__main__":
    main()