import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Data Analyst Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .skill-badge {
        display: inline-block;
        background: #f0f2f6;
        color: #262730;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .project-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .project-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 style="margin: 0; font-size: 3rem;">📊 Data Analyst Portfolio</h1>
    <p style="font-size: 1.2rem; margin-top: 1rem; opacity: 0.9;">
        Transforming Data into Actionable Insights
    </p>
</div>
""", unsafe_allow_html=True)

# Introduction
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 👋 Selamat Datang!")
    st.markdown("""
    Saya adalah seorang **Data Analyst** yang berfokus pada analisis data, visualisasi, 
    dan pengembangan insights berbasis data untuk mendukung pengambilan keputusan bisnis.
    
    Portfolio ini menampilkan beberapa proyek analisis data yang telah saya kerjakan, 
    menggunakan berbagai teknik analisis statistik, machine learning, dan visualisasi data interaktif.
    """)
    
    st.markdown("### 🎯 Keahlian Utama")
    st.markdown("""
    <div>
        <span class="skill-badge">Python</span>
        <span class="skill-badge">Pandas</span>
        <span class="skill-badge">NumPy</span>
        <span class="skill-badge">Scikit-learn</span>
        <span class="skill-badge">Altair</span>
        <span class="skill-badge">Streamlit</span>
        <span class="skill-badge">SQL</span>
        <span class="skill-badge">Statistical Analysis</span>
        <span class="skill-badge">Machine Learning</span>
        <span class="skill-badge">Data Visualization</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-container">
        <p class="metric-value">2</p>
        <p class="metric-label">Featured Projects</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="metric-container">
        <p class="metric-value">5+</p>
        <p class="metric-label">Technologies Used</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Projects Overview
st.markdown("## 🚀 Featured Projects")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="project-card">
        <h3>📈 Stock Price Analysis</h3>
        <p>Analisis mendalam terhadap data harga saham dengan visualisasi interaktif, 
        termasuk candlestick charts, moving averages, dan analisis volatilitas.</p>
        <p><strong>Teknologi:</strong> Python, Pandas, Altair, Statistical Analysis</p>
        <p><strong>Dataset:</strong> 1,500+ data points</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="project-card">
        <h3>🔍 Credit Card Fraud Detection</h3>
        <p>Analisis pola fraud pada transaksi kartu kredit menggunakan teknik machine learning 
        dan visualisasi untuk mengidentifikasi anomali.</p>
        <p><strong>Teknologi:</strong> Python, Scikit-learn, PCA, Altair</p>
        <p><strong>Dataset:</strong> 284,000+ transactions</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# About Section
st.markdown("## 📫 Kontak & Informasi")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🔗 GitHub**  
    [github.com/yandri918/data_analyst](https://github.com/yandri918/data_analyst)
    """)

with col2:
    st.markdown("""
    **📧 Email**  
    Contact via GitHub
    """)

with col3:
    st.markdown("""
    **💼 LinkedIn**  
    Connect on LinkedIn
    """)

st.markdown("---")

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <p>Built with ❤️ using Streamlit & Altair</p>
    <p style="font-size: 0.9rem;">© 2026 Data Analyst Portfolio. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Navigation")
    st.info("👈 Pilih halaman di atas untuk melihat proyek analisis data")
    
    st.markdown("---")
    
    st.markdown("### 📊 Quick Stats")
    st.metric("Total Projects", "2")
    st.metric("Total Data Points", "285K+")
    st.metric("Visualization Types", "15+")
    
    st.markdown("---")
    
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
    - **Languages:** Python
    - **Libraries:** Pandas, NumPy, Scikit-learn
    - **Visualization:** Altair, Streamlit
    - **Analysis:** Statistical Methods, ML
    """)
