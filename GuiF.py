# ==========================================
# Streamlit GUI for Flexural Capacity Prediction
# Using Best ML Model
# Developed by Dr. Ahmed Aandor
# ==========================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os


# ===== Paths (same folder as GUI) =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "xgboost_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "xgboost_scaler.pkl")
IMAGE_PATH = os.path.join(BASE_DIR, "Image1.png")  # uploaded image

# ===== Load Model and Scaler =====
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ===== Page Setup =====
#st.set_page_config(page_title="Flexural Capacity Prediction for FRCM-Strengthened RC Beams and Slabs", layout="wide")
#st.markdown("<h1 style='font-size: 36px;'>Flexural Capacity Prediction for FRCM-Strengthened RC Beams and Slabs</h1>", unsafe_allow_html=True)
st.set_page_config(
    page_title="Flexural Capacity Prediction for FRCM-Strengthened RC Beams and Slabs", layout="wide"
)

st.markdown("""
    <h1 style='
        font-size: 40px;
        color: #8B0000;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    '>
        Flexural Capacity Prediction for FRCM-Strengthened RC Beams and Slabs
    </h1>
""", unsafe_allow_html=True)

st.markdown("<p style='font-size: 30px;'>This application predicts the <strong>flexural capacity (M)</strong> of reinforced concrete beams and slabs strengthened with FRCM composites using a trained XGBoost ML model.</p>", unsafe_allow_html=True)



# ===== Layout: Sliders + Image =====
col1, col2= st.columns([1, 2])

with col1:
    #st.header("Input Beam/Material Properties")
    st.markdown("<h2 style='font-size: 34px;'>🔧 Input Beam/Material Properties</h2>", unsafe_allow_html=True)


    st.markdown("<p style='font-size:28px; font-weight:bold;margin-bottom:0px;'>Member width (b) [mm]</p>", unsafe_allow_html=True)
    b = st.slider("",
                    min_value=120, max_value=1000, value=300, step=10)

    st.markdown("<p style='font-size:28px; font-weight:bold;margin-bottom:0px;'>Member depth (h) [mm]</p>", unsafe_allow_html=True)
    h = st.slider("",
                    min_value=100, max_value=500, value=500, step=10)
    
    st.markdown("<p style='font-size:28px; font-weight:bold;margin-bottom:0px;'>Bending length (L) [mm]</p>", unsafe_allow_html=True)
    L = st.slider("",
                    min_value=508.0, max_value=1500.0, value=1000.0, step=10.0)
    
    st.markdown("<p style='font-size:28px; font-weight:bold;margin-bottom:0px;'>Concrete compressive strength (fc) [MPa]</p>", unsafe_allow_html=True)
    fc = st.slider("",
                     min_value=15.1, max_value=67.5, value=30.0, step=0.1)
    

    st.markdown("<p style='font-size:28px; font-weight:bold;margin-bottom:0px;'>Steel yield strength (fs) [MPa]</p>", unsafe_allow_html=True)
    fs = st.slider("",
                     min_value=200.0, max_value=700.0, value=400.0, step=1.0)
    
    st.markdown("<p style='font-size:28px; font-weight:bold;margin-bottom:0px;'>Steel reinforcement area (Ast) [mm²]</p>", unsafe_allow_html=True)
    Ast = st.slider("",
                      min_value=70, max_value=810, value=150, step=10)
    
    st.markdown("<p style='font-size:28px; font-weight:bold;margin-bottom:0px;'>Compressive steel reinforcement area (Asc) [mm²]</p>", unsafe_allow_html=True)
    Asc = st.slider("",
                      min_value=0, max_value=610, value=0, step=10)

    st.markdown("<p style='font-size:28px; font-weight:bold;margin-bottom:0px;'>FRCM reinforcement area (Af) [mm²]</p>", unsafe_allow_html=True)
    Af = st.slider("",
                     min_value=10.0, max_value=2000.0, value=100.0, step=5.0)

    st.markdown("<p style='font-size:28px; font-weight:bold;margin-bottom:0px;'>Elastic modulus of fibers (Ef) [GPa]</p>", unsafe_allow_html=True)
    Ef = st.slider("",
                     min_value=10, max_value=270, value=50, step=10)
    

with col2:
    #st.markdown("**👨‍🔬 Developed by Dr. Ahmed Mandor**")
    st.markdown("""
        <div style='text-align: right; margin-bottom: -10px;'>
            <p style='color: #2E8B57; font-size: 35px; font-weight: bold;'>👨‍🔬 Developed by Dr. Ahmed Mandor</p>
        </div>
    """, unsafe_allow_html=True)


    st.image(IMAGE_PATH, caption="FRCM-Strengthened Reinforced Concrete Section", use_container_width=True)


    # ===== Prepare Summary Table =====
    st.markdown("<h3 style='font-size: 30px;'>📋 Input Summary</h3>", unsafe_allow_html=True)

    input_dict = {
        'Parameter': ['b (mm)', 'h (mm)', 'L (mm)', 'fc (MPa)', ' fs (MPa)', 'Ast (mm2)', 'Asc (mm2)', 'Af (mm2)', 'Ef (GPa)'],
        'Value': [b, h, L, fc, fs, Ast, Asc, Af, Ef]
    }
    summary_df = pd.DataFrame(input_dict)
    #st.subheader("Input Summary")
    #st.table(summary_df)
    #st.table(summary_df.set_index('Parameter').T)
    # ===== Prepare Summary Table =====

# ===== Prepare Horizontal Summary Table =====

# Build horizontal HTML table
    table_html = f"""
    <style>
    .horizontal-table {{
        font-size: 20px;
        font-weight: bold;
        color: #333;
        border-collapse: collapse;
        width: 100%;
    }}
    .horizontal-table th, .horizontal-table td {{
        border: 1px solid #ccc;
        padding: 8px 12px;
        text-align: center;
    }}
    .horizontal-table th {{
        background-color: #f2f2f2;
    }}
    </style>

    <table class="horizontal-table">
    <tr>
        <th>b (mm)</th>
        <th>h (mm)</th>
        <th>L (mm)</th>
        <th>fc (MPa)</th>
        <th>fs (MPa)</th>
        <th>Ast (mm²)</th>
        <th>Asc (mm²)</th>
        <th>Af (mm²)</th>
        <th>Ef (GPa)</th>
    </tr>
    <tr>
        <td>{b}</td>
        <td>{h}</td>
        <td>{L}</td>
        <td>{fc}</td>
        <td>{fs}</td>
        <td>{Ast}</td>
        <td>{Asc}</td>
        <td>{Af}</td>
        <td>{Ef}</td>
    </tr>
    </table>
    """

    st.markdown(table_html, unsafe_allow_html=True)



# ===== Prepare Input for Prediction =====
input_df = pd.DataFrame({
    'b (mm)': [b],
    'h (mm)': [h],
    'L (mm)': [L],
    'fc (MPa)': [fc],
    ' fs (MPa)': [fs],
    'Ast (mm2)': [Ast],
    'Asc (mm2)': [Asc],
    'Af (mm2)': [Af],
    'Ef (GPa)': [Ef]
}, dtype=float)

input_scaled = scaler.transform(input_df)

# ===== Prediction =====
#st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
#if st.button("Predict Flexural Capacity", key="predict_btn", help="Click to predict M", 
             #type="primary", use_container_width=True):
    #prediction = model.predict(input_scaled)
    #st.success(f"Predicted Flexural Capacity (M): **{prediction[0]:.2f} kNm**", icon="💡")


    # ===== Styled Button and Prediction Result =====
# ===== Styled Button and Prediction Result =====


#Prediction logic with styled button
#if st.button("Predict Flexural Capacity"):
    #prediction = model.predict(input_scaled)
    #st.markdown(f"""
        #<div style='margin-top:20px; padding:15px; border:2px solid #2E8B57; border-radius:10px; background-color:#f0fdf4;'>
            #<p style='font-size:35px; color:#2E8B57; font-weight:bold;'>🧮 Predict Flexural Capacity = {prediction[0]:.2f} kNm</p>
        #</div>
    #""", unsafe_allow_html=True)


col_button, col_result = st.columns([1, 2])  # Adjust ratio as needed

with col_button:
    st.markdown("""
    <style>
    div[data-testid="stButton"] > button {
        background-color: #2E8B57;
        color: white;
        font-size: 35px !important;
        font-weight: bold;
        padding: 16px 40px;
        border-radius: 6px;
        border: none;
    }
    div[data-testid="stButton"] > button:hover {
        background-color: #246b45;
        color: white;
    }
    </style>
     """, unsafe_allow_html=True)


    predict = st.button("Predict Flexural Capacity")
    

with col_result:
    if predict:
        prediction = model.predict(input_scaled)

        st.markdown(f"""
            <div style='padding:15px; border:2px solid #2E8B57; border-radius:10px; background-color:#f0fdf4;'>
                <p style='font-size:30px; color:#2E8B57; font-weight:bold; margin:0;'>🧮 Predicted Flexural Capacity = {prediction[0]:.2f} kNm</p>
            </div>
        """, unsafe_allow_html=True)




# ===== Author =====
#st.markdown("---")
#st.caption("© Developed by Dr. Ahmed Mandor")
#st.markdown("<p style='font-size: 14px;'><strong>© Developed by Dr. Ahmed Mandor</strong></p>", unsafe_allow_html=True)

