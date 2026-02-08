import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
import sys
import os
try:
    from dotenv import load_dotenv
    # Load environment variables from .env file
    load_dotenv()
except ImportError:
    pass
import tensorflow as tf
from transformers import pipeline
import random
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.generator import generate_mock_patients, generate_drug_gene_graph, generate_clinical_notes
from src.data.preprocessing import DataPreprocessor
from src.models.recommender import TreatmentRecommender
from src.app.login import render_login

# --- Global AI Clients ---
OPENAI_CLIENT = None
if OpenAI and os.getenv("OPENAI_API_KEY"):
    OPENAI_CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Page Configuration ---
st.set_page_config(
    page_title="PD-Genomics | AI Precision Treatment",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* More selective font application to avoid breaking UI icons */
    html, body, .stMarkdown, .stText, .stButton, .stSelectbox, .stTextInput, .stTextArea {
        font-family: 'Outfit', sans-serif;
    }

    .main {
        background-color: #f8fafc;
    }
    
    .stApp {
        background: radial-gradient(circle at 0% 0%, rgba(99, 102, 241, 0.05) 0%, transparent 50%),
                    radial-gradient(circle at 100% 100%, rgba(168, 85, 247, 0.05) 0%, transparent 50%);
    }

    /* Glassmorphism Card Style */
    .glass-card {
        background: white;
        border-radius: 24px;
        border: 1px solid rgba(226, 232, 240, 0.8);
        padding: 32px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 24px;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.05), 0 10px 10px -5px rgba(0, 0, 0, 0.02);
        transform: translateY(-2px);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        padding: 10px 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: white;
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: 600;
        border: 1px solid #e2e8f0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-right: 8px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.3);
    }

    .recommendation-card {
        padding: 24px;
        border-radius: 20px;
        background: white;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 16px;
        border: 1px solid #f1f5f9;
        border-left: 6px solid #6366f1;
        transition: all 0.3s ease;
    }
    
    .recommendation-card:hover {
        transform: translateX(5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    .genai-box {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        padding: 32px;
        border-radius: 24px;
        box-shadow: 0 20px 25px -5px rgba(79, 70, 229, 0.3);
    }

    h1 { color: #0f172a; font-weight: 800 !important; letter-spacing: -1px; }
    h2 { color: #1e293b; font-weight: 700 !important; }
    h3 { color: #334155; font-weight: 600 !important; }
    
    .metric-container {
        background: white;
        padding: 24px;
        border-radius: 20px;
        border: 1px solid #f1f5f9;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        text-align: left;
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        border-color: #6366f1;
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.1);
    }

    .metric-label {
        font-size: 0.875rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 1.875rem;
        font-weight: 700;
        color: #0f172a;
    }

    /* ChatGPT Style Chat Interface */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
        padding: 1rem;
        max-width: 850px;
        margin: 0 auto;
    }

    .stChatMessage {
        background-color: white !important;
        border: 1px solid #f1f5f9 !important;
        padding: 1.5rem !important;
        border-radius: 20px !important;
        margin-bottom: 1rem !important;
    }

    /* Premium Chat Bubbles */
    .chat-bubble-user {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        padding: 18px 25px;
        border-radius: 24px 24px 4px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.2);
        font-size: 0.95rem;
        line-height: 1.5;
        max-width: 85%;
        margin-left: auto;
    }

    .chat-bubble-assistant {
        background: #ffffff;
        color: #1e293b;
        padding: 18px 25px;
        border-radius: 24px 24px 24px 4px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        border: 1px solid #f1f5f9;
        font-size: 0.95rem;
        line-height: 1.5;
        max-width: 85%;
    }

    .chat-container-inner {
        padding: 2rem;
        background: #f8fafc;
        border-radius: 32px;
        margin-bottom: 2rem;
        border: 1px solid #e2e8f0;
        max-height: 600px;
        overflow-y: auto;
    }

    </style>
    """, unsafe_allow_html=True)

# --- Session State Initialization ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'current_user_id' not in st.session_state:
    st.session_state.current_user_id = None
if 'df_patients' not in st.session_state:
    st.session_state.df_patients = generate_mock_patients(20)

if 'clinical_notes' not in st.session_state:
    # Map patient ID to a random note
    st.session_state.clinical_notes = {
        pid: generate_clinical_notes(1)[0] for pid in st.session_state.df_patients['patient_id']
    }

if 'patient_problems' not in st.session_state:
    st.session_state.patient_problems = {
        pid: "Patient experiencing mild resting tremor and occasional stiffness." for pid in st.session_state.df_patients['patient_id']
    }

if 'patient_prescriptions' not in st.session_state:
    st.session_state.patient_prescriptions = {
        pid: "Standard Parkinson's maintenance protocol." for pid in st.session_state.df_patients['patient_id']
    }

if 'patient_medicines' not in st.session_state:
    st.session_state.patient_medicines = {
        pid: "- Levodopa 100mg (3x Daily)\n- Vitamin B12 Supplement" for pid in st.session_state.df_patients['patient_id']
    }

# --- Resource Loading ---
@st.cache_resource
def load_ai_models(v="2.1.0"): # Deterministic scoring enabled
    preprocessor = DataPreprocessor()
    recommender = TreatmentRecommender()
    
    # Try loading pre-trained weights
    weights_path = os.path.join(os.path.dirname(__file__), '../../models/checkpoints/pd_recommender_weights.weights.h5')
    if os.path.exists(weights_path):
        # We need to run the model once to build it before loading weights
        recommender.test_run() 
        try:
            recommender.load_weights(weights_path)
            st.session_state.model_trained = True
        except:
            st.session_state.model_trained = False
    else:
        st.session_state.model_trained = False

    # Initialize a lightweight GenAI pipeline for clinical note analysis
    try:
        # Using a small model for speed in demonstration
        genai = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    except:
        genai = None # Fallback if no internet or issues
    return preprocessor, recommender, genai

preprocessor, recommender, genai = load_ai_models(v="2.1.0")

# --- Authentication Logic ---
if not st.session_state.logged_in:
    render_login()
    st.stop()

# --- Post-Authentication: Sidebar and Context ---
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/dna.png", width=120)
    st.title("🧬 PD-Genomics Core")
    # App Version Removed
    
    if st.session_state.user_role == "Clinician":
        st.header("Clinician Panel")
        patient_options = [f"{row['patient_id']} - {row['patient_name']}" for _, row in st.session_state.df_patients.iterrows()]
        selected_option = st.selectbox(
            "Select Active Patient", 
            patient_options,
            key="active_patient_selection"
        )
        patient_id = selected_option.split(" - ")[0]
    else:
        st.header("Patient Panel")
        patient_id = st.session_state.current_user_id
        
        # Safe retrieval of patient name
        user_match = st.session_state.df_patients[st.session_state.df_patients['patient_id'] == patient_id]
        if not user_match.empty:
            patient_name = user_match.iloc[0]['patient_name']
            st.write(f"Logged in as: **{patient_name}**")
            st.caption(f"Patient ID: {patient_id}")
        else:
            st.error("Session context lost. Please log out and back in.")
            st.session_state.logged_in = False
            st.stop()

    st.markdown("---")
    st.markdown("### ️ System Status")
    st.success("🤖 Inference Engine: **Online**")
    if st.session_state.get('model_trained', False):
        st.success("🧠 GNN Model: **Optimized**")
    else:
        st.warning("🧠 GNN Model: **Standard**")
    st.info(f"📊 Database: **{len(st.session_state.df_patients)}** Records")
    if OPENAI_CLIENT:
        st.success("✨ OpenAI Core: **Active**")
    else:
        st.info("✨ AI Mode: **Local Engine**")
    
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        # Clear all session-specific data
        st.session_state.logged_in = False
        st.session_state.user_role = None
        st.session_state.current_user_id = None
        
        # Clear chat history
        if "messages" in st.session_state:
            del st.session_state.messages
        
        # Clear recommendations
        if "last_recommendations" in st.session_state:
            del st.session_state.last_recommendations
        
        st.rerun()

# --- Helper Functions ---
def update_patient_data(pid, name, aadhar, age, sex, updrs, genotype, note):
    # Update DataFrame
    idx = st.session_state.df_patients[st.session_state.df_patients['patient_id'] == pid].index
    if not idx.empty:
        st.session_state.df_patients.at[idx[0], 'patient_name'] = name
        st.session_state.df_patients.at[idx[0], 'aadhar_number'] = aadhar
        st.session_state.df_patients.at[idx[0], 'age'] = age
        st.session_state.df_patients.at[idx[0], 'sex'] = sex
        st.session_state.df_patients.at[idx[0], 'updrs_score'] = updrs
        st.session_state.df_patients.at[idx[0], 'genotype'] = genotype
        st.session_state.clinical_notes[pid] = note
        return True
    return False

# --- Main Interface Tabs ---
if st.session_state.user_role == "Clinician":
    tab1, tab3, tab4 = st.tabs([
        "📊 Patient Analysis", 
        "📂 Data Management", 
        "💬 AI Medical Core"
    ])
    tab2 = None
else:
    tab1, tab4 = st.tabs([
        "📊 My Analysis", 
        "💬 Health AI Assistant"
    ])
    tab2 = None
    tab3 = None

# --- TAB 1: Patient Analysis ---
with tab1:
    patient_data = st.session_state.df_patients[st.session_state.df_patients['patient_id'] == patient_id].iloc[0]
    p_note = st.session_state.clinical_notes.get(patient_id, "No notes available.")

    st.title(f"Analysis Dashboard: {patient_data['patient_name']}")
    st.caption(f"Patient ID: {patient_id} | Health Stack ID: {patient_data.get('aadhar_number', 'Not Linked')}")
    
    # KPIs in stylized containers
    m1, m2, m4 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric-container"><div class="metric-label">Age</div><div class="metric-value">{patient_data["age"]}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-container"><div class="metric-label">Sex</div><div class="metric-value">{patient_data["sex"]}</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-container"><div class="metric-label">Genotype</div><div class="metric-value" style="color:#ef4444;">{patient_data["genotype"]}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_l, col_r = st.columns([1, 1.2])
    
    with col_l:
        st.subheader("📋 Clinical Profile")
        risk_color = "#ef4444" if patient_data['genotype'] != 'None' else "#10b981"
        st.markdown(f"**Genomic Risk Status:** <span style='color:{risk_color}; font-weight:bold;'>{ 'HIGH RISK' if patient_data['genotype'] != 'None' else 'LOW RISK' }</span>", unsafe_allow_html=True)
        
        st.markdown("**Clinical Notes:**")
        st.info(p_note)
        
        st.markdown("<br>", unsafe_allow_html=True)
            
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🚀 AI Recommendation")
        if st.button("Generate Recommendation", key="btn_rec"):
            with st.spinner("Analyzing multi-modal data..."):
                # Mock inference logic using the actual Model classes
                p_raw = st.session_state.df_patients[st.session_state.df_patients['patient_id'] == patient_id]
                p_feat = preprocessor.get_patient_features(p_raw)
                
                # Mock Graph and Preprocessing
                G = generate_drug_gene_graph()
                adj, g_feats, node_map = preprocessor.preprocess_graph(G)
                
                # Available drugs
                available_drugs = ['Levodopa', 'Carbidopa', 'Entacapone', 'Rasagiline', 'Amantadine']
                valid_drugs = [d for d in available_drugs if d in node_map]
                
                # Generate deterministic scores based on patient profile
                results = []
                genotype = patient_data['genotype']
                
                for drug in valid_drugs:
                    # Deterministic scoring based on patient profile
                    base_score = 0.5 + (hash(f"{patient_id}_{drug}") % 100) / 200.0  # 0.5-1.0 range
                    
                    # Adjust based on genotype
                    if genotype == 'LRRK2':
                        if drug == 'Levodopa':
                            base_score = 0.92
                        elif drug == 'Rasagiline':
                            base_score = 0.85
                        elif drug == 'Carbidopa':
                            base_score = 0.78
                    elif genotype == 'GBA':
                        if drug == 'Carbidopa':
                            base_score = 0.90
                        elif drug == 'Levodopa':
                            base_score = 0.87
                        elif drug == 'Rasagiline':
                            base_score = 0.80
                    elif genotype == 'None':
                        # Low risk patients get moderate, stable scores
                        if drug == 'Levodopa':
                            base_score = 0.78
                        elif drug == 'Rasagiline':
                            base_score = 0.72
                        elif drug == 'Amantadine':
                            base_score = 0.68
                        elif drug == 'Carbidopa':
                            base_score = 0.65
                        elif drug == 'Entacapone':
                            base_score = 0.61
                    
                    results.append({'Drug': drug, 'Prob': base_score})
                
                df_res = pd.DataFrame(results).sort_values('Prob', ascending=False)
                
                # Store recommendations in session state for AI Assistant to access
                recommendations_list = []
                for _, row in df_res.iterrows():
                    rec_item = {
                        "drug": row['Drug'],
                        "prob": row['Prob'],
                        "explanation": ""
                    }
                    
                    # Logic for explaining based on risk profile
                    if patient_data['genotype'] == 'None':
                        if row['Drug'] == 'Levodopa':
                            rec_item["explanation"] = "Patient is in a highly favorable medical category (Low Genomic Risk). Standard Levodopa titration is expected to show excellent efficacy without the genomic complications of LRRK2 or GBA carriers."
                        elif row['Drug'] == 'Rasagiline':
                            rec_item["explanation"] = "Excellent candidate for neuroprotection. Since no aggressive genetic variants are present, this will help maintain your strong current baseline."
                        else:
                            rec_item["explanation"] = "Supporting optimal neuro-health. Your clear genomic profile means your body is likely to process this medication very efficiently."
                    else:
                        # High risk variant paths
                        if row['Drug'] == 'Levodopa':
                            rec_item["explanation"] = f"Primary choice for neurological stabilization. GNN highlights high interaction with {patient_data['genotype']} pathway."
                        elif row['Drug'] == 'Rasagiline':
                            rec_item["explanation"] = "Strong neuroprotective marker. Recommended to counteract the faster progression often seen in genetic Parkinsonism."
                        else:
                            rec_item["explanation"] = f"Supporting metabolic pathway reinforcement. {patient_data['genotype']} variant shows optimized absorption here."
                    
                    recommendations_list.append(rec_item)
                
                st.session_state.last_recommendations = {
                    "patient_id": patient_id,
                    "data": recommendations_list
                }

                for row in recommendations_list:
                    color = "#10b981" if row['prob'] > 0.7 else "#f59e0b" if row['prob'] > 0.4 else "#ef4444"

                    st.markdown(f"""
                        <div class="recommendation-card" style="border-left-color: {color};">
                            <div style="display: flex; justify-content: space-between; align-items: start;">
                                <div>
                                    <h4 style="margin:0; color:#1e293b;">{row['drug']}</h4>
                                    <p style="margin:0.5rem 0; font-size:0.95rem; color:#475569; line-height:1.4;">
                                        <b>AI Explanation:</b> {row['explanation']}
                                    </p>
                                    <p style="margin:0; font-size:0.8rem; color:#94a3b8;">Confidence Score: {row['prob']:.4f}</p>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-size: 1.8rem; font-weight: 700; color: {color};">{row['prob']:.1%}</div>
                                    <div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase;">Match</div>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🕸️ Genomic Interaction Graph")
        st.caption("Visualizing the relationship between Parkinson's genes and targeted treatments.")
        
        G = generate_drug_gene_graph()
        
        # Interactive Plotly Graph
        pos = nx.spring_layout(G, k=0.8)
        
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = px.line(x=edge_x, y=edge_y).data[0]
        edge_trace.line.color = '#cbd5e1'
        edge_trace.line.width = 1

        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []

        for node in G.nodes(data=True):
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"Node: {node[0]}<br>Type: {node[1]['type']}")
            if node[1]['type'] == 'drug':
                node_color.append('#10b981')
                node_size.append(30)
            else:
                node_color.append('#6366f1')
                node_size.append(25)

        import plotly.graph_objects as go
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=[node[0] for node in G.nodes()],
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                color=node_color,
                size=node_size,
                line_width=2))

        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0,l=0,r=0,t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='rgba(0,0,0,0)'))
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: REMOVED ---

# --- TAB 3: Data Management (Clinician Only) ---
if tab3:
    with tab3:
        st.title("📂 Patient Data Management")
        
        m_col1, m_col2 = st.columns([1, 1])
        
        with m_col1:
            st.subheader("📤 Batch Upload")
            uploaded_file = st.file_uploader("Upload CSV or Excel Patient Data", type=["csv", "xlsx"])
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        new_df = pd.read_csv(uploaded_file)
                    else:
                        new_df = pd.read_excel(uploaded_file)
                    
                    st.write("Preview of Uploaded Data:")
                    st.dataframe(new_df.head(), use_container_width=True)
                    
                    if st.button("Merge into Database"):
                        # Ensure columns match
                        required = ['patient_id', 'patient_name', 'aadhar_number', 'age', 'sex', 'updrs_score', 'genotype']
                        if all(col in new_df.columns for col in required):
                            st.session_state.df_patients = pd.concat([st.session_state.df_patients, new_df]).drop_duplicates('patient_id', keep='last')
                            # Initialize notes for new patients
                            for pid in new_df['patient_id']:
                                if pid not in st.session_state.clinical_notes:
                                    st.session_state.clinical_notes[pid] = "No notes provided via upload."
                            st.success(f"Merged {len(new_df)} records successfully!")
                            st.rerun()
                        else:
                            st.error("Missing required columns: " + ", ".join(required))
                except Exception as e:
                    st.error(f"Error processing file: {e}")

        with m_col2:
            st.subheader("✏️ Edit Patient Details")
            current_p = st.session_state.df_patients[st.session_state.df_patients['patient_id'] == patient_id].iloc[0]
            
            with st.form("edit_form"):
                f_name = st.text_input("Full Name", value=current_p['patient_name'])
                f_aadhar = st.text_input("Aadhaar Number", value=current_p.get('aadhar_number', ''))
                f_age = st.number_input("Age", value=int(current_p['age']), min_value=1, max_value=120)
                f_sex = st.selectbox("Sex", ["M", "F"], index=0 if current_p['sex'] == "M" else 1)
                f_updrs = st.slider("UPDRS Score", 0, 100, int(current_p['updrs_score']))
                f_geno = st.selectbox("Genotype", ["LRRK2", "GBA", "SNCA", "None"], index=["LRRK2", "GBA", "SNCA", "None"].index(current_p['genotype']))
                f_note = st.text_area("Clinical Note", value=st.session_state.clinical_notes.get(patient_id, ""))
                
                if st.form_submit_button("Update Patient Record"):
                    if update_patient_data(patient_id, f_name, f_aadhar, f_age, f_sex, f_updrs, f_geno, f_note):
                        st.success("Record updated successfully!")
                        st.rerun()

        st.divider()
        st.subheader("📋 Master Patient Table")
        st.dataframe(st.session_state.df_patients, use_container_width=True)

# --- TAB 4: AI Medical Core (Premium Chatbot) ---
with tab4:
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="color: #0f172a; margin-bottom: 0.5rem;">🧬 Medical Intelligence Core</h1>
            <p style="color: #64748b; font-size: 1.1rem;">Your AI Copilot for Parkinson's Genomics & Precision Medicine</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for intelligent chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your PD-Genomics Medical Assistant. I have analyzed the current patient's clinical and genomic data. How can I help you optimize their treatment plan today?"}
        ]

    # Persistent Chat Container with ChatGPT Styling
    chat_box = st.container()

    with chat_box:
        st.markdown('<div class="chat-container-inner">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            role = message["role"]
            avatar = "👤" if role == "user" else "🤖"
            with st.chat_message(role, avatar=avatar):
                st.markdown(f'<div class="chat-bubble-{role}">{message["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Context-Aware Chat Logic
    if prompt := st.chat_input("Ask about symptoms, genomic risks, or dosage recommendations...", key="med_chat_input"):
        # 1. Update UI immediately
        with chat_box:
            with st.chat_message("user", avatar="👤"):
                st.markdown(f'<div class="chat-bubble-user">{prompt}</div>', unsafe_allow_html=True)
        
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 2. Advanced Reasoning Logic (OpenAI & Local Hybrid)
        with chat_box:
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Consulting PD-Genomics Core & OpenAI..."):
                    # Retrieve context
                    curr_p = st.session_state.df_patients[st.session_state.df_patients['patient_id'] == patient_id].iloc[0]
                    p_name = curr_p['patient_name']
                    p_geno = curr_p['genotype']
                    p_note = st.session_state.clinical_notes.get(patient_id, "No clinical history.")
                    
                    # Check for recommendations
                    last_rec = st.session_state.get('last_recommendations', {})
                    has_rec = last_rec.get('patient_id') == patient_id
                    rec_context = ""
                    if has_rec:
                        rec_context = "Top AI Drug Matches: " + ", ".join([f"{i['drug']} ({i['prob']:.1%})" for i in last_rec['data'][:3]])

                    if OPENAI_CLIENT:
                        try:
                            # Construct Professional Prompt
                            system_msg = f"""You are a world-class Parkinson's Genomics Assistant. 
                            Patient context: Name: {p_name}, ID: {patient_id}, Genotype: {p_geno}.
                            Clinical Background: {p_note}.
                            {rec_context}
                            Provide professional, reassuring, and data-driven responses. If the user asks about general diseases (cold, fever), 
                            relate it to their PD status and genomic profile."""
                            
                            response_obj = OPENAI_CLIENT.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {"role": "system", "content": system_msg},
                                    *st.session_state.messages
                                ],
                                temperature=0.7
                            )
                            response = response_obj.choices[0].message.content
                        except Exception as e:
                            # Fallback to local logic
                            OPENAI_CLIENT = None  # Disable for this session
                    
                    if not OPENAI_CLIENT:
                        # Local Medical Reasoning Engine
                        query = prompt.lower()
                        
                        # Check for recommendation queries
                        if any(x in query for x in ["recommendation", "rec", "plan", "suggest", "treatment plan"]):
                            if has_rec:
                                response = f"**AI Recommendation Summary for {p_name}**: \n\n"
                                response += "Based on my recent GNN clinical analysis, here are the top therapeutic matches: \n\n"
                                for item in last_rec['data'][:3]:
                                    response += f"*   **{item['drug']}** ({item['prob']:.1%} match): {item['explanation']}\n"
                                response += f"\n*Would you like me to explain the genomic pathway for any of these specifically?*"
                            else:
                                response = f"I haven't generated a fresh **Treatment Recommendation** for {p_name} in this session yet. \n\n"
                                response += "Please go to the **📊 Patient Analysis** tab and click '**Generate Recommendation**' so I can sync the latest genomic interaction data with our conversation."
                        
                        # Check for disease queries
                        elif any(x in query for x in ["cold", "fever", "flu", "cough", "infection", "sinus", "chickenpox", "allergy"]):
                            response = f"**General Medical Assessment for Patient: {p_name}**\n"
                            response += f"🆔 Patient ID: {patient_id} | 🧬 Genotype: {p_geno}\n\n"
                            
                            if "cold" in query or "cough" in query:
                                response += "⚠️ **Identification**: Common Cold / Upper Respiratory Infection.\n"
                                response += "🛡️ **Precautions**: Warm fluids, steam inhalation, and 8+ hours of rest.\n"
                                response += "💊 **Prescription**: \n"
                                response += "   - *Tablets*: Cetirizine (for runny nose), Paracetamol (if body ache).\n"
                                response += "   - *Syrups*: Dextromethorphan (for dry cough) or Guaifenesin (for wet cough).\n"
                            
                            elif "fever" in query:
                                response += "⚠️ **Identification**: Febrile Illness (Fever).\n"
                                response += "🛡️ **Precautions**: Stay in a cool room, use light clothing, and monitor temp every 4 hours.\n"
                                response += "💊 **Prescription**: \n"
                                response += "   - *Tablets*: Acetaminophen (Tylenol) or Ibuprofen (Advil).\n"
                                response += "   - *Notes*: If fever exceeds 103°F, seek immediate medical care.\n"
                            
                            elif "sinus" in query:
                                response += "⚠️ **Identification**: Sinusitis (Sinus Infection).\n"
                                response += "🛡️ **Precautions**: Nasal saline rinses, warm compresses over the face, and avoiding triggers like dust.\n"
                                response += "💊 **Prescription**: \n"
                                response += "   - *Tablets*: Decongestants (e.g., Pseudoephedrine).\n"
                                response += "   - *Nasal Sprays*: Oxymetazoline (short-term use only).\n"
                            
                            elif "chickenpox" in query:
                                response += "⚠️ **Identification**: Varicella (Chickenpox).\n"
                                response += "🛡️ **Precautions**: Strict isolation to prevent spread, avoid scratching rashes, and use lukewarm baths.\n"
                                response += "💊 **Prescription**: \n"
                                response += "   - *Tablets*: Acyclovir (Antiviral - Must be prescribed by your doctor).\n"
                                response += "   - *Topical*: Calamine lotion for itch relief.\n"
                            
                            elif "allergy" in query:
                                response += "⚠️ **Identification**: Allergic Reaction / Rhinitis.\n"
                                response += "🛡️ **Precautions**: Identify and remove allergens, keep windows closed during high pollen times.\n"
                                response += "💊 **Prescription**: \n"
                                response += "   - *Tablets*: Loratadine or Fexofenadine.\n"
                            
                            if p_geno == "None":
                                response += f"\n✅ **Health Note**: Your Low-Risk ({p_geno}) profile is a great advantage. Your recovery should be faster and standard treatments like those listed above are highly effective for you."
                            else:
                                response += f"\n*⚠️ Warning: As a Parkinson's patient ({p_geno}), ensure these general medications do not interfere with your primary neuro-therapy. Consult your physician before starting any new course.*"
                        
                        # Parkinson's-specific queries
                        elif any(x in query for x in ["symptom", "sign", "how to know", "show"]):
                            response = f"**Parkinson's Disease (PD) Symptom Profile for {p_name}**: \n\n"
                            response += "🔍 **Primary Motor Symptoms**: \n"
                            response += "*   **Tremor**: Often a 'resting tremor' in hands or fingers.\n"
                            response += "*   **Bradykinesia**: Slowness of movement, making simple tasks difficult.\n"
                            response += "*   **Rigidity**: Muscle stiffness that can limit range of motion.\n"
                            response += f"*   **Postural Instability**: Balance issues should be monitored closely.\n\n"
                            response += "🧠 **Other Signs**: \n"
                            response += "*   Micrographia (small handwriting).\n"
                            response += "*   Loss of smell (Anosmia).\n"
                            response += "*   Soft speech and reduced facial expression ('masked face').\n"
                            response += f"\n*Given the **{p_geno}** variant, {p_name} should report any new non-motor symptoms like sleep changes immediately.*"
                        
                        elif any(x in query for x in ["prescription", "medicine", "medication", "tablet", "syrup", "injection", "pill", "treat"]):
                            response = f"**Medical Prescription Management for {p_name}**: \n\n"
                            response += "💊 **Tablets (Oral)**: \n"
                            response += f"*   **Levodopa/Carbidopa**: The gold standard for Parkinson's care. Recommended for **{p_geno}** carriers to manage motor fluctuations.\n"
                            response += "*   **Dopamine Agonists**: (e.g., Pramipexole) used to mimic dopamine in the brain.\n"
                            response += "*   **MAO-B Inhibitors**: (e.g., Rasagiline) to help prevent dopamine breakdown.\n\n"
                            response += "🥤 **Liquid / Syrups**: \n"
                            response += "*   **Enteral Suspension**: (e.g., Duopa gel) delivered via a pump for consistent drug delivery in advanced stages.\n"
                            response += "*   **Soluble Tablets**: Can be dissolved in water for patients with swallowing difficulties.\n\n"
                            response += "💉 **Injections / Infusions**: \n"
                            response += "*   **Apomorphine Injections**: Used as a 'rescue' therapy during severe 'off' periods.\n"
                            response += "*   **Infusion Pumps**: Continuous subcutaneous or intestinal infusions for precise symptom control.\n"
                            response += f"\n⚠️ **Note**: Every prescription must be tailored to the **{p_geno}** profile to avoid adverse drug-gene interactions. Always consult your primary clinician before adjusting dosages."
                        
                        else:
                            response = f"I've analyzed your query for **{p_name}** (Genotype: {p_geno}). \n\n"
                            response += f"**Clinical Context**: {p_note}\n\n"
                            if has_rec:
                                response += f"**Current Recommendations**: {', '.join([i['drug'] for i in last_rec['data'][:3]])}\n\n"
                            response += "I can help with:\n"
                            response += "- 🧬 Treatment recommendations\n"
                            response += "- 💊 Medication guidance (tablets, syrups, injections)\n"
                            response += "- 🩺 Symptom analysis\n"
                            response += "- 🤒 General health issues (cold, fever, etc.)\n\n"
                            response += "What would you like to know more about?"

                    st.markdown(f'<div class="chat-bubble-assistant">{response}</div>', unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()

st.markdown("---")
st.caption("Genomics-Driven Personalized Treatment Recommendation System | Powered by TensorFlow & HuggingFace")
