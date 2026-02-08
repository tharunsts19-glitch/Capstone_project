import streamlit as st
import pandas as pd
import os

def render_login():
    """
    Renders a standard Streamlit login page for PD-Genomics without custom CSS.
    """
    st.title("🧬 PD-Genomics AI")
    st.subheader("Secure access to precision healthcare")
    st.divider()

    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        
        login_type = st.radio(
            "Access Level", 
            ["Patient", "Clinician"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        with st.form("login_form", clear_on_submit=False):
            if login_type == "Patient":
                identifier = st.text_input("Patient ID / Name", placeholder="e.g. PAT001")
                email = st.text_input("Email Access", placeholder="your.name@gmail.com")
                submit = st.form_submit_button("Sign In as Patient")
                
                if submit:
                    if identifier and email:
                        # Logic to check patient credentials
                        df_patients = st.session_state.get('df_patients', pd.DataFrame())
                        if not df_patients.empty:
                            match = df_patients[
                                ((df_patients['patient_id'].str.lower() == identifier.lower()) | 
                                 (df_patients['patient_name'].str.lower() == identifier.lower())) & 
                                (df_patients['email'].str.lower() == email.lower())
                            ]
                            
                            if not match.empty:
                                st.session_state.logged_in = True
                                st.session_state.user_role = "Patient"
                                st.session_state.current_user_id = match.iloc[0]['patient_id']
                                st.success(f"Verified: Welcome back, {match.iloc[0]['patient_name']}")
                                st.rerun()
                            else:
                                st.error("Verification failed. Please check your credentials.")
                        else:
                            st.error("System database unavailable. Please contact administrator.")
                    else:
                        st.warning("Please enter both ID and Email.")
            
            else:
                username = st.text_input("Clinician ID", placeholder="Enter username")
                password = st.text_input("Access Key", type="password", placeholder="••••••••")
                submit = st.form_submit_button("Sign In as Clinician")
                
                if submit:
                    if username == "admin" and password == "pdcore":
                        st.session_state.logged_in = True
                        st.session_state.user_role = "Clinician"
                        st.success("Authorization successful.")
                        st.rerun()
                    else:
                        st.error("Invalid Clinician credentials.")
        
        # Restored Demo Credentials for the new patient list
        if login_type == "Patient":
            with st.expander("🔑 Patient Access List (Demo)"):
                st.info("Use the Patient ID and Email below to sign in:")
                df_demo = st.session_state.get('df_patients', pd.DataFrame())
                if not df_demo.empty:
                    # Sort by ID and take top 10 to ensure no shuffling
                    df_stable = df_demo.sort_values('patient_id').head(10)
                    st.table(df_stable[['patient_id', 'patient_name', 'email']])
                else:
                    st.warning("Database still initializing...")

if __name__ == "__main__":
    # If run standalone, show the login page
    render_login()
