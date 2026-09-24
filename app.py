
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Gastritis Clinical Assessment / Tathmini ya Tumbo", layout="wide")

@st.cache_resource
def load_assets():
    scaler = joblib.load('scaler.pkl')
    rf_model = joblib.load('random_forest_model.pkl')
    feature_names = joblib.load('feature_names.pkl')
    p_values = joblib.load('p_values.pkl')
    return scaler, rf_model, feature_names, p_values

try:
    scaler, model, feature_names, p_values = load_assets()
except Exception as e:
    st.error("Model assets not found. Run training cells first. / Faili za mfano hazikupatikana.")
    st.stop()

# ---------------------------------------------------------
# 1. LANGUAGE DICTIONARY (TRANSLATIONS)
# ---------------------------------------------------------
translations = {
    "English": {
        "title": "🩺 Gastritis Diagnostic Assessment & Aftercare App",
        "subtitle": "Offline decision support system driven strictly by statistically significant clinical indicators ($p < 0.05$).",
        "lang_select": "Choose Language / Chagua Lugha",
        "sidebar_pvals": "Statistically Significant Predictors",
        "sidebar_inputs": "Enter Patient Data",
        "col_indicator": "Clinical Indicator",
        "sex_label": "Sex",
        "female": "Female",
        "male": "Male",
        "present": "Present (1)",
        "absent": "Absent (0)",
        "entered_profile": "Entered Patient Profile",
        "entered_val": "Entered Value",
        "diag_outcome": "Diagnostic Outcome",
        "btn_predict": "Generate Prediction & Care Plan",
        "risk_metric": "Gastritis Risk Probability",
        "high_risk_title": "⚠️ Diagnostic Result: High Probability of Gastritis Detected",
        "recs_title": "📋 Clinical Recommendations",
        "recs_content": """
        - **Further Evaluation:** Schedule endoscopic assessment (EGD) to verify mucosal erosion/inflammation.
        - **Diagnostic Testing:** Perform urea breath testing or stool antigen assay for *H. pylori*.
        - **Therapeutic Intervention:** Consider starting prescribed proton-pump inhibitors (PPIs) or antacids per clinician orders.
        - **NSAID Review:** Immediately review and manage NSAID or aspirin intake.
        """,
        "care_title": "🩹 Patient Aftercare Plan",
        "care_content": """
        1. **Dietary Adjustments:** Avoid triggers like heavy spices, citrus/acidic items, alcohol, and caffeine.
        2. **Meal Timing:** Eat smaller, structured meals every 3–4 hours instead of large, heavy portions.
        3. **Stress & Rest:** Implement stress reduction and avoid lying down for at least 2 hours post-meal.
        4. **Warning Signs:** Seek emergency care if experiencing persistent vomiting, black/tarry stools, or severe abdominal pain.
        """,
        "low_risk_title": "✅ Diagnostic Result: Low Risk / Normal Assessment",
        "maint_title": "🌿 Maintenance Recommendations",
        "maint_content": """
        - Maintain balanced hydration and a fiber-conscious diet.
        - Avoid taking strong pain relievers on an empty stomach.
        - Re-assess if symptoms emerge or persist.
        """
    },
    "Kiswahili": {
        "title": "🩺 Tathmini ya Kiuuguzi ya Tumbo (Gastritis) na Mpango wa Matunzo",
        "subtitle": "Mfumo wa kusaidia maamuzi ya matibabu unaotumia viashiria vilivyothibitishwa kisayansi ($p < 0.05$).",
        "lang_select": "Chagua Lugha / Choose Language",
        "sidebar_pvals": "Viashiria Muhimu Kisayansi",
        "sidebar_inputs": "Ingiza Taarifa za Mgonjwa",
        "col_indicator": "Kiashiria cha Kiuuguzi",
        "sex_label": "Jinsia",
        "female": "Mwanamke",
        "male": "Mwanaume",
        "present": "Ipo / ndiyo (1)",
        "absent": "Haipo / Hapana (0)",
        "entered_profile": "Taarifa za Mgonjwa Zilizoingizwa",
        "entered_val": "Thamani Iliyoingizwa",
        "diag_outcome": "Matokeo ya Uchunguzi",
        "btn_predict": "Zalisha Utabiri na Mpango wa Matunzo",
        "risk_metric": "Uwezekano wa Hatari ya Gastritis",
        "high_risk_title": "⚠️ Matokeo: Hatari Kubwa ya Gastritis Imegunduliwa",
        "recs_title": "📋 Mapendekezo ya Kitatibu",
        "recs_content": """
        - **Uchunguzi Zaidi:** Panga kufanya kipimo cha endoskopia (EGD) ili kukagua uvimbe au vidonda vya tumbo.
        - **Vipimo vya Maabara:** Fanya kipimo cha pumzi (urea breath test) au kinyesi ili kuchunguza bakteria wa *H. pylori*.
        - **Matibabu ya Dawa:** Anza kutumia dawa za kupunguza tindikali (PPIs au antacids) kwa maelekezo ya daktari.
        - **Acha Dawa za Maumivu:** Sitisha matumizi ya dawa za maumivu kama NSAIDs au aspirin.
        """,
        "care_title": "🩹 Mpango wa Matunzo Baada ya Huduma",
        "care_content": """
        1. **Mabadiliko ya Mlo:** Epuka vyakula vyenye viungo vingi, ukali/tindikali (kama ndimu), pombe, na kahawa.
        2. **Muda wa Kula:** Kula milo midogo mara kwa mara (kila masaa 3–4) badala ya kula chakula kingi kwa wakati mmoja.
        3. **Kupumzika:** Punguza msongo wa mawazo na usilale mara tu baada ya kula (subiri angalau masaa 2).
        4. **Dalili za Hatari:** Tafuta huduma ya dharura mara moja ukipata kutapika kulikoingia damu, kinyesi cheusi, au maumivu makali ya tumbo.
        """,
        "low_risk_title": "✅ Matokeo: Hatari ni Ndogo / Hali ni Kawaida",
        "maint_title": "🌿 Mapendekezo ya Kutunza Afya",
        "maint_content": """
        - Kunywa maji ya kutosha na kula vyakula vyenye nyuzi (fiber).
        - Epuka kumeza dawa kali za maumivu ukiwa na njaa.
        - Rudi kufanyiwa uchunguzi tena ukihisi dalili zozote.
        """
    }
}

# ---------------------------------------------------------
# 2. SIDEBAR LANGUAGE SELECTION
# ---------------------------------------------------------
lang_choice = st.sidebar.radio("🌐 Language / Lugha", options=["English", "Kiswahili"])
t = translations[lang_choice]

st.title(t["title"])
st.markdown(t["subtitle"])

# Display Statistically Significant Variables Info
st.sidebar.header(t["sidebar_pvals"])
p_val_df = pd.DataFrame([
    {t["col_indicator"]: col.replace('_', ' ').title(), "p-value": p_values[col]}
    for col in feature_names
])
st.sidebar.dataframe(p_val_df, hide_index=True)

st.sidebar.header(t["sidebar_inputs"])

# ---------------------------------------------------------
# 3. DYNAMIC INPUT FIELDS WITH BILINGUAL LABELS
# ---------------------------------------------------------
input_data = {}
for col in feature_names:
    col_clean = col.replace('_', ' ').title()
    
    # 1. Biological Sex handling
    if col.lower() in ['sex_m', 'sex_male', 'gender_m', 'gender_male']:
        sex_choice = st.sidebar.selectbox(t["sex_label"], options=[t["female"], t["male"]])
        input_data[col] = 1 if sex_choice == t["male"] else 0

    # 2. Continuous numerical clinical indicators
    elif any(k in col.lower() for k in ['age', 'score', 'level', 'count', 'rate', 'ph', 'bpm']):
        input_data[col] = st.sidebar.number_input(f"{col_clean}", value=30.0)

    # 3. Binary clinical symptoms
    else:
        input_data[col] = st.sidebar.selectbox(
            f"{col_clean}", 
            options=[0, 1], 
            format_func=lambda x: t["present"] if x == 1 else t["absent"]
        )

input_df = pd.DataFrame([input_data])

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(t["entered_profile"])
    st.dataframe(input_df.T.rename(columns={0: t["entered_val"]}))

with col2:
    st.subheader(t["diag_outcome"])
    if st.button(t["btn_predict"], type="primary"):
        scaled_features = scaler.transform(input_df)
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1]

        st.metric(t["risk_metric"], f"{probability * 100:.1f}%")

        if prediction == 1 or probability >= 0.5:
            st.error(t["high_risk_title"])
            st.markdown(f"### {t['recs_title']}")
            st.warning(t["recs_content"])
            st.markdown(f"### {t['care_title']}")
            st.info(t["care_content"])
        else:
            st.success(t["low_risk_title"])
            st.markdown(f"### {t['maint_title']}")
            st.write(t["maint_content"])
