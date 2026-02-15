import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import os

# -----------------------------
# إعدادات الصفحة
st.set_page_config(
    page_title="⚖️ Weight & BMI Predictor",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>⚖️ Weight & BMI Predictor</h1>
    <p style='text-align: center; color: #555;'>Predict your weight and BMI with a simple interface!</p>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# dataset صغير لتدريب النموذج
data = {
    "Height_cm": [150, 160, 170, 180, 190],
    "Weight_kg": [50, 60, 65, 75, 85]
}
df = pd.DataFrame(data)
X = df[["Height_cm"]]
y = df["Weight_kg"]

# تدريب نموذج Linear Regression
model = LinearRegression()
model.fit(X, y)

# -----------------------------
# Sidebar لإدخال البيانات
st.sidebar.header("Enter your details")
name = st.sidebar.text_input("Name:")
height = st.sidebar.number_input("Height (cm):", min_value=100, max_value=220, value=170)

# ملف لحفظ البيانات
filename = "weight_history.csv"
if os.path.exists(filename):
    history = pd.read_csv(filename)
else:
    history = pd.DataFrame(columns=["Name", "Height (cm)", "Predicted Weight (kg)", "BMI", "BMI Category"])

# -----------------------------
# زر التنبؤ
if st.sidebar.button("Predict"):
    predicted_weight = model.predict(np.array([[height]]))[0]
    bmi = predicted_weight / ((height/100)**2)
    
    # تصنيف BMI
    if bmi < 18.5:
        bmi_cat = "Underweight"
    elif bmi < 25:
        bmi_cat = "Normal weight"
    elif bmi < 30:
        bmi_cat = "Overweight"
    else:
        bmi_cat = "Obese"
    
    # عرض النتائج بشكل جذاب
    st.markdown(f"<h2 style='color: #27AE60;'>✅ {name} - Predicted Weight: {predicted_weight:.1f} kg</h2>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: #F39C12;'>BMI: {bmi:.1f} ({bmi_cat})</h3>", unsafe_allow_html=True)
    
    # حفظ البيانات في الجدول والملف
    new_row = {
        "Name": name,
        "Height (cm)": height,
        "Predicted Weight (kg)": round(predicted_weight,1),
        "BMI": round(bmi,1),
        "BMI Category": bmi_cat
    }
    history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
    history.to_csv(filename, index=False)

# -----------------------------
# عرض جدول المحاولات السابقة بشكل جذاب
st.header("📊 History of Predictions")
st.dataframe(history.style.set_properties(**{
    'background-color': '#f0f2f6',
    'color': '#000000',
    'border-color': '#ffffff'
}))

# -----------------------------
# حذف صفوف من الجدول
st.header("🗑️ Delete a Record")
if not history.empty:
    # اختر الصفوف التي تريد حذفها
    rows_to_delete = st.multiselect(
        "Select the records to delete by Name and Height",
        options=history.index,
        format_func=lambda x: f"{history.loc[x, 'Name']} - {history.loc[x, 'Height (cm)']} cm"
    )
    
    if st.button("Delete Selected"):
        history = history.drop(index=rows_to_delete)
        history.to_csv(filename, index=False)
        st.success("Selected records deleted successfully!")

# -----------------------------
# رسم مخطط تفاعلي باستخدام Plotly
st.header("📈 Height vs Predicted Weight")
if not history.empty:
    fig = px.scatter(
        history,
        x="Height (cm)",
        y="Predicted Weight (kg)",
        text="Name",
        color="BMI Category",
        color_discrete_map={
            "Underweight": "#3498DB",
            "Normal weight": "#2ECC71",
            "Overweight": "#F1C40F",
            "Obese": "#E74C3C"
        },
        hover_data={"Height (cm)":True, "Predicted Weight (kg)":True, "BMI":True}
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers+text'))
    st.plotly_chart(fig, use_container_width=True)

