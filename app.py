import numpy as np
import streamlit as st
import cv2
from tensorflow.keras.models import load_model

# إعدادات الصفحة مع الأيقونة (الفراشة) والخلفية السوداء
st.set_page_config(page_title="Cataract Diagnosis using ML", page_icon="🦋", layout="centered")

# تخصيص خلفية الصفحة إلى اللون الأسود
st.markdown(
    """
    <style>
    .stApp {
        background-color: #000000;  /* تغيير خلفية الصفحة إلى الأسود */
        color: white;  /* النصوص بلون أبيض */
    }
    .stButton>button {
        background-color: #1d3557;  /* زر بلون أزرق غامق */
        color: white;
    }
    h1, h2, h3 {
        color: #1E90FF;  /* تغيير العنوان إلى اللون الأزرق */
    }
    .stFileUploader>div {
        background-color: #333333;  /* خلفية مربع رفع الصورة */
        border: 2px solid #1E90FF;  /* إطار بلون أزرق */
        color: white;
    }
    .stTextInput>div {
        margin-bottom: 20px;  /* إضافة مسافة بين النصوص */
    }
    .stFileUploader>div {
        margin-top: 20px;  /* إضافة مسافة بين مربع رفع الصورة والنصوص */
    }
    </style>
    """, unsafe_allow_html=True
)

# عرض العنوان باللغة الإنجليزية
st.title("Cataract Diagnosis using ML")

# تعليمات إضافية
st.write("This model is designed to detect cataracts in eye images. Please upload a clear image of the eye for diagnosis.")

# تحميل النموذج المدرب
@st.cache_resource
def load_model_cached(model_path):
    return load_model(model_path)

# تحميل النموذج المدرب
model_path = '/Users/mk/Downloads/vgg19_the_last_model.keras'  # استبدل بالمسار الصحيح
model = load_model_cached(model_path)

# رفع صورة جديدة
uploaded_file = st.file_uploader("Upload an eye image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # قراءة الصورة المرفوعة
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    
    # تغيير الحجم ليطابق المدخلات للنموذج
    img_resized = cv2.resize(img, (224, 224))  # التأكد من تصغير الصورة إلى الحجم المطلوب
    img_resized = np.array(img_resized) / 255.0  # تطبيع الصورة
    
    # عرض الصورة بحجم مناسب باستخدام use_container_width بدلاً من use_column_width
    st.image(img_resized, caption="Uploaded Image", use_container_width=True)

    # إضافة بعد للدفعة
    img_resized = np.expand_dims(img_resized, axis=0)
    
    # التنبؤ
    prediction = model.predict(img_resized)

    # عرض النتيجة
    pred_label = 'Cataract' if prediction[0] > 0.5 else 'Normal'
    
    st.write(f"Prediction: {pred_label}")
    st.write(f"Prediction probability: {prediction[0][0]}")  # احرص على أن تكون القيمة رقم واحد
