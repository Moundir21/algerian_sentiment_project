import re

def clean_text(text):
    text = text.lower()
    # normalize arabic
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("ى", "ي", text)

    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # إزالة الروابط
    text = re.sub(r'\@\w+|\#', '', text) # إزالة الإشارات
    text = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '', text) # إزالة الإيميلات
    text = re.sub(r'\d+', '', text) # إزالة الأرقام (مثل أرقام الهواتف)
    text = re.sub(r'[^\w\s]', '', text) # إزالة علامات الترقيم
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text) # إزالة التشكيل العربي
    
    # 2. تسوية البيانات (Data Normalization)
    text = re.sub(r'[إأآ]', 'ا', text) # توحيد الألف
    text = re.sub(r'ة', 'ه', text) # توحيد التاء المربوطة
    text = re.sub(r'ى', 'ي', text) # توحيد الألف المقصورة
    
    # إزالة التطويل (تكرار الحروف مرتين أو أكثر إلى حرف واحد)
    text = re.sub(r'(.)\1+', r'\1', text) # ملحوظة: قد يقلص هذا من الحروف المزدوجة الأصلية، تأكد من تعليمات دكتور المادة
    text = re.sub(r'\s+', ' ', text).strip() # إزالة المسافات المكررة

    return text.strip()