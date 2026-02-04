# =========================================
# Noor AI Ultra – Phase 1 (Top Version)
# Business Content OS for "نور الوجود"
# =========================================

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from openai import OpenAI

# =========================================
# إعدادات عامة
# =========================================

st.set_page_config(
    page_title="Noor AI Ultra",
    page_icon="🚀",
    layout="wide"
)

DATA_DIR = "data"
EXPORT_DIR = "exports"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

PROFILE_FILE = os.path.join(DATA_DIR, "business_profile.json")
COST_FILE = os.path.join(DATA_DIR, "costs_ultra.json")
LIBRARY_FILE = os.path.join(DATA_DIR, "library_ultra.json")

# =========================================
# أدوات مساعدة عامة
# =========================================

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(path, default):
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    return default

def clean_json_line(line: str) -> str:
    line = line.strip()
    if line.startswith("`"):
        line = line.strip("`")
    if line.lower().startswith("json"):
        line = line[4:].strip()
    line = line.strip(", ")
    return line

# =========================================
# نظام التكلفة
# =========================================

def load_costs():
    return load_json(COST_FILE, {"total_tokens": 0, "total_cost": 0.0, "calls": 0})

def save_costs(costs):
    save_json(COST_FILE, costs)

def add_cost(tokens):
    if tokens is None:
        return
    costs = load_costs()
    costs["total_tokens"] += tokens
    costs["total_cost"] += (tokens / 1_000_000) * 0.60
    costs["calls"] += 1
    save_costs(costs)

# =========================================
# Sidebar – إعدادات النشاط و API
# =========================================

st.sidebar.title("⚙️ الإعدادات")

api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not api_key:
    st.sidebar.warning("أدخل API Key للمتابعة")
    st.stop()

client = OpenAI(api_key=api_key)

# بيانات النشاط
profile_default = {
    "name": "نور الوجود",
    "services": "صيانة عامة، ديكور داخلي، رخام",
    "audience": "أصحاب الفلل والشقق في أبوظبي",
    "dialect": "لهجة خليجية / إماراتية",
    "focus_services": "صيانة، ديكور، رخام",
    "phone": "971500000000"
}
profile = load_json(PROFILE_FILE, profile_default)

st.sidebar.subheader("🏢 بيانات النشاط")

profile["name"] = st.sidebar.text_input("اسم النشاط", profile["name"])
profile["services"] = st.sidebar.text_area("الخدمات الرئيسية", profile["services"])
profile["audience"] = st.sidebar.text_area("الجمهور المستهدف", profile["audience"])
profile["dialect"] = st.sidebar.text_input("اللهجة المفضلة", profile["dialect"])
profile["focus_services"] = st.sidebar.text_input("الخدمات التي نركز عليها", profile["focus_services"])
profile["phone"] = st.sidebar.text_input("رقم التواصل (واتساب)", profile["phone"])

if st.sidebar.button("💾 حفظ بيانات النشاط"):
    save_json(PROFILE_FILE, profile)
    st.sidebar.success("تم حفظ بيانات النشاط")

# تكلفة
costs = load_costs()
st.sidebar.subheader("💰 تكلفة الاستخدام")
st.sidebar.metric("إجمالي التكلفة", f"${costs['total_cost']:.4f}")
st.sidebar.metric("عدد الاستدعاءات", costs["calls"])
st.sidebar.metric("إجمالي Tokens", f"{costs['total_tokens']:,}")

if st.sidebar.button("🗑️ مسح بيانات التكلفة"):
    if os.path.exists(COST_FILE):
        os.remove(COST_FILE)
    st.sidebar.success("تم مسح بيانات التكلفة")
    st.experimental_rerun()

# =========================================
# Smart Types + Times
# =========================================

CONTENT_TYPES = [
    "قبل وبعد",
    "نصيحة سريعة",
    "عرض/خصم",
    "تعريف بخدمة",
    "شهادة عميل",
    "سؤال تفاعلي"
]

BEST_TIMES = [
    "10:00 صباحاً",
    "1:00 ظهراً",
    "6:00 مساءً",
    "9:00 مساءً"
]

def build_schedule(days: int):
    return [CONTENT_TYPES[i % len(CONTENT_TYPES)] for i in range(days)]

# =========================================
# AI Batch Factory
# =========================================

def build_prompt(days: int):
    schedule = build_schedule(days)
    today = datetime.today()

    lines = []

    header = f"""
أنت خبير تسويق لشركة خدمات اسمها "{profile['name']}" تعمل في مجال الصيانة والديكور والرخام في أبوظبي.

الخدمات:
{profile['services']}

الجمهور:
{profile['audience']}

اللهجة:
{profile['dialect']}

ركّز أكثر على:
{profile['focus_services']}

المطلوب:
إنشاء خطة محتوى شهرية كاملة.

لكل يوم أرجع JSON فقط يحتوي الحقول التالية:

date          (YYYY-MM-DD)
type          (نوع اليوم)
title         (عنوان قصير)
caption_long  (نص المنشور الرئيسي)
story_caption (نص قصير للستوري)
reel_idea     (فكرة Reel)
cta           (دعوة واضحة للتواصل عبر واتساب {profile['phone']})
hashtags      (5–8 هاشتاقات مناسبة)
google_post   (نص منشور Google Business)
post_time     (وقت النشر من الأوقات المقترحة)

الشروط:
- عربي احترافي + لمسة خليجية
- غير مكرر
- مقنع وواقعي
"""
    lines.append(header.strip())

    for i in range(days):
        date = (today + timedelta(days=i)).strftime("%Y-%m-%d")
        t = schedule[i]
        time = BEST_TIMES[i % len(BEST_TIMES)]
        lines.append(f"- {date} | النوع: {t} | الوقت: {time}")

    lines.append("\nأعد JSON واحد لكل سطر فقط بدون أي شرح إضافي.")

    return "\n".join(lines)

def generate_month(days: int) -> pd.DataFrame:
    prompt = build_prompt(days)

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.8,
            messages=[{"role": "user", "content": prompt}]
        )
    except Exception as e:
        st.error(f"خطأ في استدعاء الـ API: {e}")
        return pd.DataFrame()

    text = res.choices[0].message.content
    usage = getattr(res, "usage", None)
    tokens = getattr(usage, "total_tokens", None) if usage else None
    add_cost(tokens)

    rows = []
    for raw_line in text.splitlines():
        line = clean_json_line(raw_line)
        if not line:
            continue
        try:
            obj = json.loads(line)
            rows.append(obj)
        except Exception:
            continue

    if not rows:
        st.error("لم يتمكن النظام من قراءة أي JSON صالح من الرد.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    cols_order = [
        "date",
        "type",
        "title",
        "caption_long",
        "story_caption",
        "reel_idea",
        "cta",
        "hashtags",
        "google_post",
        "post_time",
    ]
    df = df[[c for c in cols_order if c in df.columns]]

    return df

# =========================================
# مكتبة Ultra
# =========================================

def load_library():
    return load_json(LIBRARY_FILE, [])

def save_to_library(df: pd.DataFrame, days: int):
    lib = load_library()
    lib.append({
        "timestamp": datetime.now().isoformat(),
        "days": days,
        "count": len(df),
        "data": df.to_dict(orient="records")
    })
    save_json(LIBRARY_FILE, lib)

# =========================================
# الواجهة الرئيسية
# =========================================

st.title("🚀 Noor AI Ultra – Phase 1 (Top)")
st.caption("Business Content OS – خطة شهر كاملة بضغطة واحدة")

tab_plan, tab_calendar, tab_library = st.tabs([
    "🧠 خطة الشهر",
    "🗓️ تقويم مبسط",
    "📚 المكتبة"
])

# =========================
# تبويب خطة الشهر
# =========================

with tab_plan:
    days = st.slider("عدد أيام الخطة", 7, 60, 30)

    if st.button("✨ ولّد خطة الشهر (Batch)", type="primary", use_container_width=True):
        with st.spinner("جاري توليد خطة شهرية كاملة..."):
            df = generate_month(days)
            if not df.empty:
                st.session_state.df = df
                st.session_state.days = days

                month_file = os.path.join(
                    DATA_DIR,
                    f"month-plan-{datetime.now().strftime('%Y-%m-%d-%H%M%S')}.json"
                )
                save_json(month_file, df.to_dict("records"))
                save_to_library(df, days)

                st.success("✅ تم توليد الخطة وحفظها في المكتبة")

    if "df" in st.session_state:
        st.subheader("📋 جدول الشهر (قابل للتعديل)")
        edited_df = st.data_editor(
            st.session_state.df,
            use_container_width=True
        )
        st.session_state.df = edited_df

        csv_data = edited_df.to_csv(index=False, encoding="utf-8-sig")
        json_data = edited_df.to_json(orient="records", force_ascii=False, indent=2)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ تحميل CSV",
                csv_data,
                file_name="noor-plan.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col2:
            st.download_button(
                "⬇️ تحميل JSON",
                json_data,
                file_name="noor-plan.json",
                mime="application/json",
                use_container_width=True
            )

# =========================
# تبويب التقويم
# =========================

with tab_calendar:
    if "df" not in st.session_state or st.session_state.df.empty:
        st.info("لا توجد خطة حالية. ولّد خطة من تبويب 'خطة الشهر'.")
    else:
        st.subheader("🗓️ تقويم مبسط للشهر")
        df = st.session_state.df
        cols_per_row = 7
        for i in range(0, len(df), cols_per_row):
            cols = st.columns(min(cols_per_row, len(df) - i))
            for j, (_, row) in enumerate(df.iloc[i:i+cols_per_row].iterrows()):
                col = cols[j]
                with col:
                    st.markdown(f"""
**{row.get('date','')}**  
{row.get('type','')}  
{row.get('title','')}  

🔹 {row.get('cta','')}
""")

# =========================
# تبويب المكتبة
# =========================

with tab_library:
    lib = load_library()
    if not lib:
        st.info("📭 المكتبة فارغة – ولّد خطة أولاً.")
    else:
        st.subheader(f"📚 المكتبة – {len(lib)} خطة محفوظة")
        for item in reversed(lib):
            ts = item["timestamp"][:19].replace("T", " ")
            label = f"📦 {ts} – {item['days']} يوم – {item['count']} سجل"
            with st.expander(label):
                df_lib = pd.DataFrame(item["data"])
                st.dataframe(df_lib, use_container_width=True)
