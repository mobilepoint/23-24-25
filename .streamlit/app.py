import streamlit as st
import pandas as pd
import os
from supabase import create_client

# Config pagină
st.set_page_config(page_title="ServicePack Reports", layout="wide")

st.title("📊 ServicePack Reports")

# Citește secretele din Streamlit Cloud (Settings → Secrets)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("❌ Lipsesc variabilele SUPABASE_URL și SUPABASE_KEY. Adaugă-le în Streamlit Cloud → Secrets.")
    st.stop()

# Creează clientul
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    st.success("Conexiune la Supabase OK.")
except Exception as e:
    st.error(f"Eroare conectare Supabase: {e}")
    st.stop()

# Afișează ultimele 5 luni din registry
try:
    data = supabase.table("period_registry").select("*").order("period_month", desc=True).limit(5).execute()
    df = pd.DataFrame(data.data)
    if df.empty:
        st.info("Nu există înregistrări în ops.period_registry.")
    else:
        st.subheader("Ultimele luni din ops.period_registry")
        st.dataframe(df)
except Exception as e:
    st.warning(f"Nu pot citi ops.period_registry: {e}")
