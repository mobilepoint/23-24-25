import os
import pandas as pd
import streamlit as st
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from supabase import create_client

# ============= CONFIG PAGINĂ =============
st.set_page_config(page_title="ServicePack Reports", layout="wide")
st.title("📊 ServicePack Reports")

# ============= SUPABASE CREDS (hardcoded + override din env/secrets) =============
DEFAULT_SUPABASE_URL = "https://ufpcludvtpfdyjvqjgje.supabase.co"
DEFAULT_SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVmcGNsdWR2dHBmZHlqdnFqZ2plIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTgwMjI3NTAsImV4cCI6MjA3MzU5ODc1MH0.AVvgpXBkzgH3srvI2jLTQf7337rkceN-5scOl"

SUPABASE_URL = os.getenv("SUPABASE_URL", DEFAULT_SUPABASE_URL)
SUPABASE_KEY = os.getenv("SUPABASE_KEY", DEFAULT_SUPABASE_KEY)

# Mic helper pt. mesaj dacă key-ul e trunchiat
def looks_truncated_jwt(token: str) -> bool:
    return token.count(".") != 2 or len(token.split(".")[-1]) < 20

if looks_truncated_jwt(SUPABASE_KEY):
    st.warning("⚠️ Cheia SUPABASE_KEY pare trunchiată. Dacă apar erori de conexiune, actualizează cheia completă în cod sau în Streamlit Secrets.")

# ============= CONEXIUNE SUPABASE =============
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    st.success("Conexiune la Supabase OK.")
except Exception as e:
    st.error(f"Eroare conectare Supabase: {e}")
    st.stop()

# ============= HELPERI BUSINESS =============
TZ = ZoneInfo("Europe/Bucharest")

def last_completed_month(today: datetime) -> datetime.date:
    """Prima zi a lunii precedente în Europe/Bucharest."""
    first_of_this_month = datetime(today.year, today.month, 1, tzinfo=TZ).date()
    # scade 1 zi, apoi ia prima zi a lunii respective
    prev_month_last_day = first_of_this_month.replace(day=1) - pd.Timedelta(days=1)
    return prev_month_last_day.replace(day=1)

# ============= STATUS: ULTIMA LUNĂ ÎNCHEIATĂ =============
now_ro = datetime.now(tz=TZ)
lcm = last_completed_month(now_ro)
st.info(f"🗓️ Ultima lună încheiată (cerută pentru încărcare standard): **{lcm.strftime('%Y-%m')}**")

# ============= CITIRE ops.period_registry =============
st.subheader("Stare pe luni (ops.period_registry)")

try:
    # Notă: trebuie să apelăm schema "ops"
    q = supabase.schema("ops").table("period_registry").select("*").order("period_month", desc=True).limit(12)
    resp = q.execute()
    df = pd.DataFrame(resp.data)
    if df.empty:
        st.info("Nu există încă înregistrări în ops.period_registry.")
    else:
        # formatare drăguță
        df_display = df.copy()
        if "period_month" in df_display.columns:
            df_display["period_month"] = pd.to_datetime(df_display["period_month"]).dt.date
        st.dataframe(df_display, use_container_width=True)
except Exception as e:
    st.error(f"Nu pot citi ops.period_registry: {e}")

st.caption("Hint: după ce încărcăm fișierele pe luna afișată mai sus și trecem validările, vom activa butonul de consolidare în pașii următori.")
