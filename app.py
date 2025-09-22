import os
import pandas as pd
import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo
from supabase import create_client, Client

# ============= CONFIG PAGINĂ =============
st.set_page_config(page_title="ServicePack Reports", layout="wide")
st.title("📊 ServicePack Reports")

# ============= SUPABASE CREDS DIN SECRETS =============
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("❌ Lipsesc variabilele SUPABASE_URL și/sau SUPABASE_KEY. Le adaugi în Streamlit → Settings → Secrets.")
    st.stop()

# ============= CONEXIUNE SUPABASE =============
try:
    sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    st.success("Conexiune la Supabase OK.")
except Exception as e:
    st.error(f"Eroare conectare Supabase: {e}")
    st.stop()

# ============= HELPERI BUSINESS =============
TZ = ZoneInfo("Europe/Bucharest")

def last_completed_month(today: datetime) -> datetime.date:
    first_of_this_month = datetime(today.year, today.month, 1, tzinfo=TZ).date()
    prev_month_last_day = first_of_this_month.replace(day=1) - pd.Timedelta(days=1)
    return prev_month_last_day.replace(day=1)

# ============= STATUS: ULTIMA LUNĂ ÎNCHEIATĂ =============
now_ro = datetime.now(tz=TZ)
lcm = last_completed_month(now_ro)
st.info(f"🗓️ Ultima lună încheiată (cerută pentru încărcare standard): **{lcm.strftime('%Y-%m')}**")

# ============= CITIRE public.period_registry (VIEW din ops.period_registry) =============
st.subheader("Stare pe luni (public.period_registry)")

try:
    resp = sb.table("period_registry").select("*").order("period_month", desc=True).limit(12).execute()
    df = pd.DataFrame(resp.data)
    if df.empty:
        st.info("Nu există încă înregistrări în period_registry.")
    else:
        df["period_month"] = pd.to_datetime(df["period_month"]).dt.date
        st.dataframe(df, use_container_width=True)
except Exception as e:
    st.error(f"Nu pot citi period_registry: {e}")

# mesaj final
st.caption(
    "Hint: după ce încărcăm fișierele pe luna afișată mai sus și trecem validările, "
    "vom activa butonul de consolidare în pașii următori."
)
