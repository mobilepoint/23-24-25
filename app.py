import os
import pandas as pd
import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo
from supabase import create_client, Client

# ============= CONFIG PAGINĂ =============
st.set_page_config(page_title="ServicePack Reports", layout="wide")
st.title("📊 ServicePack Reports")

# ============= CREDENȚIALE SUPABASE DIN SECRETS =============
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def jwt_looks_valid(token: str | None) -> bool:
    if not token:
        return False
    parts = token.split(".")
    return len(parts) == 3 and all(len(p) > 10 for p in parts)

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("❌ Lipsesc variabilele `SUPABASE_URL` și/sau `SUPABASE_KEY` în Streamlit → Settings → Secrets.")
    st.stop()

if not jwt_looks_valid(SUPABASE_KEY):
    st.error("❌ `SUPABASE_KEY` nu pare o cheie JWT completă. Copiază **anon public key** integral din Supabase → Project Settings → API.")
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

now_ro = datetime.now(tz=TZ)
lcm = last_completed_month(now_ro)
st.info(f"🗓️ Ultima lună încheiată (cerută pentru încărcare standard): **{lcm.strftime('%Y-%m')}**")

# ============= CITIRE ops.period_registry =============
st.subheader("Stare pe luni (ops.period_registry)")
try:
    # ATENȚIE: accesăm explicit schema 'ops'
    resp = sb.schema("ops").table("period_registry").select("*").order("period_month", desc=True).limit(12).execute()
    df = pd.DataFrame(resp.data)
    if df.empty:
        st.info("Nu există încă înregistrări în ops.period_registry.")
    else:
        df["period_month"] = pd.to_datetime(df["period_month"]).dt.date
        st.dataframe(df, use_container_width=True)
except Exception as e:
    st.error(
        "Nu pot citi `ops.period_registry`.\n\n"
        "Cauze probabile:\n"
        "• Cheia `anon` e greșită (vezi Project Settings → API → anon public key);\n"
        "• URL greșit (trebuie să fie *.supabase.co exact ca în dashboard);\n"
        "• Dacă folosești `anon` și ai RLS activ pe tabel, adaugă o policy `SELECT` pentru rolul public.\n\n"
        f"Detaliu tehnic: {e}"
    )

st.caption("Hint: după ce încărcăm fișierele pe luna afișată mai sus și trecem validările, vom activa butonul de consolidare în pașii următori.")
