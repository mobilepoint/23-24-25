import os
import re
import pandas as pd
import streamlit as st
from typing import Optional
from datetime import datetime, date
from zoneinfo import ZoneInfo
from supabase import create_client, Client

# ===================== CONFIG PAGINĂ =====================
st.set_page_config(page_title="ServicePack Reports", layout="wide")
st.title("📊 ServicePack Reports")

# ===================== SUPABASE CREDS ====================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("❌ Lipsesc SUPABASE_URL / SUPABASE_KEY în Streamlit → Settings → Secrets.")
    st.stop()

try:
    sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    st.success("Conexiune la Supabase OK.")
except Exception as e:
    st.error(f"Eroare conectare Supabase: {e}")
    st.stop()

# ===================== HELPERI BUSINESS ==================
TZ = ZoneInfo("Europe/Bucharest")

def last_completed_month(today: datetime) -> date:
    first_of_this_month = datetime(today.year, today.month, 1, tzinfo=TZ).date()
    prev_month_last_day = first_of_this_month.replace(day=1) - pd.Timedelta(days=1)
    return prev_month_last_day.replace(day=1)

def extract_period_from_header(df_head: pd.DataFrame) -> Optional[date]:
    text = " ".join([" ".join(map(str, row.dropna().astype(str).tolist())) for _, row in df_head.iterrows()])
    m = re.search(r'(\d{2}/\d{2}/\d{4})\s*-\s*(\d{2}/\d{2}/\d{4})', text)
    if not m:
        return None
    start = pd.to_datetime(m.group(1), dayfirst=True).date()
    return start.replace(day=1)

def parse_number(x) -> float:
    if x is None:
        return 0.0
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return 0.0
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return 0.0

def extract_last_parenthesized(text: str) -> Optional[str]:
    if not isinstance(text, str):
        text = str(text or "")
    matches = re.findall(r'\(([^()]*)\)', text)
    return matches[-1].strip() if matches else None

# --- utilitare pentru identificare robustă a coloanelor în PROFIT ---
_norm_tbl = str.maketrans({c: "" for c in " .,_-/%()[]{}:"})
def norm(s: str) -> str:
    return str(s).strip().lower().translate(_norm_tbl)

def find_profit_columns(df: pd.DataFrame):
    col_prod = col_net = col_cogs = None
    ncols = len(df.columns)
    for c in df.columns:
        n = norm(c)
        if col_prod is None and ("produs" in n):
            col_prod = c
        if col_net is None and ("vanzari" in n and "nete" in n):
            col_net = c
        if col_cogs is None and (("cost" in n and ("bunurilor" in n or "bunuri" in n) and ("vandute" in n or "vandut" in n)) or "cogs" in n):
            col_cogs = c
    if col_prod is None and ncols >= 2: col_prod = df.columns[1]      # B
    if col_net  is None and ncols >= 5: col_net  = df.columns[4]      # E
    if col_cogs is None and ncols >= 6: col_cogs = df.columns[5]      # F
    return col_prod, col_net, col_cogs

# ===================== STATUS BAR ========================
now_ro = datetime.now(tz=TZ)
lcm = last_completed_month(now_ro)
st.info(f"🗓️ Ultima lună încheiată (cerută pentru încărcare standard): **{lcm.strftime('%Y-%m')}**")

# ===================== TABS UI ===========================
tab_status, tab_upload, tab_consol, tab_debug = st.tabs(
    ["📅 Status pe luni", "⬆️ Upload fișiere", "✅ Consolidare & Rapoarte", "🧪 Debug balanțe"]
)

# ---------- TAB STATUS ----------
with tab_status:
    st.subheader("Stare pe luni (public.period_registry)")
    try:
        resp = sb.table("period_registry").select("*").order("period_month", desc=True).limit(24).execute()
        df = pd.DataFrame(resp.data)
        if df.empty:
            st.info("Nu există încă înregistrări în period_registry.")
        else:
            df["period_month"] = pd.to_datetime(df["period_month"]).dt.date
            st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Nu pot citi period_registry: {e}")

# ---------- HELPERE citire fișiere ----------
def read_head_any(uploaded_file, nrows: int) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, nrows=nrows, header=None)
        uploaded_file.seek(0)
        return df
    try:
        df = pd.read_excel(uploaded_file, nrows=nrows, header=None, engine="openpyxl")
        uploaded_file.seek(0)
        return df
    except Exception:
        pass
    try:
        df = pd.read_excel(uploaded_file, nrows=nrows, header=None, engine="xlrd")
        uploaded_file.seek(0)
        return df
    except Exception:
        raise RuntimeError("Nu pot citi fișierul: nu este Excel valid (.xlsx/.xls) și nici CSV.")

def read_full_any(uploaded_file, skiprows: int) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, skiprows=skiprows)
        uploaded_file.seek(0)
        return df
    try:
        df = pd.read_excel(uploaded_file, skiprows=skiprows, engine="openpyxl")
        uploaded_file.seek(0)
        return df
    except Exception:
        df = pd.read_excel(uploaded_file, skiprows=skiprows, engine="xlrd")
        uploaded_file.seek(0)
        return df

# ---------- TAB UPLOAD ----------
with tab_upload:
    st.subheader("Încarcă fișierul pentru luna acceptată")

    file_type = st.radio("Tip fișier", ["Profit pe produs", "Mișcări stocuri"], horizontal=True)
    uploaded_file = st.file_uploader("Alege fișierul Excel/CSV", type=["xlsx", "xls", "csv"])

    if uploaded_file is not None:
        try:
            head = read_head_any(uploaded_file, nrows=10)
        except Exception as e:
            st.error(f"Nu pot citi antetul fișierului: {e}")
            st.stop()

        period = extract_period_from_header(head)
        if not period:
            st.error("Nu am putut detecta perioada din antet (rândul 5). Verifică fișierul.")
            st.stop()

        st.write(f"📄 **Perioadă detectată:** {period.strftime('%Y-%m')}")

        if period != lcm:
            st.error(f"Fișierul este pentru {period.strftime('%Y-%m')}, dar aici acceptăm doar **{lcm.strftime('%Y-%m')}**.")
            st.stop()

        rows_json = []
        try:
            if file_type == "Profit pe produs":
                df = read_full_any(uploaded_file, skiprows=10)
                col_prod, col_net, col_cogs = find_profit_columns(df)
                if col_prod is None or col_net is None or col_cogs is None:
                    raise ValueError("Nu am reușit să identific coloanele pentru Produs / Vanzari nete / Cost bunuri vandute.")

                for _, r in df.iterrows():
                    sku = extract_last_parenthesized(r[col_prod])
                    if not sku:
                        continue
                    net_sales = parse_number(r[col_net])
                    cogs = parse_number(r[col_cogs])
                    rows_json.append({"sku": sku, "net_sales_wo_vat": net_sales, "cogs_wo_vat": cogs})

                if not rows_json:
                    raise ValueError("Nu am extras niciun rând valid (SKU).")

                res = sb.rpc("load_profit_file", {
                    "p_period": period.isoformat(),
                    "p_source_path": uploaded_file.name,
                    "p_rows": rows_json
                }).execute()
                st.success(f"Încărcat PROFIT pentru {period.strftime('%Y-%m')}. file_id: {res.data}")

            else:  # Mișcări stocuri
                df = read_full_any(uploaded_file, skiprows=9)
                norm_map = {c: norm(c) for c in df.columns}
                col_sku = next((c for c in df.columns if norm_map[c] in ["cod", "cod1", "sku"]), None)
                col_qty_open  = next((c for c in df.columns if norm_map[c].startswith("stocinitial")), None)
                col_qty_in    = next((c for c in df.columns if norm_map[c] == "intrari"), None)
                col_qty_out   = next((c for c in df.columns if norm_map[c].startswith("iesiri") and "." not in str(c)), None)
                col_qty_close = next((c for c in df.columns if norm_map[c].startswith("stocfinal")), None)
                col_val_open  = next((c for c in df.columns if norm_map[c].startswith("soldinitial")), None)
                col_val_in    = "Intrari.1" if "Intrari.1" in df.columns else next((c for c in df.columns if norm_map[c]=="intrari" and c != col_qty_in), None)
                col_val_out   = next((c for c in df.columns if norm_map[c]=="iesiri.1"), None)
                if not col_val_out:
                    dup_iesiri = [c for c in df.columns if norm_map[c].startswith("iesiri")]
                    if len(dup_iesiri) >= 2:
                        col_val_out = dup_iesiri[1]
                col_val_close = next((c for c in df.columns if norm_map[c].startswith("soldfinal")), None)

                if not all([col_sku, col_qty_open, col_qty_in, col_qty_out, col_qty_close, col_val_open, col_val_in, col_val_out, col_val_close]):
                    raise ValueError("Nu am găsit toate coloanele necesare în mișcări stocuri.")

                for _, r in df.iterrows():
                    sku = str(r[col_sku]).strip()
                    if not sku or sku.lower() in ("nan", "none"):
                        continue
                    rows_json.append({
                        "sku": sku,
                        "qty_open":  parse_number(r[col_qty_open]),
                        "qty_in":    parse_number(r[col_qty_in]),
                        "qty_out":   parse_number(r[col_qty_out]),
                        "qty_close": parse_number(r[col_qty_close]),
                        "val_open":  parse_number(r[col_val_open]),
                        "val_in":    parse_number(r[col_val_in]),
                        "val_out":   parse_number(r[col_val_out]),
                        "val_close": parse_number(r[col_val_close]),
                    })

                if not rows_json:
                    raise ValueError("Nu am extras niciun rând valid (SKU).")

                res = sb.rpc("load_miscari_file", {
                    "p_period": period.isoformat(),
                    "p_source_path": uploaded_file.name,
                    "p_rows": rows_json
                }).execute()
                st.success(f"Încărcat MISCĂRI pentru {period.strftime('%Y-%m')}. file_id: {res.data}")

                sb.rpc("update_balances_for_period", {"p_period": period.isoformat()}).execute()
                st.info("Balanțele cantități/valori au fost verificate și marcate în registry.")

        except Exception as e:
            st.error(f"Eroare la procesarea fișierului: {e}")

# ---------- CONSOLIDARE + DEBUG (rămân la fel, scurtat pt spațiu) ----------
# ... restul codului pentru tab_consol și tab_debug rămâne identic ...
