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
    s = s.replace(",", "")  # scoate separatorul de mii
    try:
        return float(s)
    except Exception:
        return 0.0

def extract_last_parenthesized(text: str) -> Optional[str]:
    if not isinstance(text, str):
        text = str(text or "")
    matches = re.findall(r'\(([^()]*)\)', text)
    return matches[-1].strip() if matches else None

def get_period_row(sb: Client, period: date) -> Optional[dict]:
    """Citește rândul din public.period_registry pentru perioada dată."""
    try:
        resp = sb.table("period_registry").select("*").eq("period_month", period.isoformat()).execute()
        rows = resp.data or []
        return rows[0] if rows else None
    except Exception:
        return None

# ===================== STATUS BAR ========================
now_ro = datetime.now(tz=TZ)
lcm = last_completed_month(now_ro)
st.info(f"🗓️ Ultima lună încheiată (cerută pentru încărcare standard): **{lcm.strftime('%Y-%m')}**")

# ===================== TABS UI ===========================
tab_status, tab_upload, tab_consol = st.tabs(["📅 Status pe luni", "⬆️ Upload fișiere", "✅ Consolidare & Rapoarte"])

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

# ---------- TAB UPLOAD ----------
with tab_upload:
    st.subheader("Încarcă fișierul pentru luna acceptată")

    file_type = st.radio("Tip fișier", ["Profit pe produs", "Mișcări stocuri"], horizontal=True)
    uploaded_file = st.file_uploader("Alege fișierul Excel/CSV", type=["xlsx", "xls", "csv"])

    if uploaded_file is not None:
        # citim primele 10 rânduri pentru a detecta perioada
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                head = pd.read_csv(uploaded_file, nrows=10, header=None)
                uploaded_file.seek(0)
            else:
                head = pd.read_excel(uploaded_file, nrows=10, header=None, engine="openpyxl")
                uploaded_file.seek(0)
        except Exception as e:
            st.error(f"Nu pot citi antetul fișierului: {e}")
            st.stop()

        period = extract_period_from_header(head)
        if not period:
            st.error("Nu am putut detecta perioada din antet (rândul 5). Verifică fișierul.")
            st.stop()

        st.write(f"📄 **Perioadă detectată:** {period.strftime('%Y-%m')}")

        # Permitem upload doar pentru ultima lună încheiată (deocamdată fără backfill)
        if period != lcm:
            st.error(f"Fișierul este pentru {period.strftime('%Y-%m')}, dar aici acceptăm doar **{lcm.strftime('%Y-%m')}**.")
            st.stop()

        rows_json = []
        try:
            if file_type == "Profit pe produs":
                # skip primele 10 rânduri; rândul 11 e header
                df = pd.read_excel(uploaded_file, skiprows=10, engine="openpyxl")
                col_prod = next((c for c in df.columns if str(c).strip().lower().startswith("produs")), None)
                col_net  = next((c for c in df.columns if str(c).strip().lower().startswith("vanzari nete")), None)
                col_cogs = next((c for c in df.columns if str(c).strip().lower().startswith("costul bunurilor")), None)
                if not all([col_prod, col_net, col_cogs]):
                    raise ValueError("Nu am găsit coloanele 'Produsul', 'Vânzări nete', 'Costul bunurilor vândute'.")

                for _, r in df.iterrows():
                    sku = extract_last_parenthesized(r[col_prod])
                    if not sku:
                        continue
                    net_sales = parse_number(r[col_net])
                    cogs = parse_number(r[col_cogs])
                    rows_json.append({"sku": sku, "net_sales_wo_vat": net_sales, "cogs_wo_vat": cogs})

                if not rows_json:
                    raise ValueError("Nu am extras niciun rând valid (SKU).")

                res = sb.rpc("load_profit_file", {"p_period": period.isoformat(), "p_source_path": uploaded_file.name, "p_rows": rows_json}).execute()
                st.success(f"Încărcat PROFIT pentru {period.strftime('%Y-%m')}. file_id: {res.data}")

            else:  # Mișcări stocuri
                df = pd.read_excel(uploaded_file, skiprows=9, engine="openpyxl")
                col_sku = next((c for c in df.columns if str(c).strip().lower() in ["cod", "cod.1", "sku"]), None)
                col_qty_open = next((c for c in df.columns if str(c).strip().lower().startswith("stoc initial")), None)
                col_qty_in   = next((c for c in df.columns if str(c).strip().lower() == "intrari"), None)
                col_qty_out  = next((c for c in df.columns if str(c).strip().lower().startswith("iesiri") and "." not in str(c)), None)
                col_qty_close= next((c for c in df.columns if str(c).strip().lower().startswith("stoc final")), None)
                col_val_open = next((c for c in df.columns if str(c).strip().lower().startswith("sold initial")), None)
                col_val_in   = "Intrari.1" if "Intrari.1" in df.columns else next((c for c in df.columns if str(c).strip().lower()=="intrari" and c != col_qty_in), None)
                col_val_out  = next((c for c in df.columns if str(c).strip().lower()=="iesiri.1"), None)
                if not col_val_out:
                    dup_iesiri = [c for c in df.columns if str(c).strip().lower().startswith("iesiri")]
                    if len(dup_iesiri) >= 2:
                        col_val_out = dup_iesiri[1]
                col_val_close= next((c for c in df.columns if str(c).strip().lower().startswith("sold final")), None)

                if not all([col_sku, col_qty_open, col_qty_in, col_qty_out, col_qty_close, col_val_open, col_val_in, col_val_out, col_val_close]):
                    raise ValueError("Nu am găsit toate coloanele de cantități/valori (Stoc initial/Intrari/Iesiri/Stoc final + Sold initial/Intrari/Iesiri/Sold final).")

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

                res = sb.rpc("load_miscari_file", {"p_period": period.isoformat(), "p_source_path": uploaded_file.name, "p_rows": rows_json}).execute()
                st.success(f"Încărcat MISCĂRI pentru {period.strftime('%Y-%m')}. file_id: {res.data}")

                sb.rpc("update_balances_for_period", {"p_period": period.isoformat()}).execute()
                st.info("Balanțele cantități/valori au fost verificate și marcate în registry.")

        except Exception as e:
            st.error(f"Eroare la procesarea fișierului: {e}")

    st.divider()
    st.caption("După ce ai încărcat **ambele** fișiere pentru luna acceptată și nu ai erori, folosește tabul „Consolidare & Rapoarte”.")

# ---------- TAB CONSOLIDARE & RAPOARTE ----------
with tab_consol:
    st.subheader(f"Consolidare pentru {lcm.strftime('%Y-%m')}")

    row = get_period_row(sb, lcm)
    if not row:
        st.warning("Nu există încă intrări pentru această lună în registru. Încarcă mai întâi fișierele în tabul „Upload”.")
    else:
        cols = st.columns(4)
        cols[0].metric("Profit încărcat?", "DA" if row.get("profit_loaded") else "NU")
        cols[1].metric("Mișcări încărcate?", "DA" if row.get("miscari_loaded") else "NU")
        cols[2].metric("Balanță cantități OK?", "DA" if row.get("balance_ok_qty") else "NU")
        cols[3].metric("Balanță valori OK?", "DA" if row.get("balance_ok_val") else "NU")

        ready = all([
            row.get("profit_loaded") is True,
            row.get("miscari_loaded") is True,
            row.get("balance_ok_qty") is True,
            row.get("balance_ok_val") is True,
        ])

        if row.get("consolidated_to_core"):
            st.success("✅ Luna este deja consolidată. Rapoartele pot folosi `mart.sales_monthly`.")
        elif not ready:
            st.error("Nu poți consolida încă. Asigură-te că ambele fișiere sunt încărcate și balanțele sunt OK.")
        else:
            if st.button("🚀 Consolidează luna"):
                try:
                    sb.rpc("consolidate_month", {"p_period": lcm.isoformat()}).execute()
                    st.success("Consolidare reușită. `core.*` a fost suprascris pentru această lună, iar `mart.sales_monthly` a fost reîmprospătat.")
                except Exception as e:
                    st.error(f"Eroare la consolidare: {e}")

    st.divider()
    st.subheader("Rapoarte")
    if row and row.get("consolidated_to_core"):
        st.success("Rapoartele pot fi generate (vom adăuga pagini dedicate în pasul următor: Top sellers, Velocity, Recomandări).")
    else:
        st.warning("Rapoartele sunt blocate până când **ultima lună încheiată** este consolidată.")
