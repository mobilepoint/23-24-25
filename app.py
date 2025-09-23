# app.py — Upload & Consolidation (XLS, mapare pe indici coloane)
import re
import datetime as dt
import pandas as pd
import streamlit as st
import xlrd
from supabase import create_client, Client

# ------------ SUPABASE ------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(page_title="Upload & Consolidation", layout="wide")
st.title("Upload & Consolidation")

# ------------ UTILS ------------
def read_xls_return_df(uploaded_file) -> pd.DataFrame:
    """Citește .XLS (format vechi) ca DataFrame brut (fără header), prin xlrd==1.2.0."""
    content = uploaded_file.getvalue()          # NU consumăm stream-ul
    book = xlrd.open_workbook(file_contents=content)
    sh = book.sheet_by_index(0)
    data = [sh.row_values(i) for i in range(sh.nrows)]
    return pd.DataFrame(data)

def first_day_from_period_line(line_text: str) -> dt.date:
    # Acceptă "Perioada: 01/01/2025 - 31/01/2025"
    m = re.search(r"(\d{2})/(\d{2})/(\d{4})\s*-\s*(\d{2})/(\d{2})/(\d{4})", str(line_text))
    if not m:
        raise ValueError("Nu pot extrage perioada din rândul 5.")
    d1, m1, y1 = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return dt.date(y1, m1, 1)

def last_closed_month(today=None) -> dt.date:
    today = today or dt.date.today()
    first_this = dt.date(today.year, today.month, 1)
    prev_last = first_this - dt.timedelta(days=1)
    return dt.date(prev_last.year, prev_last.month, 1)

def parse_money_comma_thousands(x):
    # Profit: "1,234.56" -> 1234.56 ; gol -> 0
    if x is None or x == "" or pd.isna(x):
        return 0.0
    return float(str(x).replace(",", ""))

def parse_money_plain(x):
    # Mișcări: "1234.56" (fără separator mii); gol -> 0
    if x is None or x == "" or pd.isna(x):
        return 0.0
    return float(x)

def to_int0(x):
    if x is None or x == "" or pd.isna(x):
        return 0
    return int(round(float(x)))

def extract_sku_from_b(product_text: str):
    """Ia ultimul (...) din denumire; doar strip(), fără alte transformări."""
    parts = re.findall(r"\(([^()]+)\)", str(product_text))
    return parts[-1].strip() if parts else None

# ------------ UI ------------
colA, colB = st.columns(2)
mode_backfill = st.toggle("Permit backfill (luni anterioare)", value=False)

# ==================== PROFIT (XLS) ====================
# -- PROFIT UPLOAD -----------------------------------------------------------
with colA:
    st.subheader("Încarcă Profit pe produs (Excel .XLS)")
    overwrite_profit = st.toggle(
        "Suprascrie luna dacă există (PROFIT)", 
        value=False, 
        help="Șterge din staging perioada înainte de încărcare"
    )
    f_profit = st.file_uploader("Alege fișier PROFIT", type=["xls"], key="up_profit")
    if f_profit is not None:
        sheet = read_xls_return_df(f_profit)
        period_month = first_day_from_period_line(sheet.iloc[4, 0])

        # dacă bifezi overwrite -> curăță staging + file_registry (tip profit)
        if overwrite_profit:
            st.info(f"Șterg staging PROFIT pentru {period_month}…")
            sb.rpc("reset_staging_for_period", {
                "p_period": str(period_month), 
                "p_type": "profit"
            }).execute()


            # Date: încep după headerul din rândul 11 -> de la rândul 12 (index 11)
            # Indici (0-based): B=1 (prod), E=4 (net_sales), F=5 (cogs)
            data = sheet.iloc[11:, [1, 4, 5]].copy()
            data.columns = ["product_text", "net_sales", "cogs"]
            data = data[~data["product_text"].isna()]
        except Exception as e:
            st.error(f"Nu pot citi Excel-ul: {e}")
            st.stop()

        # Restricție: doar ultima lună încheiată, dacă backfill e OFF
        if not mode_backfill and period_month != last_closed_month():
            st.error(f"Perioada din fișier este {period_month}, dar nu este ultima lună încheiată. Upload respins.")
            st.stop()

        # SKU din coloana B (ultima secvență între paranteze)
        data["sku"] = data["product_text"].apply(extract_sku_from_b)

        # ⚠️ ignorăm rândurile fără SKU
        before = len(data)
        data = data[~data["sku"].isna()].copy()
        dropped = before - len(data)
        if dropped > 0:
            st.warning(f"Atenție: {dropped} rânduri fără SKU au fost ignorate (coloana B fără paranteze).")

        # Numerice (profit): virgulă la mii, punct la zecimale
        data["net_sales"] = data["net_sales"].apply(parse_money_comma_thousands).round(2)
        data["cogs"]      = data["cogs"].apply(parse_money_comma_thousands).round(2)

        # Insert în staging + audit
        rows = [{
            "period_month": period_month.isoformat(),
            "product_text": str(r.product_text).strip(),
            "sku": r.sku,
            "net_sales_wo_vat": float(r.net_sales),
            "cogs_wo_vat": float(r.cogs),
            "source_path": f_profit.name
        } for r in data.itertuples(index=False)]

        if rows:
            sb.schema("staging").table("profit_produs").insert(rows).execute()
            sb.schema("ops").table("file_registry").insert({
                "file_type": "profit",
                "period_month": period_month.isoformat(),
                "source_path": f_profit.name,
                "rows_total": len(rows),
                "rows_loaded": len(rows),
                "status": "loaded_ok"
            }).execute()
            st.success(f"Profit încărcat pentru {period_month}. Rânduri: {len(rows)}")

# ==================== MISCĂRI (XLS) ====================
with colB:
    st.subheader("Încarcă Mișcări stocuri (Excel .XLS)")
    overwrite_misc = st.toggle(
        "Suprascrie luna dacă există (MISCĂRI)", 
        value=False, 
        help="Șterge din staging perioada înainte de încărcare"
    )
    f_misc = st.file_uploader("Alege fișier MISCĂRI", type=["xls"], key="up_misc")
    if f_misc is not None:
        sheet2 = read_xls_return_df(f_misc)
        period_month2 = first_day_from_period_line(sheet2.iloc[4, 0])

        if overwrite_misc:
            st.info(f"Șterg staging MISCĂRI pentru {period_month2}…")
            sb.rpc("reset_staging_for_period", {
                "p_period": str(period_month2), 
                "p_type": "miscari"
            }).execute()
            # Perioada: rândul 5 (index 4), col. A (index 0)
            period_month2 = first_day_from_period_line(sheet2.iloc[4, 0])

            # Date: după headerul din rândul 10 -> de la rândul 11 (index 10)
            # Indici: B=1 Produs, C=2 Cod, E=4,F=5,G=6,H=7, I=8,J=9,K=10,L=11
            cols_idx = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11]
            data2 = sheet2.iloc[10:, cols_idx].copy()
            data2.columns = [
                "product_text", "sku",
                "qty_open", "qty_in", "qty_out", "qty_close",
                "val_open", "val_in", "val_out", "val_close"
            ]
            data2 = data2[~data2["sku"].isna()]
            data2["product_text"] = data2["product_text"].astype(str).str.strip()
            data2["sku"] = data2["sku"].astype(str).str.strip()
        except Exception as e:
            st.error(f"Nu pot citi Excel-ul: {e}")
            st.stop()

        if not mode_backfill and period_month2 != last_closed_month():
            st.error(f"Perioada din fișier este {period_month2}, dar nu este ultima lună încheiată. Upload respins.")
            st.stop()

        # Conversii cantități și valori
        for c in ["qty_open", "qty_in", "qty_out", "qty_close"]:
            data2[c] = data2[c].apply(to_int0)
        for c in ["val_open", "val_in", "val_out", "val_close"]:
            data2[c] = data2[c].apply(parse_money_plain).round(2)

        # Validări
        if (data2["qty_open"] < 0).any() or (data2["qty_in"] < 0).any() or (data2["qty_close"] < 0).any():
            st.error("qty_open/qty_in/qty_close nu pot fi negative.")
            st.stop()
        if (data2["val_open"] < 0).any() or (data2["val_in"] < 0).any() or (data2["val_close"] < 0).any():
            st.error("val_open/val_in/val_close nu pot fi negative.")
            st.stop()

        # Balanțe cu toleranță ±5
        mis_qty = (data2["qty_open"] + data2["qty_in"] - data2["qty_out"] - data2["qty_close"]).abs() > 5
        mis_val = (data2["val_open"] + data2["val_in"] - data2["val_out"] - data2["val_close"]).abs() > 5
        bad = data2[mis_qty | mis_val]
        if not bad.empty:
            st.error(f"Balanțe neînchise (toleranță ±5) pe {len(bad)} rânduri. Upload respins.")
            st.stop()

        # Insert în staging + audit
        rows2 = [{
            "period_month": period_month2.isoformat(),
            "product_text": r.product_text,
            "sku": r.sku,
            "qty_open": int(r.qty_open),
            "qty_in": int(r.qty_in),
            "qty_out": int(r.qty_out),
            "qty_close": int(r.qty_close),
            "val_open": float(r.val_open),
            "val_in": float(r.val_in),
            "val_out": float(r.val_out),
            "val_close": float(r.val_close),
            "source_path": f_misc.name
        } for r in data2.itertuples(index=False)]

        if rows2:
            sb.schema("staging").table("miscari_stocuri").insert(rows2).execute()
            sb.schema("ops").table("file_registry").insert({
                "file_type": "miscari",
                "period_month": period_month2.isoformat(),
                "source_path": f_misc.name,
                "rows_total": len(rows2),
                "rows_loaded": len(rows2),
                "status": "loaded_ok"
            }).execute()
            st.success(f"Mișcări încărcate pentru {period_month2}. Rânduri: {len(rows2)}")

st.divider()

# ==================== STATUS & CONSOLIDARE ====================
st.subheader("Consolidare")

allow_recon = st.toggle(
    "Permite reconsolidare (overwrite)", 
    value=False,
    help="Afișează și lunile deja consolidate și permite reconsolidarea"
)

if allow_recon:
    # arată toate lunile unde ambele seturi sunt încărcate
    q = sb.schema("ops").table("period_registry") \
         .select("period_month, profit_loaded, miscari_loaded") \
         .eq("profit_loaded", True).eq("miscari_loaded", True) \
         .order("period_month", desc=True).execute()
    ready_list = [r["period_month"] for r in (q.data or [])]
else:
    # păstrează lista strictă (view-ul existent)
    ready = sb.table("v_ready_for_consolidation").select("*").execute()
    ready_list = [r["period_month"] for r in (ready.data or [])]

if not ready_list:
    st.info("Nicio lună pregătită (sau deja consolidată).")
else:
    sel = st.selectbox("Alege luna de consolidat", [str(x) for x in ready_list])
    btn_txt = "Reconsolidează luna selectată (overwrite)" if allow_recon else "Consolidează luna selectată"

    if st.button(btn_txt, type="primary", use_container_width=False):
        sb.rpc("consolidate_month", {"p_period": sel}).execute()
        st.success(f"Consolidare finalizată pentru {sel}.")
        st.experimental_rerun()

# ==================== RAPOARTE ====================
st.subheader("Rapoarte")
c1, c2 = st.columns(2)

with c1:
    st.markdown("**Top sellers YTD**")
    r1 = sb.table("top_sellers_ytd").select("*").execute()
    st.dataframe(pd.DataFrame(r1.data or []), use_container_width=True)

    st.markdown("**Rolling 90**")
    r2 = sb.table("sales_rolling_90").select("*").execute()
    st.dataframe(pd.DataFrame(r2.data or []), use_container_width=True)

with c2:
    st.markdown("**Stock health**")
    r3 = sb.table("stock_health").select("*").execute()
    st.dataframe(pd.DataFrame(r3.data or []), use_container_width=True)

    st.markdown("**Recomandări comandă**")
    r4 = sb.table("recomandari_comanda").select("*").execute()
    st.dataframe(pd.DataFrame(r4.data or []), use_container_width=True)
