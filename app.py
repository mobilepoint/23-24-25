import re
import io
import datetime as dt
import pandas as pd
import streamlit as st
from supabase import create_client, Client
def read_excel_file(uploaded_file):
    """
    Citeste fisiere Excel (XLS sau XLSX) din upload Streamlit.
    """
    content = uploaded_file.read()
    # engine="xlrd" e necesar pentru fisiere XLS (format vechi)
    df = pd.read_excel(io.BytesIO(content), engine="xlrd")
    return df

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(page_title="Monthly Consolidation", layout="wide")

# ---------- util ----------

def first_day_from_period_line(line_text: str) -> dt.date:
    # asteptat: "Perioada: 01/01/2023 - 31/01/2023"
    m = re.search(r"(\d{2})/(\d{2})/(\d{4})\s*-\s*(\d{2})/(\d{2})/(\d{4})", line_text)
    if not m:
        raise ValueError("Nu pot extrage perioada din randul 5.")
    d1, m1, y1 = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return dt.date(y1, m1, 1)

def last_closed_month(today=None) -> dt.date:
    today = today or dt.date.today()
    first_this_month = dt.date(today.year, today.month, 1)
    prev_month_last_day = first_this_month - dt.timedelta(days=1)
    return dt.date(prev_month_last_day.year, prev_month_last_day.month, 1)

def parse_money_comma_thousands(x):
    # format "1,234.56" sau "0" sau "", returneaza float
    if pd.isna(x) or x == "":
        return 0.0
    s = str(x).replace(",", "")
    return float(s)

def parse_money_plain(x):
    # format "1234.56" fara separator mii
    if pd.isna(x) or x == "":
        return 0.0
    return float(x)

def to_int0(x):
    if pd.isna(x) or x == "":
        return 0
    return int(round(float(x)))

SKU_RE = re.compile(r"\(([^()]*)\)\s*$")  # ultimul text dintre paranteze la finalul denumirii

# ---------- UI ----------

st.title("Upload & Consolidation")

colA, colB = st.columns(2)
mode_backfill = st.toggle("Permit backfill (luni anterioare)", value=False, help="Daca e OFF, se accepta doar ultima luna incheiata.")

# ------- bloc: PROFIT --------
with colA:
    st.subheader("Incarca Profit pe produs (Excel)")
    f_profit = st.file_uploader("Alege fisier PROFIT", type=["xlsx", "xls"], key="up_profit")
    if f_profit is not None:
        try:
            bio = io.BytesIO(f_profit.read())
            # header pe randul 11 -> 0-based index 10
            df = pd.read_excel(bio, engine="openpyxl", header=10)
            # randul 5 pentru perioada -> 0-based index 4, coloana A sau intregul rand concat
            meta = pd.read_excel(bio, engine="openpyxl", header=None, nrows=6)
        except Exception as e:
            st.error(f"Nu pot citi Excel-ul: {e}")
            st.stop()

        period_line = str(meta.iloc[4, 0])
        period_month = first_day_from_period_line(period_line)

        # restrictie: doar ultima luna inchisa daca nu e backfill
        if not mode_backfill and period_month != last_closed_month():
            st.error(f"Perioada din fisier este {period_month}, dar nu este ultima luna incheiata. Upload respins.")
            st.stop()

        # mapare coloane: B=Produsul, E=Vanzari nete (fara TVA), F=Cost bunuri vandute
        if "Produsul" not in df.columns:
            st.error("Nu gasesc coloana 'Produsul' pe randul de header 11.")
            st.stop()

        df = df[["Produsul", "Vanzari nete", "Costul bunurilor vandute"]].rename(
            columns={"Produsul":"product_text", "Vanzari nete":"net_sales", "Costul bunurilor vandute":"cogs"}
        )
        df = df.dropna(subset=["product_text"])  # ignora randuri goale

        # extrage sku din paranteze (obligatoriu)
        def extract_sku(name: str):
            m = SKU_RE.search(str(name).strip())
            if not m:
                return None
            return m.group(1).strip()

        df["sku"] = df["product_text"].apply(extract_sku)
        bad_rows = df["sku"].isna().sum()
        if bad_rows > 0:
            st.error(f"{bad_rows} rand(uri) fara SKU intre paranteze. Upload respins.")
            st.stop()

        # parsari numerice: format cu virgula la mii, punct la zecimale
        df["net_sales"] = df["net_sales"].apply(parse_money_comma_thousands).round(2)
        df["cogs"] = df["cogs"].apply(parse_money_comma_thousands).round(2)

        # pregateste inserturi
        rows = []
        for _, r in df.iterrows():
            rows.append({
                "period_month": period_month.isoformat(),
                "product_text": str(r["product_text"]).strip(),
                "sku": r["sku"],
                "net_sales_wo_vat": float(r["net_sales"]),
                "cogs_wo_vat": float(r["cogs"]),
                "source_path": f_profit.name
            })

        # scrie in staging
        if rows:
            res = sb.table("profit_produs").insert(rows, table_schema="staging").execute()
            # audit in file_registry
            sb.table("file_registry").insert({
                "file_type":"profit",
                "period_month": period_month.isoformat(),
                "source_path": f_profit.name,
                "rows_total": len(rows),
                "rows_loaded": len(rows),
                "status":"loaded_ok"
            }, table_schema="ops").execute()

            st.success(f"Profit incarcat pentru {period_month}. Randuri: {len(rows)}")

# ------- bloc: MISCARI --------
with colB:
    st.subheader("Incarca Miscari stocuri (Excel)")
    f_misc = st.file_uploader("Alege fisier MISCARE", type=["xlsx", "xls"], key="up_misc")
    if f_misc is not None:
        try:
            bio2 = io.BytesIO(f_misc.read())
            # header pe randul 10 -> 0-based index 9
            dfm = pd.read_excel(bio2, engine="openpyxl", header=9)
            meta2 = pd.read_excel(bio2, engine="openpyxl", header=None, nrows=6)
        except Exception as e:
            st.error(f"Nu pot citi Excel-ul: {e}")
            st.stop()

        period_line2 = str(meta2.iloc[4, 0])
        period_month2 = first_day_from_period_line(period_line2)

        if not mode_backfill and period_month2 != last_closed_month():
            st.error(f"Perioada din fisier este {period_month2}, dar nu este ultima luna incheiata. Upload respins.")
            st.stop()

        needed = ["Produs", "Cod", "Stoc initial", "Intrari", "Iesiri", "Stoc final",
                  "Sold initial", "Intrari.1", "Iesiri.1", "Sold final"]
        missing = [c for c in needed if c not in dfm.columns]
        if missing:
            st.error(f"Lipsesc coloane in headerul de pe randul 10: {missing}")
            st.stop()

        dm = pd.DataFrame({
            "product_text": dfm["Produs"].astype(str).str.strip(),
            "sku": dfm["Cod"].astype(str).str.strip(),
            "qty_open": dfm["Stoc initial"].apply(to_int0),
            "qty_in": dfm["Intrari"].apply(to_int0),
            "qty_out": dfm["Iesiri"].apply(to_int0),
            "qty_close": dfm["Stoc final"].apply(to_int0),
            "val_open": dfm["Sold initial"].apply(parse_money_plain).round(2),
            "val_in": dfm["Intrari.1"].apply(parse_money_plain).round(2),
            "val_out": dfm["Iesiri.1"].apply(parse_money_plain).round(2),
            "val_close": dfm["Sold final"].apply(parse_money_plain).round(2)
        })
        dm = dm[dm["sku"] != ""]  # ignora randuri goale

        # validari cheie conform regulilor B
        if (dm["qty_open"] < 0).any() or (dm["qty_in"] < 0).any() or (dm["qty_close"] < 0).any():
            st.error("qty_open/qty_in/qty_close nu pot fi negative.")
            st.stop()
        if (dm["val_open"] < 0).any() or (dm["val_in"] < 0).any() or (dm["val_close"] < 0).any():
            st.error("val_open/val_in/val_close nu pot fi negative.")
            st.stop()

        # toleranta ±5
        mis_qty = (dm["qty_open"] + dm["qty_in"] - dm["qty_out"] - dm["qty_close"]).abs() > 5
        mis_val = (dm["val_open"] + dm["val_in"] - dm["val_out"] - dm["val_close"]).abs() > 5
        bad = dm[mis_qty | mis_val]
        if not bad.empty:
            st.error(f"Balante neinchise (toleranta ±5) pe {len(bad)} randuri. Upload respins.")
            st.stop()

        rows2 = []
        for _, r in dm.iterrows():
            rows2.append({
                "period_month": period_month2.isoformat(),
                "product_text": r["product_text"],
                "sku": r["sku"],
                "qty_open": int(r["qty_open"]),
                "qty_in": int(r["qty_in"]),
                "qty_out": int(r["qty_out"]),
                "qty_close": int(r["qty_close"]),
                "val_open": float(r["val_open"]),
                "val_in": float(r["val_in"]),
                "val_out": float(r["val_out"]),
                "val_close": float(r["val_close"]),
                "source_path": f_misc.name
            })

        if rows2:
            sb.table("miscari_stocuri").insert(rows2, table_schema="staging").execute()
            sb.table("file_registry").insert({
                "file_type":"miscari",
                "period_month": period_month2.isoformat(),
                "source_path": f_misc.name,
                "rows_total": len(rows2),
                "rows_loaded": len(rows2),
                "status":"loaded_ok"
            }, table_schema="ops").execute()
            st.success(f"Miscari incarcate pentru {period_month2}. Randuri: {len(rows2)}")

st.divider()

# ------- status si consolidare -------
st.subheader("Status luni")
status = sb.table("v_period_status").select("*").execute()
df_status = pd.DataFrame(status.data or [])
st.dataframe(df_status, use_container_width=True)

# pregatite pentru consolidare (ambele fisiere incarcate si nu e consolidata)
ready  = sb.table("v_ready_for_consolidation").select("*").execute()
df_ready = pd.DataFrame(ready.data or [])

st.subheader("Consolidare")
if df_ready.empty:
    st.info("Nicio luna pregatita (sau deja consolidata).")
else:
    sel = st.selectbox("Alege luna de consolidat", df_ready["period_month"].astype(str).tolist())
    if st.button("Consolideaza luna selectata", type="primary"):
        resp = sb.rpc("consolidate_month", {"p_period": sel}).execute()
        st.success(f"Consolidare finalizata pentru {sel}.")
        st.experimental_rerun()

st.divider()

# ------- Rapoarte --------
st.subheader("Rapoarte")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Top sellers YTD**")
    r1 = sb.table("top_sellers_ytd").select("*").execute()
    st.dataframe(pd.DataFrame(r1.data or []), use_container_width=True)

    st.markdown("**Rolling 90**")
    r2 = sb.table("sales_rolling_90").select("*").execute()
    st.dataframe(pd.DataFrame(r2.data or []), use_container_width=True)

with col2:
    st.markdown("**Stock health**")
    r3 = sb.table("stock_health").select("*").execute()
    st.dataframe(pd.DataFrame(r3.data or []), use_container_width=True)

    st.markdown("**Recomandari comanda**")
    r4 = sb.table("recomandari_comanda").select("*").execute()
    st.dataframe(pd.DataFrame(r4.data or []), use_container_width=True)
