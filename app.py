import os
import io
import re
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st
from supabase import create_client

# =============================================================================
# Config UI
# =============================================================================
st.set_page_config(page_title="ServicePack Reports", layout="wide")
st.title("📊 ServicePack Reports")

# =============================================================================
# Conexiune Supabase
# =============================================================================
DEFAULT_SUPABASE_URL = "https://ufpcludvtpfdyjvqjgje.supabase.co"
DEFAULT_SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

SUPABASE_URL = os.getenv("SUPABASE_URL", DEFAULT_SUPABASE_URL)
SUPABASE_KEY = os.getenv("SUPABASE_KEY", DEFAULT_SUPABASE_KEY)

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Lipsesc variabilele **SUPABASE_URL** și/sau **SUPABASE_KEY** (Settings → Secrets).")
    st.stop()

try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    sb = supabase
    st.success("Conexiune la Supabase OK.")
except Exception as e:
    st.error(f"Eroare conectare Supabase: {e}")
    st.stop()

# =============================================================================
# Helpers business
# =============================================================================
TZ = ZoneInfo("Europe/Bucharest")

def last_completed_month(today: datetime) -> datetime.date:
    """prima zi a lunii precedente în TZ RO"""
    first_of_this_month = datetime(today.year, today.month, 1, tzinfo=TZ).date()
    prev_month_last_day = pd.Timestamp(first_of_this_month) - pd.Timedelta(days=1)
    return prev_month_last_day.date().replace(day=1)

now_ro = datetime.now(TZ)
lcm = last_completed_month(now_ro)
st.info(f"🗓️ Ultima lună încheiată (pentru încărcare standard): **{lcm.strftime('%Y-%m')}**")

# =============================================================================
# Utilitare parsing numerice & headere XLS
# =============================================================================
NUMERIC_RE_THOUSAND_DOT_DECIMAL_COMMA = re.compile(r"^\d{1,3}(\.\d{3})+(,\d+)?$")
NUMERIC_RE_THOUSAND_COMMA_DECIMAL_DOT = re.compile(r"^\d{1,3}(,\d{3})+(\.\d+)?$")

def to_number(val):
    """Tolerant: 1.021,02 | 1,021.02 | 104,04 | 104.04 | 104"""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if s == "":
        return None
    if NUMERIC_RE_THOUSAND_DOT_DECIMAL_COMMA.match(s):
        return float(s.replace(".", "").replace(",", "."))
    if NUMERIC_RE_THOUSAND_COMMA_DECIMAL_DOT.match(s):
        return float(s.replace(",", ""))
    if "," in s and "." not in s:
        return float(s.replace(",", "."))
    try:
        return float(s)
    except Exception:
        try:
            return float(s.replace(",", "."))
        except Exception:
            return None

def normalize_cols(cols) -> dict:
    norm = {}
    for c in cols:
        key = str(c).lower().replace("\n", " ").replace("\r", " ")
        key = re.sub(r"[^a-z0-9]+", " ", key)
        key = re.sub(r"\s+", " ", key).strip()
        norm[c] = key
    return norm

def extract_period_from_header(xls_bytes: bytes, row_idx: int = 4) -> datetime.date:
    """Din rândul 5 (index 4) col A: 'Perioada: 01/08/2025 - 31/08/2025' -> prima zi a lunii"""
    head = pd.read_excel(io.BytesIO(xls_bytes), sheet_name=0, header=None, nrows=row_idx+2)
    txt = str(head.iat[row_idx, 0]) if head.shape[0] > row_idx else ""
    m = re.search(r"(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", txt or "")
    if not m:
        return last_completed_month(datetime.now(TZ))
    raw = m.group(1)
    for fmt in ("%d/%m/%Y", "%d.%m.%Y", "%d-%m-%Y", "%d/%m/%y"):
        try:
            d = datetime.strptime(raw, fmt).date()
            return d.replace(day=1)
        except Exception:
            continue
    return last_completed_month(datetime.now(TZ))

# =============================================================================
# UPLOAD & LOADERS
# =============================================================================
tab_status, tab_upload, tab_consol, tab_reports = st.tabs(
    ["🗓️ Status pe luni", "⬆️ Upload fișiere", "✅ Consolidare & Rapoarte", "📊 Rapoarte"]
)

# -------------------- TAB: Status pe luni --------------------
with tab_status:
    st.subheader("Stare pe luni (ops.period_registry)")
    try:
        # dacă schema ops.period_registry nu e expusă, folosim public.period_registry
        try:
            resp = supabase.schema("public").table("period_registry").select("*").order("period_month", desc=True).limit(24).execute()
        except Exception:
            resp = supabase.table("period_registry").select("*").order("period_month", desc=True).limit(24).execute()
        df = pd.DataFrame(resp.data or [])
        if df.empty:
            st.info("Nu există încă înregistrări în period_registry.")
        else:
            if "period_month" in df.columns:
                df["period_month"] = pd.to_datetime(df["period_month"]).dt.date
            st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.warning(f"Nu pot citi period_registry: {e}")

# -------------------- TAB: Upload fișiere --------------------
with tab_upload:
    st.subheader("📥 Import Raport profit pe produs")
    up_profit = st.file_uploader("Alege fișierul XLS/XLSX (formatul standard, rândul 11 = antet)", type=["xls", "xlsx"], key="profit_upload_v2")

    def extract_sku_from_product(text: str):
        s = str(text) if text is not None else ""
        m = re.search(r"\(([^()]+)\)\s*$", s)
        return m.group(1).strip() if m else None

    def load_profit_file_to_staging(xls_bytes: bytes):
        perioada = extract_period_from_header(xls_bytes, row_idx=4)
        df_raw = pd.read_excel(io.BytesIO(xls_bytes), sheet_name=0, header=10)
        # eliminăm coloanele complet goale
        df_raw = df_raw.loc[:, ~df_raw.columns.to_series().astype(str).str.fullmatch(r"\s*nan\s*", case=False)]
        norm = normalize_cols(df_raw.columns)

        col_prod = next((c for c in df_raw.columns if norm[c] == "produsul"), None)
        col_net  = next((c for c in df_raw.columns if norm[c].startswith("vanzari nete")), None)
        col_cogs = next((c for c in df_raw.columns if norm[c].startswith("costul bunurilor vandute")), None)

        if not col_prod or not col_net or not col_cogs:
            raise RuntimeError("Nu am găsit coloanele 'Produsul', 'Vanzari nete', 'Costul bunurilor vandute'.")

        out = pd.DataFrame({
            "perioada_luna": perioada.isoformat(),
            "sku": df_raw[col_prod].apply(extract_sku_from_product),
            "net_sales_wo_vat": df_raw[col_net].apply(to_number),
            "cogs_wo_vat": df_raw[col_cogs].apply(to_number),
        })
        out = out[out["sku"].notna()]
        out = out[(out["net_sales_wo_vat"].notna()) | (out["cogs_wo_vat"].notna())]
        if out.empty:
            return 0, 0

        out = out.where(pd.notnull(out), None)
        recs = out.to_dict(orient="records")
        BATCH = 1000
        written = 0
        for i in range(0, len(recs), BATCH):
            supabase.schema("staging").table("profit_produs").insert(recs[i:i+BATCH]).execute()
            written += len(recs[i:i+BATCH])

        return len(out), written, perioada

    if up_profit is not None:
        try:
            bytes_profit = up_profit.read()
            rows_in, rows_out, per = load_profit_file_to_staging(bytes_profit)
            st.success(f"Import PROFIT OK: parse={rows_in}, scrise={rows_out}, lună={per}.")
            qsum = (
                supabase.schema("staging").table("profit_produs")
                .select("sum(net_sales_wo_vat) as s_net, sum(cogs_wo_vat) as s_cogs")
                .eq("perioada_luna", per.isoformat()).execute()
            )
            st.caption(f"Sume pentru {per}: {qsum.data}")
        except Exception as e:
            st.error(f"Eroare la importul fișierului de profit: {e}")

    st.markdown("---")
    st.subheader("📥 Import Raport mișcări stocuri")
    up_stock = st.file_uploader("Alege fișierul XLS/XLSX (formatul standard, rândul 10 = antet)", type=["xls", "xlsx"], key="stock_upload_v2")

    def load_stock_file_to_staging(xls_bytes: bytes):
        perioada = extract_period_from_header(xls_bytes, row_idx=4)  # rândul 5
        df_raw = pd.read_excel(io.BytesIO(xls_bytes), sheet_name=0, header=9)  # rândul 10 antet
        # eliminăm coloanele complet goale
        df_raw = df_raw.loc[:, ~df_raw.columns.to_series().astype(str).str.fullmatch(r"\s*nan\s*", case=False)]

        # Unele antete sunt duplicate (Intrari/Iesiri la Cantități și la Valori). Le identificăm după poziție.
        cols = list(df_raw.columns)
        norm = normalize_cols(cols)

        # coloane obligatorii
        col_sku = next((c for c in cols if norm[c] in ("cod", "sku")), None)
        # Cantități
        col_qty_open  = next((c for c in cols if norm[c] == "stoc initial"), None)
        # prima "Intrari" după stoc initial = cantități
        try:
            idx_open = cols.index(col_qty_open)
            col_qty_in = cols[idx_open + 1]
            col_qty_out = cols[idx_open + 2]
            col_qty_close = cols[idx_open + 3]
        except Exception:
            col_qty_in = next((c for c in cols if norm[c] == "intrari"), None)
            # la nevoie folosim fallback
            col_qty_out = next((c for c in cols if norm[c] == "iesiri"), None)
            col_qty_close = next((c for c in cols if norm[c] == "stoc final"), None)

        # Valori — căutăm al doilea set (după cantități)
        # găsim "Sold initial" (valoare)
        col_val_open = None
        for c in cols:
            if norm[c] in ("sold initial", "sold initiali", "sold initial "):
                col_val_open = c
                break
        if not col_val_open:
            # poate e exact "sold initial"
            col_val_open = next((c for c in cols if norm[c].startswith("sold initial")), None)

        if not all([col_sku, col_qty_open, col_qty_in, col_qty_out, col_qty_close, col_val_open]):
            raise RuntimeError("Nu am putut detecta setul de coloane pentru mișcări (sku/qty/val).")

        # pe valori, după sold initial urmează intrari/iesiri/sold final
        try:
            idx_val_open = cols.index(col_val_open)
            col_val_in = cols[idx_val_open + 1]
            col_val_out = cols[idx_val_open + 2]
            col_val_close = cols[idx_val_open + 3]
        except Exception:
            # fallback pe nume
            col_val_in = next((c for c in cols if norm[c] == "intrari"), None)
            col_val_out = next((c for c in cols if norm[c] == "iesiri"), None)
            col_val_close = next((c for c in cols if norm[c].startswith("sold final")), None)

        out = pd.DataFrame({
            "perioada_luna": perioada.isoformat(),
            "sku": df_raw[col_sku].astype(str).str.strip(),
            "qty_open":  df_raw[col_qty_open].apply(to_number),
            "qty_in":    df_raw[col_qty_in].apply(to_number),
            "qty_out":   df_raw[col_qty_out].apply(to_number),
            "qty_close": df_raw[col_qty_close].apply(to_number),
            "val_open":  df_raw[col_val_open].apply(to_number),
            "val_in":    df_raw[col_val_in].apply(to_number),
            "val_out":   df_raw[col_val_out].apply(to_number),
            "val_close": df_raw[col_val_close].apply(to_number),
        })

        out = out[out["sku"].notna()]
        out = out[(out[["qty_open","qty_in","qty_out","qty_close","val_open","val_in","val_out","val_close"]].notna().any(axis=1))]
        if out.empty:
            return 0, 0

        out = out.where(pd.notnull(out), None)
        recs = out.to_dict(orient="records")
        BATCH = 1000
        written = 0
        for i in range(0, len(recs), BATCH):
            supabase.schema("staging").table("miscari_stocuri").insert(recs[i:i+BATCH]).execute()
            written += len(recs[i:i+BATCH])

        return len(out), written, perioada

    if up_stock is not None:
        try:
            bytes_stock = up_stock.read()
            rows_in, rows_out, per = load_stock_file_to_staging(bytes_stock)
            st.success(f"Import MIȘCĂRI OK: parse={rows_in}, scrise={rows_out}, lună={per}.")
            qsum = (
                supabase.schema("staging").table("miscari_stocuri")
                .select("sum(qty_out) as s_out, sum(val_out) as v_out")
                .eq("perioada_luna", per.isoformat()).execute()
            )
            st.caption(f"Sume pentru {per}: {qsum.data}")
        except Exception as e:
            st.error(f"Eroare la importul mișcărilor: {e}")

# -------------------- TAB: Consolidare & Rapoarte --------------------
with tab_consol:
    st.subheader(f"Consolidare pentru {lcm.strftime('%Y-%m')}")
    c1, c2, c3 = st.columns(3)
    tol_qty = c1.number_input("Toleranță cantități (buc)", min_value=0.0, value=0.01, step=0.01, format="%.2f")
    tol_val = c2.number_input("Toleranță valori (lei fără TVA)", min_value=0.0, value=5.00, step=0.50, format="%.2f")
    overwrite = c3.checkbox("Forțează overwrite luna curentă în core/mart", value=True)

    # Semnale simple: există date în staging pentru luna LCM?
    try:
        has_profit = supabase.schema("staging").table("profit_produs").select("id", count="exact").eq("perioada_luna", lcm.isoformat()).limit(1).execute().count > 0
    except Exception:
        has_profit = False
    try:
        has_moves = supabase.schema("staging").table("miscari_stocuri").select("id", count="exact").eq("perioada_luna", lcm.isoformat()).limit(1).execute().count > 0
    except Exception:
        has_moves = False

    cols = st.columns(4)
    cols[0].metric("Profit încărcat?", "DA" if has_profit else "NU")
    cols[1].metric("Mișcări încărcate?", "DA" if has_moves else "NU")

    # status balanțe (dacă view-ul există)
    delta_qty = 0.0
    delta_val = 0.0
    try:
        v = supabase.table("v_balance_summary").select("*").eq("period_month", lcm.isoformat()).limit(1).execute().data
        if v:
            delta_qty = float(v[0].get("total_diff_qty", 0) or 0)
            delta_val = float(v[0].get("total_diff_val", 0) or 0)
    except Exception:
        pass
    cols[2].metric("Balanță cantități OK?", "DA" if abs(delta_qty) <= tol_qty else "NU", f"Δ {delta_qty:.2f}")
    cols[3].metric("Balanță valori OK?", "DA" if abs(delta_val) <= tol_val else "NU", f"Δ {delta_val:.2f}")

    st.markdown("")

    do_consol = st.button("🚀 Consolidează luna", use_container_width=False, type="primary", disabled=not (has_profit and has_moves))
    if do_consol:
        try:
            # apel la funcția din DB
            res = supabase.rpc(
                "consolidate_month_tolerant",
                {"p_period": lcm.isoformat(), "p_tol_qty": tol_qty, "p_tol_val": tol_val, "p_overwrite": overwrite},
            ).execute()
            st.success("Consolidare reușită. Datele din core/mart au fost reîmprospătate.")
        except Exception as e:
            st.error(f"Eroare la consolidare: {e}")

# -------------------- TAB: Rapoarte --------------------
with tab_reports:
    st.subheader("🏆 Top sellers (fără TVA)")
    ytd_start = datetime(datetime.now(TZ).year, 1, 1, tzinfo=TZ).date()

    c1, c2, c3, c4 = st.columns([1.4, 1.2, 1.2, 1])
    mode   = c1.selectbox("Interval", ["YTD (an curent)", "Ultimele 3 luni", "Ultimele 6 luni", "Custom"], index=0)
    top_n  = int(c2.number_input("Top N", min_value=5, max_value=300, value=50, step=5))
    sort_by = c3.selectbox("Sortează după", ["Vânzări nete (lei)", "Profit (lei)", "Cantități vândute (buc)", "Marjă %", "Preț mediu/unitate"], index=0)
    asc    = c4.checkbox("Crescător", value=False)

    if mode == "YTD (an curent)":
        start_date, end_date = ytd_start, lcm
    elif mode == "Ultimele 3 luni":
        start_date, end_date = (pd.Timestamp(lcm) - pd.offsets.MonthBegin(3)).date(), lcm
    elif mode == "Ultimele 6 luni":
        start_date, end_date = (pd.Timestamp(lcm) - pd.offsets.MonthBegin(6)).date(), lcm
    else:
        cc1, cc2 = st.columns(2)
        start_date = cc1.date_input("De la luna", ytd_start)
        end_date   = cc2.date_input("Până la luna", lcm)

    try:
        resp = sb.table("v_sales_monthly") \
                 .select("period_month, sku, net_sales_wo_vat, cogs_wo_vat, profit_wo_vat, margin_pct, qty_sold") \
                 .gte("period_month", start_date.isoformat()) \
                 .lte("period_month", end_date.isoformat()) \
                 .execute()
        df = pd.DataFrame(resp.data or [])
        if df.empty:
            st.info("Nu există date pentru intervalul ales.")
        else:
            # conversii defensive
            for c in ["net_sales_wo_vat", "cogs_wo_vat", "profit_wo_vat", "margin_pct", "qty_sold"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            ag = (
                df.groupby("sku", as_index=False)
                  .agg(
                      net_sales_wo_vat=("net_sales_wo_vat", "sum"),
                      cogs_wo_vat=("cogs_wo_vat", "sum"),
                      profit_wo_vat=("profit_wo_vat", "sum"),
                      qty_sold=("qty_sold", "sum"),
                  )
            )
            ag["margin_pct"] = (ag["profit_wo_vat"] / ag["net_sales_wo_vat"] * 100).where(ag["net_sales_wo_vat"] != 0)
            ag["price_per_unit"] = (ag["net_sales_wo_vat"] / ag["qty_sold"]).where(ag["qty_sold"] > 0)

            total_sales = float(ag["net_sales_wo_vat"].sum())
            total_qty   = float(ag["qty_sold"].sum())
            ag["sales_share_%"] = (ag["net_sales_wo_vat"] / total_sales * 100).round(2).fillna(0)

            sort_map = {
                "Vânzări nete (lei)"      : "net_sales_wo_vat",
                "Profit (lei)"            : "profit_wo_vat",
                "Cantități vândute (buc)" : "qty_sold",
                "Marjă %"                 : "margin_pct",
                "Preț mediu/unitate"      : "price_per_unit",
            }
            ag = ag.sort_values(sort_map[sort_by], ascending=asc).head(top_n)

            out = ag.rename(columns={
                "sku": "SKU",
                "net_sales_wo_vat": "Vânzări nete (lei)",
                "cogs_wo_vat": "Cost achiziție (lei)",
                "profit_wo_vat": "Profit (lei)",
                "margin_pct": "Marjă %",
                "qty_sold": "Cantități vândute (buc)",
                "price_per_unit": "Preț mediu / unitate (lei)",
                "sales_share_%": "Pondere vânzări %",
            })

            st.caption(
                f"Interval: **{start_date} → {end_date}** • Rânduri: {len(out)} • "
                f"Total vânzări: {total_sales:,.2f} lei • Total buc: {total_qty:,.0f}"
            )
            st.dataframe(out, use_container_width=True)

            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Export CSV", data=csv,
                               file_name=f"top_sellers_{start_date}_{end_date}.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Eroare la citire v_sales_monthly: {e}")
