# app.py
# ============================================================================
# Upload & Consolidare (Supabase + Streamlit)
# - Profit pe produs (.XLS)
# - Mișcări stocuri (.XLS)
# Mapare fixă pe indecși, fără diacritice în DB.
# ============================================================================

import os
import io
import re
from datetime import datetime, date
from typing import Tuple, List

import pandas as pd
import streamlit as st

# Supabase
from supabase import create_client, Client

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Lipsește SUPABASE_URL sau SUPABASE_KEY din environment.")
    st.stop()

sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(page_title="Upload & Consolidation", layout="wide")

# -----------------------------------------------------------------------------
# UTILS
# -----------------------------------------------------------------------------
SKU_RE = re.compile(r"\(([A-Za-z0-9\-]+)\)\s*$")


def _to_num_common(v) -> float:
    """Convertor numeric general: acceptă numeric; curăță NBSP, spații."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s or s in ("-", "—"):
        return 0.0
    s = s.replace("\u00a0", "").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


def _to_num_profit(v) -> float:
    """Profit: format cu virgulă la mii și punct la zecimale (ex: 1,234.56)."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s or s in ("-", "—"):
        return 0.0
    s = s.replace("\u00a0", "").replace(" ", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


def _extract_period_from_head(df_head10: pd.DataFrame) -> date:
    """
    Caută în primele rânduri ceva de forma:
      01/08/2025 - 31/08/2025  (acceptă și ., -)
    întoarce prima zi a lunii.
    """
    txt = " ".join(
        str(x)
        for x in df_head10.astype(str).fillna("").values.ravel().tolist()
        if x not in (None, "nan")
    )
    m = re.search(
        r"(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4}).*?(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})",
        txt,
    )
    if not m:
        raise ValueError("Nu am putut detecta perioada din primele rânduri.")
    d1 = m.group(1).replace(".", "/").replace("-", "/")
    dt = datetime.strptime(d1, "%d/%m/%Y").date()
    return date(dt.year, dt.month, 1)


# -----------------------------------------------------------------------------
# PARSER PROFIT (.XLS)
# -----------------------------------------------------------------------------
def parse_profit_xls(file_like) -> Tuple[pd.DataFrame, int]:
    """
    Raport Profit pe produs (.XLS)
    Mapping pe indecși:
      B (col=1) -> product_text; SKU = ultima secvență dintre paranteze
      E (col=4) -> net_sales_wo_vat (RON, fără TVA)
      F (col=5) -> cogs_wo_vat (RON, fără TVA)
    Primele 10 rânduri sunt meta; rândurile fără SKU sunt ignorate.
    """
    df_raw = pd.read_excel(file_like, engine="xlrd", header=None, dtype=object)
    if df_raw.shape[1] < 6:
        raise ValueError("Fișierul PROFIT pare incomplet (minim 6 coloane).")

    period_month = _extract_period_from_head(df_raw.iloc[:10, :])

    data = df_raw.iloc[10:, :]  # după meta
    col_prod_text = data.iloc[:, 1]  # B
    col_net = data.iloc[:, 4]  # E
    col_cogs = data.iloc[:, 5]  # F

    rows = []
    ignored = 0
    for prod_text, v_net, v_cogs in zip(col_prod_text, col_net, col_cogs):
        if (prod_text is None) or (isinstance(prod_text, float) and pd.isna(prod_text)):
            continue
        prod_s = str(prod_text).strip()
        m = SKU_RE.search(prod_s)
        if not m:
            ignored += 1
            continue
        sku = m.group(1).strip()
        net = _to_num_profit(v_net)
        cogs = _to_num_profit(v_cogs)
        rows.append(
            {
                "period_month": period_month,
                "product_text": prod_s,
                "sku": sku,
                "net_sales_wo_vat": round(net, 2),
                "cogs_wo_vat": round(cogs, 2),
            }
        )

    df = pd.DataFrame(
        rows,
        columns=["period_month", "product_text", "sku", "net_sales_wo_vat", "cogs_wo_vat"],
    )
    return df, ignored


# -----------------------------------------------------------------------------
# PARSER MISCĂRI (.XLS)
# -----------------------------------------------------------------------------
def parse_miscari_xls(file_like) -> Tuple[pd.DataFrame, List[str]]:
    """
    Raport Mișcări stocuri (.XLS)
    Mapping pe indecși:
      C (col=2) -> sku (exact)
      E/F/G/H (4..7) -> qty_open, qty_in, qty_out, qty_close
      I/J/K/L (8..11) -> val_open, val_in, val_out, val_close
    Primele 9 rânduri sunt meta; antetul e pe rândul 10 (noi folosim indecși).
    """
    df_raw = pd.read_excel(file_like, engine="xlrd", header=None, dtype=object)
    if df_raw.shape[1] < 12:
        raise ValueError("Fișierul MISCĂRI pare incomplet (minim 12 coloane).")

    period_month = _extract_period_from_head(df_raw.iloc[:10, :])
    data = df_raw.iloc[10:, :]  # după meta

    col_prod_text = data.iloc[:, 1]  # B (nefolosit în core, doar audit)
    col_sku = data.iloc[:, 2]  # C
    q_open = data.iloc[:, 4]  # E
    q_in = data.iloc[:, 5]  # F
    q_out = data.iloc[:, 6]  # G (poate fi negativ pentru retururi)
    q_close = data.iloc[:, 7]  # H
    v_open = data.iloc[:, 8]  # I
    v_in = data.iloc[:, 9]  # J
    v_out = data.iloc[:, 10]  # K (poate fi negativ pentru retururi)
    v_close = data.iloc[:, 11]  # L

    rows = []
    errors = []

    for idx, (p_text, sku, qo, qi, qo_out, qc, vo, vi, vo_out, vc) in enumerate(
        zip(col_prod_text, col_sku, q_open, q_in, q_out, q_close, v_open, v_in, v_out, v_close),
        start=11,
    ):
        if sku is None or (isinstance(sku, float) and pd.isna(sku)):
            continue
        sku = str(sku).strip()

        qty_open = int(round(_to_num_common(qo)))
        qty_in = int(round(_to_num_common(qi)))
        qty_out = int(round(_to_num_common(qo_out)))  # poate fi negativ
        qty_close = int(round(_to_num_common(qc)))

        val_open = round(_to_num_common(vo), 2)
        val_in = round(_to_num_common(vi), 2)
        val_out = round(_to_num_common(vo_out), 2)  # poate fi negativ
        val_close = round(_to_num_common(vc), 2)

        # Validări conform regulilor
        if qty_open < 0 or qty_in < 0 or qty_close < 0:
            errors.append(f"Rând {idx}: cantități negative nepermise (open/in/close).")
            continue

        # Balanță cu toleranță ±5 (buc) și ±5 lei
        if abs((qty_open + qty_in) - qty_out - qty_close) > 5:
            errors.append(f"Rând {idx} SKU {sku}: balanta cantități nu se închide.")
            continue
        if abs((val_open + val_in) - val_out - val_close) > 5:
            errors.append(f"Rând {idx} SKU {sku}: balanta valori nu se închide.")
            continue

        rows.append(
            {
                "period_month": period_month,
                "product_text": ("" if p_text is None else str(p_text).strip()),
                "sku": sku,
                "qty_open": qty_open,
                "qty_in": qty_in,
                "qty_out": qty_out,
                "qty_close": qty_close,
                "val_open": val_open,
                "val_in": val_in,
                "val_out": val_out,
                "val_close": val_close,
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "period_month",
            "product_text",
            "sku",
            "qty_open",
            "qty_in",
            "qty_out",
            "qty_close",
            "val_open",
            "val_in",
            "val_out",
            "val_close",
        ],
    )
    return df, errors


# -----------------------------------------------------------------------------
# DB HELPERS
# -----------------------------------------------------------------------------
def insert_rows(table: str, rows: List[dict], schema: str = "staging") -> int:
    """Insert bulk în staging.<table>"""
    if not rows:
        return 0
    res = sb.table(table, table_schema=schema).insert(rows).execute()
    return len(res.data) if res.data else 0


def delete_staging_for_period(table: str, period: date, schema: str = "staging"):
    sb.table(table, table_schema=schema).delete().eq("period_month", period).execute()


def register_file(period: date, file_type: str, file_name: str):
    sb.table("file_registry", table_schema="ops").insert(
        {
            "period_month": str(period),
            "file_type": file_type,
            "file_name": file_name,
        }
    ).execute()


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("Upload & Consolidation")

colL, colR = st.columns(2, gap="large")

# ------------------------- PROFIT -------------------------------------------
with colL:
    st.subheader("Încarcă Profit pe produs (Excel .XLS)")
    up_profit = st.file_uploader(
        "Alege fișier PROFIT", type=["xls"], accept_multiple_files=False, key="upl_profit"
    )
    backfill = st.toggle("Permite backfill (luni anterioare)", value=False, help="Permite încărcarea oricărei luni anterioare.")

    if up_profit is not None:
        try:
            df_profit, ignored = parse_profit_xls(io.BytesIO(up_profit.read()))
            period = df_profit["period_month"].iloc[0] if not df_profit.empty else None

            if period and (period >= date.today().replace(day=1) or (not backfill and period < date.today().replace(day=1))):
                # regulă: accepți doar ultima lună încheiată dacă backfill = False
                last_closed = (date.today().replace(day=1) - pd.offsets.MonthBegin(1)).date()  # prima zi luna trecută
                if period != last_closed and not backfill:
                    st.error(f"Se acceptă doar ultima lună încheiată ({last_closed}).")
                    st.stop()

            # overwrite în staging
            delete_staging_for_period("profit_produs", df_profit["period_month"].iloc[0])
            inserted = insert_rows("profit_produs", df_profit.to_dict(orient="records"))
            register_file(df_profit["period_month"].iloc[0], "profit", up_profit.name)

            if ignored:
                st.warning(f"Atenție: {ignored} rânduri fără SKU au fost ignorate (coloana B fără paranteze).")
            st.success(f"Profit încărcat pentru {df_profit['period_month'].iloc[0]}. Rânduri: {inserted}")
        except Exception as e:
            st.error(f"Nu am putut procesa fișierul PROFIT: {e}")

# ------------------------- MISCĂRI -----------------------------------------
with colR:
    st.subheader("Încarcă Mișcări stocuri (Excel .XLS)")
    up_misc = st.file_uploader(
        "Alege fișier MISCARE", type=["xls"], accept_multiple_files=False, key="upl_misc"
    )

    if up_misc is not None:
        try:
            df_misc, errs = parse_miscari_xls(io.BytesIO(up_misc.read()))
            if errs:
                with st.expander("Erori de validare găsite", expanded=False):
                    for e in errs:
                        st.write("• ", e)
                if any("balanta" in e for e in errs):
                    st.error("Există erori de balanță; încarcă un fișier corect.")
                    st.stop()

            delete_staging_for_period("miscari_stocuri", df_misc["period_month"].iloc[0])
            inserted = insert_rows("miscari_stocuri", df_misc.to_dict(orient="records"))
            register_file(df_misc["period_month"].iloc[0], "miscari", up_misc.name)
            st.success(f"Mișcări încărcate pentru {df_misc['period_month'].iloc[0]}. Rânduri: {inserted}")
        except Exception as e:
            st.error(f"Nu am putut procesa fișierul MISCARE: {e}")

st.divider()

# ------------------------- STATUS LUNI --------------------------------------
st.subheader("Status luni")
try:
    status = sb.table("v_period_status", table_schema="ops").select("*").execute()
    st.dataframe(pd.DataFrame(status.data or []), use_container_width=True, hide_index=True)
except Exception as e:
    st.warning(f"Nu am putut citi statusul lunilor: {e}")

# ------------------------- CONSOLIDARE --------------------------------------
st.subheader("Consolidare")
allow_recon = st.toggle("Permite reconsolidare (overwrite)", value=False, help="Șterge & rescrie luna selectată în core/mart.")
# lunile disponibile
period_opts = (
    sb.table("period_registry", table_schema="ops")
    .select("period_month")
    .order("period_month")
    .execute()
)
period_values = [r["period_month"] for r in (period_opts.data or [])]
sel = st.selectbox("Alege luna de consolidat", options=period_values, index=len(period_values) - 1 if period_values else 0)

btn_txt = "Reconsolidează luna selectată (overwrite)" if allow_recon else "Consolidează luna selectată"
if st.button(btn_txt, type="primary"):
    try:
        # dacă ai funcția consolidate_month care știe overwrite, altfel rulează separat
        sb.rpc("consolidate_month", {"p_period": sel}).execute()
        st.success(f"Consolidare finalizată pentru {sel}.")
        st.rerun()
    except Exception as e:
        st.error(f"Eroare la consolidare: {e}")

st.divider()

# ------------------------- RAPOARTE -----------------------------------------
st.subheader("Rapoarte")

# Top sellers YTD
try:
    r1 = sb.table("top_sellers_ytd", table_schema="public").select("*").execute()
    df1 = pd.DataFrame(r1.data or [])
    st.markdown("**Top sellers YTD**")
    st.dataframe(df1, use_container_width=True, hide_index=True)
except Exception as e:
    st.warning(f"Nu am putut citi top_sellers_ytd: {e}")

# Rolling 90 (opțional, dacă ai view)
try:
    r2 = sb.table("sales_monthly", table_schema="public").select("*").gte(
        "perioada_luna", (date.today() - pd.Timedelta(days=90)).strftime("%Y-%m-01")
    ).execute()
    df2 = pd.DataFrame(r2.data or [])
    if not df2.empty:
        st.markdown("**Rolling 90 (din sales_monthly)**")
        st.dataframe(df2, use_container_width=True, hide_index=True)
except Exception as e:
    pass

# Recomandări comandă (dacă ai view dedicat)
try:
    r3 = sb.table("recomandari_comanda", table_schema="mart").select("*").execute()
    df3 = pd.DataFrame(r3.data or [])
    if not df3.empty:
        st.markdown("**Recomandări comandă**")
        st.dataframe(df3, use_container_width=True, hide_index=True)
except Exception:
    pass
