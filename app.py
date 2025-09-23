# ---- APP.PY (ServicePack Reports - Mapping Strict, compat Python 3.8+) ----
import os
import io
import re
import pandas as pd
import streamlit as st
from typing import Optional, Tuple, Dict, List
from datetime import datetime, date
from zoneinfo import ZoneInfo
from supabase import create_client, Client

# ===================== CONFIG PAGINĂ =====================
st.set_page_config(page_title="ServicePack Reports (Mapping Strict)", layout="wide")
st.title("📊 ServicePack Reports (Mapping Strict)")

# ===================== SUPABASE CREDS ====================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("❌ Lipsesc SUPABASE_URL / SUPABASE_KEY în Streamlit → Settings → Secrets.")
    st.stop()

try:
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)  # type: Client
    st.success("Conexiune la Supabase OK.")
except Exception as e:
    st.error(f"Eroare conectare Supabase: {e}")
    st.stop()

# ===================== HELPERI GENERALE ==================
TZ = ZoneInfo("Europe/Bucharest")
NBSP = "\xa0"

def last_completed_month(today: datetime) -> date:
    first_of_this_month = datetime(today.year, today.month, 1, tzinfo=TZ).date()
    prev_month_last_day = first_of_this_month.replace(day=1) - pd.Timedelta(days=1)
    return prev_month_last_day.replace(day=1)

def extract_period_from_header(df_head: pd.DataFrame) -> Optional[date]:
    text = " ".join([" ".join(map(str, row.dropna().astype(str).tolist())) for _, row in df_head.iterrows()])
    m = re.search(r'(\d{2}[./-]\d{2}[./-]\d{2,4})\s*-\s*(\d{2}[./-]\d{2}[./-]\d{2,4})', text)
    if not m:
        return None
    start = pd.to_datetime(m.group(1), dayfirst=True, errors="coerce")
    if pd.isna(start):
        return None
    return start.date().replace(day=1)

def _extract_sku_from_product(text: str) -> Optional[str]:
    if not isinstance(text, str):
        text = str(text or "")
    m = re.search(r"\(([^()]+)\)\s*$", text)
    return m.group(1).strip() if m else None

# numeric parser robust (RO/EN)
NUMERIC_RE_THOUSAND_DOT_DECIMAL_COMMA = re.compile(r"^\d{1,3}(\.\d{3})+(,\d+)?$")
NUMERIC_RE_THOUSAND_COMMA_DECIMAL_DOT = re.compile(r"^\d{1,3}(,\d{3})+(\.\d+)?$")

def _to_number(val) -> Optional[float]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace(NBSP, "").replace(" ", "")
    s = re.sub(r"[^\d,.\-]", "", s)
    if s == "" or s.lower() in ("nan", "none"):
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
        return None

def parse_number(x) -> float:
    v = _to_number(x)
    return 0.0 if v is None else float(v)

def _extract_period_from_profit_header(xls_bytes: bytes) -> date:
    head = pd.read_excel(io.BytesIO(xls_bytes), sheet_name=0, header=None, nrows=8)
    txt = ""
    try:
        txt = str(head.iat[4, 0])
    except Exception:
        pass
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

# ===================== PRODUSE (STAGING) ==================
def _extract_product_name(product_text: str) -> str:
    if not isinstance(product_text, str):
        product_text = str(product_text or "")
    m = re.search(r"\s*\([^()]*\)\s*$", product_text)
    return product_text[:m.start()].strip() if m else product_text.strip()

def _ensure_products_from_profit(sb: Client, rows: List[Dict]) -> int:
    candidates: Dict[str, str] = {}
    for r in rows:
        sku = (str(r.get("sku") or "").strip().upper())
        if not sku:
            continue
        if sku not in candidates:
            name = _extract_product_name(r.get("product_text"))
            candidates[sku] = name
    if not candidates:
        return 0

    sku_list = list(candidates.keys())
    existing = set()
    try:
        CH = 1000
        for i in range(0, len(sku_list), CH):
            part = sku_list[i:i+CH]
            resp = sb.schema("staging").table("products").select("sku").in_("sku", part).execute()
            for row in (resp.data or []):
                existing.add((row.get("sku") or "").upper())
    except Exception:
        pass

    to_insert = [{"sku": s, "name": candidates[s]} for s in sku_list if s not in existing]
    if not to_insert:
        return 0

    sb.schema("staging").table("products").insert(to_insert).execute()
    return len(to_insert)

# ===================== MAPPING STRICT =====================
def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

def _load_mapping_from_csv(csv_bytes: bytes) -> Dict[str, Dict[str, str]]:
    df = pd.read_csv(io.BytesIO(csv_bytes))
    cols = {_norm_key(c): c for c in df.columns}
    need = {"table", "logical", "excel_header"}
    if not need.issubset(set(cols.keys())):
        raise RuntimeError("CSV mapping invalid. Coloane necesare: table, logical, excel_header")
    col_table = cols[_norm_key("table")]
    col_log   = cols[_norm_key("logical")]
    col_hdr   = cols[_norm_key("excel_header")]
    mp: Dict[str, Dict[str, str]] = {}
    for _, r in df.iterrows():
        t = str(r[col_table]).strip().lower()
        l = str(r[col_log]).strip().lower()
        h = str(r[col_hdr]).strip()
        if t not in mp:
            mp[t] = {}
        mp[t][l] = h
    return mp

# ===================== LOADERE (STRICT) ==================
def load_raw_profit_to_catalog(
    xls_bytes: bytes, expected_period: date, source_path: str, mapping: Dict[str, Dict[str, str]]
) -> Tuple[int, int, date]:
    if "raw_profit" not in mapping:
        raise RuntimeError("Mapping-ul nu conține secțiunea 'raw_profit'.")
    mp = mapping["raw_profit"]
    for k in ("product", "net", "cogs"):
        if k not in mp or not mp[k]:
            raise RuntimeError("Mapping 'raw_profit' lipsește cheia '%s'." % k)

    perioada = _extract_period_from_profit_header(xls_bytes)
    df = pd.read_excel(io.BytesIO(xls_bytes), sheet_name=0, header=10)

    missing = [h for h in (mp["product"], mp["net"], mp["cogs"]) if h not in df.columns]
    if missing:
        raise RuntimeError("În Excel lipsesc coloanele din mapping: %s" % missing)

    df_reset = df.reset_index(drop=True)
    base = pd.DataFrame({
        "row_number": (df_reset.index + 1).astype(int),
        "product_text": df_reset[mp["product"]],
        "sku": df_reset[mp["product"]].apply(_extract_sku_from_product),
        "net_sales_wo_vat": df_reset[mp["net"]].apply(_to_number),
        "cogs_wo_vat": df_reset[mp["cogs"]].apply(_to_number),
    })

    base = base[base["sku"].notna()]
    base = base[(base["net_sales_wo_vat"].notna()) | (base["cogs_wo_vat"].notna())]

    if perioada != expected_period:
        raise RuntimeError("Fișierul este pentru %s, dar accept doar %s" % (perioada, expected_period))

    # completează staging.products pentru SKU-urile noi
    product_rows = base[["sku", "product_text"]].to_dict(orient="records")
    _ensure_products_from_profit(sb, product_rows)

    # purge & insert in staging.raw_profit
    sb.schema("staging").table("raw_profit").delete().eq("period_month", perioada.isoformat()).execute()

    base = base.where(pd.notnull(base), None)
    base["period_month"] = perioada.isoformat()
    base["source_path"] = source_path

    records = base.to_dict(orient="records")
    written = 0
    for i in range(0, len(records), 1000):
        chunk = records[i:i+1000]
        sb.schema("staging").table("raw_profit").insert(chunk).execute()
        written += len(chunk)

    st.info("SUM(NET)=%.2f • SUM(COGS)=%.2f" % (
        float(base["net_sales_wo_vat"].fillna(0).sum()),
        float(base["cogs_wo_vat"].fillna(0).sum())
    ))

    return len(records), written, perioada

def load_raw_miscari_to_catalog(
    xls_bytes: bytes, expected_period: date, source_path: str, mapping: Dict[str, Dict[str, str]]
) -> Tuple[int, int, date]:
    if "raw_miscari" not in mapping:
        raise RuntimeError("Mapping-ul nu conține secțiunea 'raw_miscari'.")
    mp = mapping["raw_miscari"]
    for k in ("sku", "qty_out", "val_out"):
        if k not in mp or not mp[k]:
            raise RuntimeError("Mapping 'raw_miscari' lipsește cheia '%s'." % k)

    head = pd.read_excel(io.BytesIO(xls_bytes), sheet_name=0, header=None, nrows=10)
    period = extract_period_from_header(head)
    if not period or period != expected_period:
        raise RuntimeError("Perioada din antet nu corespunde ultimei luni încheiate (%s)." % expected_period)

    df = pd.read_excel(io.BytesIO(xls_bytes), sheet_name=0, skiprows=9)

    missing = [h for h in (mp["sku"], mp["qty_out"], mp["val_out"]) if h not in df.columns]
    if missing:
        raise RuntimeError("În Excel (mișcări) lipsesc coloanele din mapping: %s" % missing)

    df_reset = df.reset_index(drop=True)
    out = pd.DataFrame({
        "row_number": (df_reset.index + 1).astype(int),
        "sku": df_reset[mp["sku"]].astype(str).str.strip().str.upper(),
        "qty_out": df_reset[mp["qty_out"]].apply(parse_number),
        "val_out": df_reset[mp["val_out"]].apply(parse_number),
    })

    out = out[out["sku"].notna() & (out["sku"].str.len() > 0)]

    sb.schema("staging").table("raw_miscari").delete().eq("period_month", period.isoformat()).execute()

    out = out.where(pd.notnull(out), None)
    out["period_month"] = period.isoformat()
    out["source_path"]  = source_path

    records = out.to_dict(orient="records")
    written = 0
    for i in range(0, len(records), 1000):
        chunk = records[i:i+1000]
        sb.schema("staging").table("raw_miscari").insert(chunk).execute()
        written += len(chunk)

    st.info("SUM(QTY_OUT)=%.0f • SUM(VAL_OUT)=%.2f" % (
        float(out["qty_out"].fillna(0).sum()),
        float(out["val_out"].fillna(0).sum())
    ))

    return len(records), written, period

# ===================== UI / TABS =========================
now_ro = datetime.now(tz=TZ)
lcm = last_completed_month(now_ro)
st.info("🗓️ Ultima lună încheiată: **%s**" % lcm.strftime('%Y-%m'))

tab_status, tab_upload, tab_debug = st.tabs(
    ["📅 Status pe luni", "⬆️ Upload fișiere (mapping strict)", "🧪 Debug staging"]
)

# ---------- TAB STATUS ----------
with tab_status:
    st.subheader("Stare pe luni (period_registry)")
    try:
        resp = sb.table("period_registry").select("*").order("period_month", desc=True).limit(24).execute()
        df = pd.DataFrame(resp.data or [])
        if df.empty:
            st.info("Nu există încă înregistrări în period_registry.")
        else:
            df["period_month"] = pd.to_datetime(df["period_month"]).dt.date
            st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Nu pot citi period_registry: {e}")

# ---------- TAB UPLOAD (MAPPING STRICT) ----------
with tab_upload:
    st.subheader("⬆️ Upload fișiere (Mapping strict) → schema `staging`")

    if "mapping_cfg" not in st.session_state:
        st.session_state["mapping_cfg"] = None

    map_file = st.file_uploader("📄 CSV mapping (table, logical, excel_header)", type=["csv"], key="map_upload")
    if map_file is not None:
        try:
            mapping_cfg = _load_mapping_from_csv(map_file.read())
            st.session_state["mapping_cfg"] = mapping_cfg
            st.success("Mapping încărcat ✔️")
        except Exception as e:
            st.error(f"Eroare mapping: {e}")

    mapping_cfg = st.session_state["mapping_cfg"]
    if not mapping_cfg:
        st.warning("Încarcă mai întâi CSV-ul de mapping.")
    else:
        file_type = st.radio("Tip fișier", ["Profit pe produs", "Mișcări stocuri"], horizontal=True)
        uploaded_file = st.file_uploader("Excel (XLS/XLSX)", type=["xlsx", "xls"])
        if uploaded_file is not None:
            try:
                data = uploaded_file.read()
                if file_type == "Profit pe produs":
                    rows_in, rows_written, period_detected = load_raw_profit_to_catalog(
                        data, lcm, uploaded_file.name, mapping_cfg
                    )
                    st.success("Import PROFIT OK (%s) • rânduri: %d" % (period_detected.strftime('%Y-%m'), rows_in))
                else:
                    rows_in, rows_written, period_detected = load_raw_miscari_to_catalog(
                        data, lcm, uploaded_file.name, mapping_cfg
                    )
                    st.success("Import MISCĂRI OK (%s) • rânduri: %d" % (period_detected.strftime('%Y-%m'), rows_in))
            except Exception as e:
                st.error("Eroare import: %s" % e)

# ---------- TAB DEBUG ----------
with tab_debug:
    st.subheader("📋 Preview staging pentru %s" % lcm.strftime('%Y-%m'))
    try:
        p = sb.schema("staging").table("raw_profit").select("*").eq("period_month", lcm.isoformat()).limit(1000).execute()
        m = sb.schema("staging").table("raw_miscari").select("*").eq("period_month", lcm.isoformat()).limit(1000).execute()
        dfp = pd.DataFrame(p.data or [])
        dfm = pd.DataFrame(m.data or [])
        c1, c2 = st.columns(2)
        with c1:
            st.caption("staging.raw_profit")
            st.dataframe(dfp, use_container_width=True)
        with c2:
            st.caption("staging.raw_miscari")
            st.dataframe(dfm, use_container_width=True)
        st.caption("RAW PROFIT: %d rânduri • RAW MISCĂRI: %d rânduri" % (len(dfp), len(dfm)))
    except Exception as e:
        st.error("Eroare la citirea din staging: %s" % e)

# ---- END APP.PY (Mapping Strict) ----
