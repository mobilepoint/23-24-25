# ---- APP.PY (ServicePack Reports • RAW → staging) ----
import os
import io
import re
import pandas as pd
import streamlit as st
from typing import Optional, Tuple
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

# ===================== UTILITARE GENERALE ==================
TZ = ZoneInfo("Europe/Bucharest")
NBSP = "\xa0"  # non-breaking space pentru numere formate RO

def last_completed_month(today: datetime) -> date:
    first_of_this_month = datetime(today.year, today.month, 1, tzinfo=TZ).date()
    prev_month_last_day = first_of_this_month.replace(day=1) - pd.Timedelta(days=1)
    return prev_month_last_day.replace(day=1)

def extract_period_from_header(df_head: pd.DataFrame) -> Optional[date]:
    """
    Caută în primele ~10 rânduri un text de forma:
    'Perioada: 01/01/2023 - 31/01/2023' și returnează prima zi a lunii.
    """
    text = " ".join([" ".join(map(str, row.dropna().astype(str).tolist())) for _, row in df_head.iterrows()])
    m = re.search(r'(\d{2}[./-]\d{2}[./-]\d{2,4})\s*-\s*(\d{2}[./-]\d{2}[./-]\d{2,4})', text)
    if not m:
        return None
    start = pd.to_datetime(m.group(1), dayfirst=True, errors="coerce")
    if pd.isna(start):
        return None
    return start.date().replace(day=1)

def _normalize_columns(cols) -> dict:
    """Returnează {col_original: col_normalizata} pentru matching tolerant."""
    norm_map = {}
    for c in cols:
        key = str(c).lower().replace("\n", " ").replace("\r", " ")
        key = re.sub(r"[^a-z0-9]+", " ", key)
        key = re.sub(r"\s+", " ", key).strip()
        norm_map[c] = key
    return norm_map

def _extract_sku_from_product(text: str) -> Optional[str]:
    """SKU este ultimul șir între paranteze în coloana Produsul."""
    if not isinstance(text, str):
        text = str(text or "")
    m = re.search(r"\(([^()]+)\)\s*$", text)
    return m.group(1).strip() if m else None

# ---- numeric parser robust (RO/EN, spații, NBSP, mii, zecimale) ----
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
    """Variantă care întoarce 0.0 pentru neparseabile (folosită la mișcări)."""
    v = _to_number(x)
    return 0.0 if v is None else float(v)

def _extract_period_from_profit_header(xls_bytes: bytes) -> date:
    """Citește rândul 5 col A (ex: 'Perioada: 01/08/2025 - 31/08/2025') și întoarce prima zi a lunii."""
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
    """Numele fără ultimul (...) (de obicei SKU)."""
    if not isinstance(product_text, str):
        product_text = str(product_text or "")
    m = re.search(r"\s*\([^()]*\)\s*$", product_text)
    return product_text[:m.start()].strip() if m else product_text.strip()

def _ensure_products_from_profit(sb: Client, rows: list[dict]) -> int:
    """
    Dacă SKU există în staging.products → ignorăm.
    Dacă NU există → inserăm (sku, name) cu numele extras din fișierul de profit.
    """
    candidates = {}
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

    # Ce SKU-uri există deja
    existing = set()
    try:
        CH = 1000
        for i in range(0, len(sku_list), CH):
            part = sku_list[i:i+CH]
            resp = (
                sb.schema("staging")
                  .table("products")
                  .select("sku")
                  .in_("sku", part)
                  .execute()
            )
            for row in (resp.data or []):
                existing.add((row.get("sku") or "").upper())
    except Exception:
        pass

    to_insert = [{"sku": s, "name": candidates[s]} for s in sku_list if s not in existing]
    if not to_insert:
        return 0

    sb.schema("staging").table("products").insert(to_insert).execute()
    return len(to_insert)

# ===================== LOADERE RAW → STAGING ==================
def load_raw_profit_to_catalog(xls_bytes: bytes, expected_period: date, source_path: str) -> Tuple[int, int, date]:
    """
    Încarcă PROFIT: staging.raw_profit + completează staging.products.
    """
    perioada = _extract_period_from_profit_header(xls_bytes)

    df_raw = pd.read_excel(io.BytesIO(xls_bytes), sheet_name=0, header=10)
    df_raw = df_raw.loc[:, ~df_raw.columns.to_series().astype(str).str.fullmatch(r"\s*nan\s*", case=False)]
    norm_map = _normalize_columns(df_raw.columns)

    col_prod = next((c for c in df_raw.columns if norm_map[c] in ("produsul", "produs")), None)
    if col_prod is None and len(df_raw.columns) >= 2:
        col_prod = df_raw.columns[1]

    # fallback simplu pe pozițiile "clasice" (E=net, F=cogs)
    col_net  = df_raw.columns[4]
    col_cogs = df_raw.columns[5]

    df_reset = df_raw.reset_index(drop=True)
    base = pd.DataFrame({
        "row_number": (df_reset.index + 1).astype(int),
        "product_text": df_reset[col_prod],
        "sku": df_reset[col_prod].apply(_extract_sku_from_product),
        "net_sales_wo_vat": df_reset[col_net].apply(_to_number),
        "cogs_wo_vat": df_reset[col_cogs].apply(_to_number),
    })

    base = base[base["sku"].notna()]
    base = base[(base["net_sales_wo_vat"].notna()) | (base["cogs_wo_vat"].notna())]

    if perioada != expected_period:
        raise RuntimeError(
            f"Fișierul este pentru {perioada.strftime('%Y-%m')}, dar aici acceptăm doar {expected_period.strftime('%Y-%m')}."
        )

    if base.empty:
        return 0, 0, perioada

    # 1) completează staging.products
    product_rows = base[["sku", "product_text"]].to_dict(orient="records")
    inserted = _ensure_products_from_profit(sb, product_rows)
    if inserted:
        st.info(f"Adăugate {inserted} SKU noi în staging.products.")

    # 2) purge pe lună în staging.raw_profit
    sb.schema("staging").table("raw_profit").delete().eq("period_month", perioada.isoformat()).execute()

    # 3) insert batch
    base = base.where(pd.notnull(base), None)
    base["period_month"] = perioada.isoformat()
    base["source_path"] = source_path

    records = base.to_dict(orient="records")
    written = 0
    BATCH = 1000
    for i in range(0, len(records), BATCH):
        chunk = records[i:i+BATCH]
        sb.schema("staging").table("raw_profit").insert(chunk).execute()
        written += len(chunk)

    return len(records), written, perioada

def load_raw_miscari_to_catalog(xls_bytes: bytes, expected_period: date, source_path: str) -> Tuple[int, int, date]:
    """
    Încarcă MISCĂRI (qty_out / val_out) în staging.raw_miscari.
    """
    head = pd.read_excel(io.BytesIO(xls_bytes), sheet_name=0, header=None, nrows=10)
    period = extract_period_from_header(head)
    if not period:
        raise RuntimeError("Nu am putut detecta perioada din antet.")
    if period != expected_period:
        raise RuntimeError(
            f"Fișierul este pentru {period.strftime('%Y-%m')}, dar aici acceptăm doar {expected_period.strftime('%Y-%m')}."
        )

    df = pd.read_excel(io.BytesIO(xls_bytes), sheet_name=0, skiprows=9)
    norm_map2 = {c: re.sub(r"[^a-z0-9]", "", str(c).lower()) for c in df.columns}

    col_sku     = next((c for c in df.columns if norm_map2[c] in ["cod", "sku"]), None)
    col_qty_out = next((c for c in df.columns if "iesiri" in norm_map2[c] and ".1" not in str(c)), None)
    col_val_out = next((c for c in df.columns if "iesiri" in norm_map2[c] and ".1" in str(c)), None)

    if not all([col_sku, col_qty_out, col_val_out]):
        raise RuntimeError("Nu am găsit toate coloanele necesare în mișcări stocuri (sku / ieșiri buc / ieșiri lei).")

    df_reset = df.reset_index(drop=True)
    out = pd.DataFrame({
        "row_number": (df_reset.index + 1).astype(int),
        "sku": df_reset[col_sku].astype(str).str.strip().str.upper(),
        "qty_out": df_reset[col_qty_out].apply(parse_number),
        "val_out": df_reset[col_val_out].apply(parse_number),
    })

    out = out[out["sku"].notna() & (out["sku"].str.len() > 0)]
    if out.empty:
        return 0, 0, period

    # purge pe lună
    sb.schema("staging").table("raw_miscari").delete().eq("period_month", period.isoformat()).execute()

    out = out.where(pd.notnull(out), None)
    out["period_month"] = period.isoformat()
    out["source_path"]  = source_path

    records = out.to_dict(orient="records")
    written = 0
    BATCH = 1000
    for i in range(0, len(records), BATCH):
        chunk = records[i:i+BATCH]
        sb.schema("staging").table("raw_miscari").insert(chunk).execute()
        written += len(chunk)

    return len(records), written, period

# ===================== UI: STATUS / UPLOAD / DEBUG ==================
now_ro = datetime.now(tz=TZ)
lcm = last_completed_month(now_ro)
st.info(f"🗓️ Ultima lună încheiată (cerută pentru încărcare standard): **{lcm.strftime('%Y-%m')}**")

tab_status, tab_upload, tab_debug = st.tabs(
    ["📅 Status pe luni", "⬆️ Upload fișiere (RAW → staging)", "🧪 Debug balanțe"]
)

# ---------- TAB STATUS ----------
with tab_status:
    st.subheader("Stare pe luni (public.period_registry)")
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

# ---------- TAB UPLOAD ----------
with tab_upload:
    st.subheader("Încarcă fișierul pentru luna acceptată (RAW → staging)")
    file_type = st.radio("Tip fișier", ["Profit pe produs", "Mișcări stocuri"], horizontal=True)
    uploaded_file = st.file_uploader("Alege fișierul Excel (XLS/XLSX)", type=["xlsx", "xls"])

    if uploaded_file is not None:
        try:
            data = uploaded_file.read()
            if file_type == "Profit pe produs":
                rows_in, rows_written, period_detected = load_raw_profit_to_catalog(data, lcm, uploaded_file.name)
                st.success(f"Import RAW PROFIT OK ({period_detected.strftime('%Y-%m')}). Rânduri parse: {rows_in}, scrise: {rows_written}.")
            else:
                rows_in, rows_written, period_detected = load_raw_miscari_to_catalog(data, lcm, uploaded_file.name)
                st.success(f"Import RAW MISCĂRI OK ({period_detected.strftime('%Y-%m')}). Rânduri parse: {rows_in}, scrise: {rows_written}.")
        except Exception as e:
            st.error(f"Eroare la procesarea fișierului: {e}")

# ---------- TAB DEBUG ----------
with tab_debug:
    st.subheader(f"Verifică ce s-a încărcat în staging pentru {lcm.strftime('%Y-%m')}")
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
        st.caption(f"RAW PROFIT rows: {len(dfp)} • RAW MISCĂRI rows: {len(dfm)}")
    except Exception as e:
        st.error(f"Eroare la citirea datelor din staging: {e}")

# ---- END APP.PY (ServicePack Reports • RAW → staging) ----
