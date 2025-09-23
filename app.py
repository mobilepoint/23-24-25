# ---- APP.PY (ServicePack Reports) ----
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

# ===================== HELPERI BUSINESS & NORMALIZARE ==================
TZ = ZoneInfo("Europe/Bucharest")

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
    m = re.search(r'(\d{2}/\d{2}/\d{4})\s*-\s*(\d{2}/\d{2}/\d{4})', text)
    if not m:
        return None
    start = pd.to_datetime(m.group(1), dayfirst=True).date()
    return start.replace(day=1)

# --- utilitare normalizare text (coloane) ---
_norm_tbl = str.maketrans({c: "" for c in " .,_-/%()[]{}:"})
def norm(s: str) -> str:
    """lower + elimină spații/punctuație (fără diacritice)."""
    return str(s).strip().lower().translate(_norm_tbl)

def _normalize_columns(cols) -> dict:
    """Returnează {col_original: col_normalizata} pentru matching tolerant."""
    norm_map = {}
    for c in cols:
        key = (
            str(c)
            .lower()
            .replace("\n", " ")
            .replace("\r", " ")
        )
        key = re.sub(r"[^a-z0-9]+", " ", key)  # doar litere/cifre/spații
        key = re.sub(r"\s+", " ", key).strip()
        norm_map[c] = key
    return norm_map

# ---- NUMERIC CONVERSION (unificat) ----
NBSP = "\xa0"  # non-breaking space
NUMERIC_RE_THOUSAND_DOT_DECIMAL_COMMA = re.compile(r"^\d{1,3}(\.\d{3})+(,\d+)?$")
NUMERIC_RE_THOUSAND_COMMA_DECIMAL_DOT = re.compile(r"^\d{1,3}(,\d{3})+(\.\d+)?$")

def _to_number(val) -> Optional[float]:
    """
    Convertor robust pentru string-uri RO/EN, inclusiv separatori de mii cu NBSP/spații.
    Acceptă: 1.021,02 | 1,021.02 | 1 021,02 | 104,04 | 104.04 | 104
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (int, float)):
        return float(val)

    s = str(val)
    if not s:
        return None

    # normalizează: strip + înlocuiește NBSP și spații dintre mii
    s = s.strip().replace(NBSP, "").replace(" ", "")
    # elimină orice alt text (ex. ' lei')
    s = re.sub(r"[^\d,.\-]", "", s)

    if s == "" or s.lower() in ("nan", "none"):
        return None

    # 1.021,02 (thousand=., decimal=,)
    if NUMERIC_RE_THOUSAND_DOT_DECIMAL_COMMA.match(s):
        return float(s.replace(".", "").replace(",", "."))
    # 1,021.02 (thousand=,, decimal=.)
    if NUMERIC_RE_THOUSAND_COMMA_DECIMAL_DOT.match(s):
        return float(s.replace(",", ""))
    # doar virgulă -> presupunem decimal=,
    if "," in s and "." not in s:
        return float(s.replace(",", "."))
    # fallback: decimal='.'
    try:
        return float(s)
    except Exception:
        return None

# Backward-compat: parse_number delegă la _to_number
def parse_number(x) -> float:
    v = _to_number(x)
    return 0.0 if v is None else float(v)

# ---- SKU / PRODUS helpers ----
def extract_last_parenthesized(text: str) -> Optional[str]:
    """Extrage ultimul () din șir -> SKU. Ex: 'Name (X) (SKU-123)' => 'SKU-123'."""
    if not isinstance(text, str):
        text = str(text or "")
    matches = re.findall(r'\(([^()]*)\)', text)
    return matches[-1].strip() if matches else None

def _extract_sku_from_product(text: str) -> Optional[str]:
    """SKU este ultimul șir între paranteze în coloana Produsul."""
    if not isinstance(text, str):
        text = str(text)
    m = re.search(r"\(([^()]+)\)\s*$", text)
    return m.group(1).strip() if m else None

def _extract_product_name(product_text: str) -> str:
    """
    Din 'Produsul' extrage numele fără ultimul (...) (care de obicei e SKU).
    Exemplu: 'Carcasa Samsung A50 (negru) (GH82-30465A)' -> 'Carcasa Samsung A50 (negru)'
    Dacă nu găsește paranteze, returnează textul original curățat.
    """
    if not isinstance(product_text, str):
        product_text = str(product_text or "")
    m = re.search(r"\s*\([^()]*\)\s*$", product_text)
    if m:
        return product_text[:m.start()].strip()
    return product_text.strip()

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

# ---- PRODUCTS master: dacă SKU nu există, îl adăugăm cu nume din profit ----
def _ensure_products_from_profit(sb: Client, rows: list[dict]) -> int:
    """
    Primește rânduri din profit (cu 'sku' și 'product_text').
    - normalizează SKU: UPPER + trim
    - extrage numele cu _extract_product_name
    - inserează NUMAI SKU-urile care NU există deja în catalog.products
    Returnează câte produse noi a inserat.
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

    existing = set()
    try:
        CH = 1000
        for i in range(0, len(sku_list), CH):
            part = sku_list[i:i+CH]
            resp = sb.schema("catalog").table("products").select("sku").in_("sku", part).execute()
            for row in (resp.data or []):
                existing.add((row.get("sku") or "").upper())
    except Exception:
        pass

    to_insert = [{"sku": sku, "name": candidates[sku]} for sku in sku_list if sku not in existing]
    if not to_insert:
        return 0

    sb.schema("catalog").table("products").insert(to_insert).execute()
    return len(to_insert)

# (compat) find_profit_columns încă disponibil dacă ai nevoie pe viitor
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
    if col_prod is None and ncols >= 2:
        col_prod = df.columns[1]
    if col_net is None and ncols >= 5:
        col_net = df.columns[4]
    if col_cogs is None and ncols >= 6:
        col_cogs = df.columns[5]
    return col_prod, col_net, col_cogs

def get_period_row(sb: Client, period: date) -> Optional[dict]:
    """Citește rândul din public.period_registry pentru perioada dată (dacă există în proiectul tău)."""
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

# =========================================================
# =============== LOADER-E RAW (schema: catalog) ==========
# =========================================================

# ---- LOADER: RAW PROFIT (catalog.raw_profit + products) ----
def load_raw_profit_to_catalog(xls_bytes: bytes, expected_period: date, source_path: str) -> tuple[int, int, date]:
    """
    Încarcă raportul 'Profit pe produs' în catalog.raw_profit.
    - extrage period_month din antet
    - parsează header (rând 11, adică header=10)
    - row_number = 1..n
    - normalizează numere (mii '.'/spațiu/NBSP, decimale ','/'.')
    - adaugă SKU-urile noi în catalog.products (cu numele din fișierul de profit)
    - purge pe luna importată înainte de insert (doar în catalog.raw_profit)
    Returnează (rows_in, rows_written, period_detected)
    """
    perioada = _extract_period_from_profit_header(xls_bytes)

    # citire cu header pe rândul 11 -> header=10
    df_raw = pd.read_excel(io.BytesIO(xls_bytes), sheet_name=0, header=10)
    df_raw = df_raw.loc[:, ~df_raw.columns.to_series().astype(str).str.fullmatch(r"\s*nan\s*", case=False)]
    norm_map = _normalize_columns(df_raw.columns)

    # găsire coloane
    col_prod = next((c for c in df_raw.columns if norm_map[c] in ("produsul", "produs")), None)
    if col_prod is None and len(df_raw.columns) >= 2:
        col_prod = df_raw.columns[1]

    # candidați pentru net și cogs
    def candidates_for(keys_contains: list[str]) -> list[str]:
        cans = []
        for c in df_raw.columns:
            k = norm_map[c]
            if all(sub in k for sub in keys_contains):
                cans.append(c)
        # fallback: primele coloane numerice după B
        for idx in range(3, min(9, len(df_raw.columns))):
            cans.append(df_raw.columns[idx])
        # unicizează
        seen, out = set(), []
        for c in cans:
            if c not in seen:
                out.append(c); seen.add(c)
        return out

    def pick_best_numeric(cols: list[str]) -> Optional[str]:
        best_col, best_nonnull, best_sum = None, -1, -1.0
        for c in cols:
            try:
                s = df_raw[c].apply(_to_number)
            except Exception:
                continue
            nonnull = int(s.notna().sum())
            total = float(s.fillna(0).sum())
            if nonnull > best_nonnull or (nonnull == best_nonnull and total > best_sum):
                best_col, best_nonnull, best_sum = c, nonnull, total
        return best_col

    col_net  = pick_best_numeric(candidates_for(["vanzari", "nete"]))
    col_cogs = pick_best_numeric(candidates_for(["cost", "bunurilor"])) or pick_best_numeric(candidates_for(["cogs"]))

    if not col_prod or not col_net or not col_cogs:
        raise RuntimeError("Nu pot identifica coloanele obligatorii: Produsul / Vanzari nete / Costul bunurilor vandute.")

    # construim baza
    df_reset = df_raw.reset_index(drop=True)
    base = pd.DataFrame({
        "row_number": (df_reset.index + 1).astype(int),
        "product_text": df_reset[col_prod],
        "sku": df_reset[col_prod].apply(_extract_sku_from_product),
        "net_sales_wo_vat": df_reset[col_net].apply(_to_number),
        "cogs_wo_vat": df_reset[col_cogs].apply(_to_number),
    })

    # filtre valide
    base = base[base["sku"].notna()]
    base = base[(base["net_sales_wo_vat"].notna()) | (base["cogs_wo_vat"].notna())]

    if perioada != expected_period:
        raise RuntimeError(f"Fișierul este pentru {perioada.strftime('%Y-%m')}, dar aici acceptăm doar {expected_period.strftime('%Y-%m']}.")

    if base.empty:
        return 0, 0, perioada

    # 1) completează produse pentru SKU-urile noi
    product_rows = base[["sku", "product_text"]].to_dict(orient="records")
    inserted = _ensure_products_from_profit(sb, product_rows)
    if inserted:
        st.info(f"Adăugate {inserted} SKU noi în catalog.products.")

    # 2) purge luna în raw_profit
    sb.schema("catalog").table("raw_profit").delete().eq("period_month", perioada.isoformat()).execute()

    # 3) insert în batch
    base = base.where(pd.notnull(base), None)
    base["period_month"] = perioada.isoformat()
    base["source_path"] = source_path

    records = base[["period_month","row_number","product_text","sku","net_sales_wo_vat","cogs_wo_vat","source_path"]].to_dict(orient="records")

    written = 0
    BATCH = 1000
    for i in range(0, len(records), BATCH):
        chunk = records[i:i+BATCH]
        sb.schema("catalog").table("raw_profit").insert(chunk).execute()
        written += len(chunk)

    return len(records), written, perioada
# ---- END LOADER: RAW PROFIT ----

# ---- LOADER: RAW MISCĂRI (catalog.raw_miscari) ----
def load_raw_miscari_to_catalog(xls_bytes: bytes, expected_period: date, source_path: str) -> tuple[int, int, date]:
    """
    Încarcă 'Mișcări stocuri' în catalog.raw_miscari:
    - detectează perioada din antet
    - header la rândul 10 (skiprows=9)
    - row_number = 1..n
    - mapare strictă pe cantități/valori (qty_out = 'Ieșiri'/ 'Iesiri', val_out = 'Ieșiri.1'/ 'Iesiri.1', etc.)
    - purge pe luna importată înainte de insert (doar în catalog.raw_miscari)
    """
    # detectare perioadă din antet
    head = pd.read_excel(io.BytesIO(xls_bytes), sheet_name=0, header=None, nrows=10)
    period = extract_period_from_header(head)
    if not period:
        raise RuntimeError("Nu am putut detecta perioada din antet.")
    if period != expected_period:
        raise RuntimeError(f"Fișierul este pentru {period.strftime('%Y-%m')}, dar aici acceptăm doar {expected_period.strftime('%Y-%m')}.")

    # citire tabel
    df = pd.read_excel(io.BytesIO(xls_bytes), sheet_name=0, skiprows=9)
    norm_map2 = {c: norm(c) for c in df.columns}

    col_sku       = next((c for c in df.columns if norm_map2[c] in ["cod", "cod1", "sku"]), None)
    col_qty_open  = next((c for c in df.columns if norm_map2[c].startswith("stocinitial")), None)
    col_qty_in    = next((c for c in df.columns if norm_map2[c] == "intrari"), None)

    # cantități
    # ex: 'Iesiri'/'Ieșiri' (qty) vs 'Iesiri.1'/'Ieșiri.1' (val)
    col_qty_out   = next((c for c in df.columns if norm_map2[c].startswith("iesiri") and ".1" not in str(c).lower()), None)
    col_qty_close = next((c for c in df.columns if norm_map2[c].startswith("stocfinal")), None)

    # valori
    col_val_open  = next((c for c in df.columns if norm_map2[c].startswith("soldinitial")), None)
    col_val_in    = next((c for c in df.columns if norm_map2[c] == "intrari.1" or (norm_map2[c] == "intrari" and c != col_qty_in)), None)
    col_val_out   = next((c for c in df.columns if norm_map2[c].startswith("iesiri") and ".1" in str(c).lower()), None)
    col_val_close = next((c for c in df.columns if norm_map2[c].startswith("soldfinal")), None)

    if not all([col_sku, col_qty_open, col_qty_in, col_qty_out, col_qty_close, col_val_open, col_val_in, col_val_out, col_val_close]):
        raise RuntimeError("Nu am găsit toate coloanele necesare în mișcări stocuri.")

    df_reset = df.reset_index(drop=True)
    out = pd.DataFrame({
        "row_number": (df_reset.index + 1).astype(int),
        "product_text": df_reset.get("Produsul") if "Produsul" in df_reset.columns else None,
        "sku": df_reset[col_sku].astype(str).str.strip().str.upper(),
        "qty_open":  df_reset[col_qty_open].apply(_to_number),
        "qty_in":    df_reset[col_qty_in].apply(_to_number),
        "qty_out":   df_reset[col_qty_out].apply(_to_number),
        "qty_close": df_reset[col_qty_close].apply(_to_number),
        "val_open":  df_reset[col_val_open].apply(_to_number),
        "val_in":    df_reset[col_val_in].apply(_to_number),
        "val_out":   df_reset[col_val_out].apply(_to_number),
        "val_close": df_reset[col_val_close].apply(_to_number),
    })

    out = out[out["sku"].notna() & (out["sku"].str.len() > 0)]

    if out.empty:
        return 0, 0, period

    # purge pe lună
    sb.schema("catalog").table("raw_miscari").delete().eq("period_month", period.isoformat()).execute()

    # insert în batch
    out = out.where(pd.notnull(out), None)
    out["period_month"] = period.isoformat()
    out["source_path"]  = source_path

    records = out.to_dict(orient="records")
    written = 0
    BATCH = 1000
    for i in range(0, len(records), BATCH):
        chunk = records[i:i+BATCH]
        sb.schema("catalog").table("raw_miscari").insert(chunk).execute()
        written += len(chunk)

    return len(records), written, period
# ---- END LOADER: RAW MISCĂRI ----

# ===================== UI: UPLOAD RAW (catalog) =========================
with tab_upload:
    st.subheader("Încarcă fișierul pentru luna acceptată (RAW → catalog)")
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

# ===================== UI: CONSOLIDARE & RAPOARTE =======================
with tab_consol:
    st.subheader(f"Consolidare pentru {lcm.strftime('%Y-%m')}")
    c1, c2, c3 = st.columns([1,1,2])
    TOL_QTY = c1.number_input("Toleranță cantități (buc)", min_value=0.0, value=0.01, step=0.01, format="%.2f")
    TOL_VAL = c2.number_input("Toleranță valori (lei fără TVA)", min_value=0.0, value=5.00, step=0.10, format="%.2f")
    overwrite = c3.checkbox("Forțează overwrite luna curentă în core/mart", value=True)

    row = get_period_row(sb, lcm)

    try:
        sum_resp = sb.table("v_balance_summary").select("*").eq("period_month", lcm.isoformat()).execute()
        sum_df = pd.DataFrame(sum_resp.data) if sum_resp.data else pd.DataFrame()
        total_diff_qty = float(sum_df["total_diff_qty"].iloc[0]) if not sum_df.empty else None
        total_diff_val = float(sum_df["total_diff_val"].iloc[0]) if not sum_df.empty else None
    except Exception:
        total_diff_qty = None
        total_diff_val = None

    if not row:
        st.warning("Nu există încă intrări pentru această lună în registru. Încarcă mai întâi fișierele în tabul „Upload”.")
    else:
        profit_ok  = bool(row.get("profit_loaded"))
        miscari_ok = bool(row.get("miscari_loaded"))
        qty_ok_reg = bool(row.get("balance_ok_qty"))
        val_ok_reg = bool(row.get("balance_ok_val"))

        qty_ok_eff = qty_ok_reg or (total_diff_qty is not None and abs(total_diff_qty) <= TOL_QTY)
        val_ok_eff = val_ok_reg or (total_diff_val is not None and abs(total_diff_val) <= TOL_VAL)

        cols = st.columns(4)
        cols[0].metric("Profit încărcat?", "DA" if profit_ok else "NU")
        cols[1].metric("Mișcări încărcate?", "DA" if miscari_ok else "NU")
        cols[2].metric("Balanță cantități OK?", "DA" if qty_ok_eff else "NU", delta=None if total_diff_qty is None else f"Δ {total_diff_qty:.2f}")
        cols[3].metric("Balanță valori OK?", "DA" if val_ok_eff else "NU", delta=None if total_diff_val is None else f"Δ {total_diff_val:.2f}")

        ready = all([profit_ok, miscari_ok, qty_ok_eff, val_ok_eff])

        if row.get("consolidated_to_core"):
            st.success("✅ Luna este deja consolidată. Rapoartele pot folosi `mart.sales_monthly`.")
        elif not ready:
            st.error("Nu poți consolida încă. Asigură-te că ambele fișiere sunt încărcate și balanțele sunt în toleranțe.")
        else:
            if st.button("🚀 Consolidează luna"):
                try:
                    if overwrite:
                        purge = sb.rpc("purge_core_month", {"p_period": lcm.isoformat()}).execute()
                        info = purge.data or {}
                        st.info(
                            f"Purge luna → profit: {info.get('profit_before', 0)}→{info.get('profit_after', 0)}, "
                            f"mișcări: {info.get('miscari_before', 0)}→{info.get('miscari_after', 0)}"
                        )
                        if (info.get("profit_after", 0) or 0) > 0 or (info.get("miscari_after", 0) or 0) > 0:
                            st.error("Nu pot continua: există încă rânduri în core.fact_* pentru luna curentă. Verifică permisiunile/RLS.")
                            st.stop()

                    sb.rpc(
                        "consolidate_month_tolerant",
                        {
                            "p_period": lcm.isoformat(),
                            "p_tol_qty": TOL_QTY,
                            "p_tol_val": TOL_VAL,
                            "p_overwrite": False,
                        },
                    ).execute()
                    st.success("Consolidare reușită. `core.*` a fost suprascris pentru această lună, iar materialized view-urile au fost reîmprospătate (dacă există).")
                except Exception as e:
                    st.error(f"Eroare la consolidare: {e}")

# ---------- TAB 🧪 DEBUG BALANȚE ----------
with tab_debug:
    st.subheader(f"Verifică diferențele de balanță pentru {lcm.strftime('%Y-%m')}")
    try:
        s = sb.table("v_balance_summary").select("*").eq("period_month", lcm.isoformat()).execute()
        sdf = pd.DataFrame(s.data)
        if not sdf.empty:
            sdf["period_month"] = pd.to_datetime(sdf["period_month"]).dt.date
            st.write("**Rezumat:**")
            st.dataframe(sdf, use_container_width=True)
        else:
            st.info("Nu există rezumat pentru această perioadă (posibil să nu fie încărcate mișcările).")

        d = sb.table("v_balance_issues").select("*").eq("period_month", lcm.isoformat()).limit(10000).execute()
        ddf = pd.DataFrame(d.data)
        if ddf.empty:
            st.success("🎉 Nicio diferență de balanță pe SKU.")
        else:
            ddf["period_month"] = pd.to_datetime(ddf["period_month"]).dt.date
            st.write("**Diferențe pe SKU (abs(diff_qty) > 0.01 sau abs(diff_val) > 0.01):**")
            st.dataframe(ddf, use_container_width=True)
    except Exception as e:
        st.error(f"Eroare la citirea v_balance_*: {e}")

# ---------- RAPOARTE: Top sellers ----------
st.divider()
st.header("Rapoarte")

tab_top, tab_velocity, tab_reco = st.tabs(["🏆 Top sellers", "📈 Velocity (în lucru)", "🧠 Recomandări (în lucru)"])

with tab_top:
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
        resp = (
            sb.table("v_sales_monthly")
              .select("period_month, sku, net_sales_wo_vat, cogs_wo_vat, profit_wo_vat, margin_pct, qty_sold")
              .gte("period_month", start_date.isoformat())
              .lte("period_month", end_date.isoformat())
              .execute()
        )
        df = pd.DataFrame(resp.data or [])
        if df.empty:
            st.info("Nu există date pentru intervalul ales.")
        else:
            for c in ["net_sales_wo_vat", "cogs_wo_vat", "profit_wo_vat", "margin_pct", "qty_sold"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            df["period_month"] = pd.to_datetime(df["period_month"]).dt.to_period("M").dt.to_timestamp()

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
            st.download_button(
                "📥 Export CSV",
                data=csv,
                file_name=f"top_sellers_{start_date}_{end_date}.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Eroare la citire v_sales_monthly: {e}")

with tab_velocity:
    st.info("📈 Velocity/rotație va folosi mișcările pe cantități (intrări/ieșiri). Îl adăugăm după ce confirmi că Top sellers arată corect.")

with tab_reco:
    st.info("🧠 Recomandările (ce să cumperi / cu cât să vinzi) vor combina Top sellers + stoc curent + ținte de marjă.")
# ---- END APP.PY (ServicePack Reports) ----
