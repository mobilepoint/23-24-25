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

def parse_number(x) -> float:
    """ Acceptă mii cu virgulă și zecimale cu punct. Stringuri goale => 0. """
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
    """Extrage ultimul () din șir -> SKU. Ex: 'Name (X) (SKU-123)' => 'SKU-123'."""
    if not isinstance(text, str):
        text = str(text or "")
    matches = re.findall(r'\(([^()]*)\)', text)
    return matches[-1].strip() if matches else None

# --- utilitare pentru identificare robustă a coloanelor în PROFIT ---
_norm_tbl = str.maketrans({c: "" for c in " .,_-/%()[]{}:"})
def norm(s: str) -> str:
    """lower + elimină spații/punctuație (fără diacritice)."""
    return str(s).strip().lower().translate(_norm_tbl)

def find_profit_columns(df: pd.DataFrame):
    """
    Caută coloanele pentru:
      - produs (B)
      - vânzări nete (E)
      - cost bunuri vândute (F)
    Întâi după nume, apoi fallback la poziții 1,4,5 (0-based).
    """
    col_prod = col_net = col_cogs = None
    ncols = len(df.columns)

    # 1) după nume
    for c in df.columns:
        n = norm(c)
        if col_prod is None and ("produs" in n):
            col_prod = c
        if col_net is None and ("vanzari" in n and "nete" in n):
            col_net = c
        if col_cogs is None and (("cost" in n and ("bunurilor" in n or "bunuri" in n) and ("vandute" in n or "vandut" in n)) or "cogs" in n):
            col_cogs = c

    # 2) fallback pe poziții (format fix din raport)
    if col_prod is None and ncols >= 2:
        col_prod = df.columns[1]  # B
    if col_net is None and ncols >= 5:
        col_net = df.columns[4]   # E
    if col_cogs is None and ncols >= 6:
        col_cogs = df.columns[5]  # F

    return col_prod, col_net, col_cogs

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
        # 1) Citim antetul și detectăm perioada
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

        # 2) Permitem upload doar pentru ultima lună încheiată
        if period != lcm:
            st.error(f"Fișierul este pentru {period.strftime('%Y-%m')}, dar aici acceptăm doar **{lcm.strftime('%Y-%m')}**.")
            st.stop()

        rows_json = []
        try:
            if file_type == "Profit pe produs":
                # 3) Citire full cu fallback (skip primele 10 rânduri; rândul 11 e header)
                df = read_full_any(uploaded_file, skiprows=10)

                # 4) Identificare coloane robustă + fallback la poziții
                col_prod, col_net, col_cogs = find_profit_columns(df)
                if col_prod is None or col_net is None or col_cogs is None:
                    raise ValueError("Nu am reușit să identific coloanele pentru Produs / Vanzari nete / Cost bunuri vandute.")

                # 5) Transformare în JSON pentru RPC
                for _, r in df.iterrows():
                    sku = extract_last_parenthesized(r[col_prod])
                    if not sku:
                        continue
                    net_sales = parse_number(r[col_net])
                    cogs = parse_number(r[col_cogs])
                    rows_json.append({"sku": sku, "net_sales_wo_vat": net_sales, "cogs_wo_vat": cogs})

                if not rows_json:
                    raise ValueError("Nu am extras niciun rând valid (SKU).")

                # 6) Apel RPC
                res = sb.rpc("load_profit_file", {
                    "p_period": period.isoformat(),
                    "p_source_path": uploaded_file.name,
                    "p_rows": rows_json
                }).execute()
                st.success(f"Încărcat PROFIT pentru {period.strftime('%Y-%m')}. file_id: {res.data}")

            else:  # Mișcări stocuri
                # 3) Citire full cu fallback (skip primele 9 rânduri; rândul 10 e header)
                df = read_full_any(uploaded_file, skiprows=9)

                # 4) Identificare coloane (flexibil, ca în screenshot)
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

                # 5) Transformare în JSON pentru RPC
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

                # 6) Apel RPC + balanțe
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

    st.divider()
    st.caption("După ce ai încărcat **ambele** fișiere pentru luna acceptată și nu ai erori, folosește tabul „Consolidare & Rapoarte”.")

# ---------- TAB CONSOLIDARE & Rapoarte ----------
with tab_consol:
    st.subheader(f"Consolidare pentru {lcm.strftime('%Y-%m')}")

    c1, c2, c3 = st.columns([1,1,2])
    TOL_QTY = c1.number_input("Toleranță cantități (buc)", min_value=0.0, value=0.01, step=0.01, format="%.2f")
    TOL_VAL = c2.number_input("Toleranță valori (lei fără TVA)", min_value=0.0, value=5.00, step=0.10, format="%.2f")
    overwrite = c3.checkbox("Forțează overwrite luna curentă în core/mart", value=True)

    # citim statusul din registry
    row = get_period_row(sb, lcm)

    # citim și rezumatul de balanță din view (pt. toleranțe)
    sum_resp = sb.table("v_balance_summary").select("*").eq("period_month", lcm.isoformat()).execute()
    sum_df = pd.DataFrame(sum_resp.data) if sum_resp.data else pd.DataFrame()
    total_diff_qty = float(sum_df["total_diff_qty"].iloc[0]) if not sum_df.empty else None
    total_diff_val = float(sum_df["total_diff_val"].iloc[0]) if not sum_df.empty else None

    if not row:
        st.warning("Nu există încă intrări pentru această lună în registru. Încarcă mai întâi fișierele în tabul „Upload”.")
    else:
        # booleene „oficiale”
        profit_ok  = bool(row.get("profit_loaded"))
        miscari_ok = bool(row.get("miscari_loaded"))
        qty_ok_reg = bool(row.get("balance_ok_qty"))
        val_ok_reg = bool(row.get("balance_ok_val"))

        # booleene „efective” cu toleranțe
        qty_ok_eff = qty_ok_reg or (total_diff_qty is not None and abs(total_diff_qty) <= TOL_QTY)
        val_ok_eff = val_ok_reg or (total_diff_val is not None and abs(total_diff_val) <= TOL_VAL)

        cols = st.columns(4)
        cols[0].metric("Profit încărcat?", "DA" if profit_ok else "NU")
        cols[1].metric("Mișcări încărcate?", "DA" if miscari_ok else "NU")
        cols[2].metric(
            "Balanță cantități OK?",
            "DA" if qty_ok_eff else "NU",
            delta=None if total_diff_qty is None else f"Δ {total_diff_qty:.2f}"
        )
        cols[3].metric(
            "Balanță valori OK?",
            "DA" if val_ok_eff else "NU",
            delta=None if total_diff_val is None else f"Δ {total_diff_val:.2f}"
        )

        ready = all([profit_ok, miscari_ok, qty_ok_eff, val_ok_eff])

        if row.get("consolidated_to_core"):
            st.success("✅ Luna este deja consolidată. Rapoartele pot folosi `mart.sales_monthly`.")
        elif not ready:
            st.error("Nu poți consolida încă. Asigură-te că ambele fișiere sunt încărcate și balanțele sunt în toleranțe.")
        else:
            if st.button("🚀 Consolidează luna"):
                try:
                    # 1) dacă vrem overwrite, curățăm LUNA din tabelele core (profit + mișcări) în mod verificabil
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

                    # 2) consolidarea tolerantă (nu mai lăsăm funcția să șteargă, am purjat noi)
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

    st.divider()
    st.subheader("Rapoarte")
    if row and (row.get("consolidated_to_core") or ready):
        st.success("Rapoartele pot fi generate (urmează pagini dedicate: Top sellers, Velocity, Recomandări).")
    else:
        st.warning("Rapoartele sunt blocate până când **ultima lună încheiată** este consolidată.")
# ---------- TAB 🧪 DEBUG BALANȚE ----------



with tab_debug:
    st.subheader(f"Verifică diferențele de balanță pentru {lcm.strftime('%Y-%m')}")

    try:
        # Rezumat pe lună
        s = sb.table("v_balance_summary").select("*").eq("period_month", lcm.isoformat()).execute()
        sdf = pd.DataFrame(s.data)
        if not sdf.empty:
            sdf["period_month"] = pd.to_datetime(sdf["period_month"]).dt.date
            st.write("**Rezumat:**")
            st.dataframe(sdf, use_container_width=True)
        else:
            st.info("Nu există rezumat pentru această perioadă (posibil să nu fie încărcate mișcările).")

        # Detalii pe SKU
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
# ---------- TAB RAPOARTE: Top sellers ----------
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
            # conversie numerică defensivă (în SQL e numeric, dar poate veni ca text)
            for c in ["net_sales_wo_vat", "cogs_wo_vat", "profit_wo_vat", "margin_pct", "qty_sold"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            df["period_month"] = pd.to_datetime(df["period_month"]).dt.to_period("M").dt.to_timestamp()

            # agregare pe SKU
            ag = (
                df.groupby("sku", as_index=False)
                  .agg(
                      net_sales_wo_vat=("net_sales_wo_vat", "sum"),
                      cogs_wo_vat=("cogs_wo_vat", "sum"),
                      profit_wo_vat=("profit_wo_vat", "sum"),
                      qty_sold=("qty_sold", "sum"),
                  )
            )
            # recalcul marja & preț mediu
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
# ---------- END RAPOARTE ----------
