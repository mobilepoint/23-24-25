# ---- RAW PROFIT: robust header mapping + preview ----
def detect_profit_columns(df: pd.DataFrame) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Întoarce (col_prod, col_net, col_cogs) detectate robust.
    1) Încearcă după nume normalizat.
    2) Fallback data-driven pe coloane numerice.
    """
    norm_map = _normalize_columns(df.columns)

    def find_col(tokens: list[str]) -> Optional[str]:
        for c in df.columns:
            k = norm_map[c]
            if all(t in k for t in tokens):
                return c
        return None

    # produs
    col_prod = find_col(["produs"]) or find_col(["produsul"])
    if col_prod is None and len(df.columns) >= 2:
        col_prod = df.columns[1]  # fallback: a doua coloană (B)

    # net/cogs după nume
    col_net = find_col(["vanzari", "nete"])
    col_cogs = find_col(["cost", "bunurilor", "vandute"]) or find_col(["cogs"])

    # fallback data-driven dacă lipsesc
    if col_net is None or col_cogs is None:
        # scorăm coloanele numerice: (nonnull, sumă)
        numeric_candidates = []
        for c in df.columns:
            try:
                s = df[c].apply(_to_number)
            except Exception:
                continue
            nonnull = int(s.notna().sum())
            total = float((s.fillna(0)).sum())
            # păstrăm doar coloane cu ceva semnal numeric
            if nonnull > 0 and (total != 0):
                numeric_candidates.append((c, nonnull, abs(total), total))
        # sortăm după nonnull desc, apoi sumă absolută desc
        numeric_candidates.sort(key=lambda t: (t[1], t[2]), reverse=True)

        # alegem prima drept NET dacă lipsește
        if col_net is None and numeric_candidates:
            col_net = numeric_candidates[0][0]

        # alegem următoarea diferită drept COGS dacă lipsește
        if col_cogs is None:
            for c, *_ in numeric_candidates:
                if c != col_net:
                    col_cogs = c
                    break

    return col_prod, col_net, col_cogs


def load_raw_profit_to_catalog(xls_bytes: bytes, expected_period: date, source_path: str) -> tuple[int, int, date]:
    """
    Încarcă PROFIT: staging.raw_profit + completează staging.products.
    Mapare robustă pentru 'Vânzări nete' și 'Cost bunurilor vândute'.
    """
    perioada = _extract_period_from_profit_header(xls_bytes)

    # header-ul raportului este pe rândul 11 (index 10)
    df_raw = pd.read_excel(io.BytesIO(xls_bytes), sheet_name=0, header=10)
    # elimină complet coloanele goale
    df_raw = df_raw.loc[:, ~df_raw.columns.to_series().astype(str).str.fullmatch(r"\s*nan\s*", case=False)]

    col_prod, col_net, col_cogs = detect_profit_columns(df_raw)

    if not col_prod or not col_net or not col_cogs:
        raise RuntimeError(
            f"Nu pot identifica coloanele obligatorii: Produs / Vânzări nete / COGS. "
            f"Detectate: produs={col_prod}, net={col_net}, cogs={col_cogs}"
        )

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
        raise RuntimeError(
            f"Fișierul este pentru {perioada.strftime('%Y-%m')}, dar aici acceptăm doar {expected_period.strftime('%Y-%m')}."
        )

    if base.empty:
        st.warning("După filtrare nu a rămas niciun rând valid în profit.")
        return 0, 0, perioada

    # 🔎 Preview & sanity check în UI
    total_net = float(base["net_sales_wo_vat"].fillna(0).sum())
    total_cogs = float(base["cogs_wo_vat"].fillna(0).sum())
    st.info(
        f"Mapare PROFIT → produs='{col_prod}', net='{col_net}', cogs='{col_cogs}'. "
        f"SUM(NET)={total_net:,.2f} • SUM(COGS)={total_cogs:,.2f}"
    )
    st.dataframe(base.head(10), use_container_width=True)

    # 1) completează staging.products (doar SKU-urile lipsă)
    product_rows = base[["sku", "product_text"]].to_dict(orient="records")
    inserted = _ensure_products_from_profit(sb, product_rows)
    if inserted:
        st.success(f"Adăugate {inserted} SKU noi în staging.products.")

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
# ---- END RAW PROFIT: robust header mapping + preview ----
