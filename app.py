# -*- coding: utf-8 -*-
import base64
import csv
import datetime as dt
import math
import re
import sys

import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Comparator Profit pe Produs", layout="wide")
TODAY = dt.date.today()  # dacă vrei fix 2025-09-24, setează manual.
GIFT_KEYWORDS = ("woorewards-freeproduct", "freeproduct", "cupon")

st.caption(f"Python: {sys.version.split()[0]} • Streamlit: {st.__version__}")

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def download_link_df(df: pd.DataFrame, filename: str, label: str) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    href = f"data:text/csv;base64,{base64.b64encode(csv_bytes).decode()}"
    st.markdown(f'<a href="{href}" download="{filename}">{label}</a>', unsafe_allow_html=True)

def clean_number(val) -> float:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 0.0
    if isinstance(val, (int, float, np.floating, np.integer)):
        return float(val)
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return 0.0
    s = s.replace("\xa0", "").replace(" ", "")
    if "." in s and "," in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        if "," in s and "." not in s:
            s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return 0.0

def extract_sku(name: str) -> str | None:
    if not isinstance(name, str):
        return None
    m = re.search(r"\(([^()]*)\)\s*$", name.strip())
    return m.group(1).strip().upper() if m else None

def is_gift(name: str) -> bool:
    if not isinstance(name, str):
        return False
    n = name.lower()
    return any(k in n for k in GIFT_KEYWORDS)

def find_cols(df: pd.DataFrame) -> tuple[str, str, str]:
    cols = list(df.columns)
    lower = [str(c).strip().lower() for c in cols]

    def pick(cands, fallback_index):
        for i, nm in enumerate(lower):
            if any(c in nm for c in cands):
                return cols[i]
        return cols[fallback_index] if fallback_index < len(cols) else None

    produs = pick(("produsul", "produs", "denumire"), 0)
    vanz   = pick(("vânzări nete", "vanzari nete", "incasari", "încasări"), 3)
    cost   = pick(("costul bunurilor", "costul bunurilor vândute", "costul bunurilor vandute", "cogs", "cost"), 4)
    if not all([produs, vanz, cost]):
        raise ValueError("Nu am găsit coloanele necesare (A/D/E). Verifică headerele sau pozițiile.")
    return produs, vanz, cost

@st.cache_data(show_spinner=False)
def load_table(uploaded_file, sep_choice: str | None = None) -> pd.DataFrame:
    """
    Încarcă CSV/XLSX robust.
    - CSV: detectează automat ; , \t | sau folosește alegerea din UI.
    - Excel: openpyxl.
    - .xls: respins (convertește la .xlsx).
    """
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        uploaded_file.seek(0)
        sample = uploaded_file.read(65536).decode("utf-8", errors="replace")
        uploaded_file.seek(0)

        delim_map = {"auto": None, "semicolon ;": ";", "comma ,": ",", "tab \\t": "\t", "pipe |": "|"}
        sep = delim_map.get(sep_choice or "auto", None)

        if sep is None:
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=";,|\t")
                sep = dialect.delimiter
            except Exception:
                sep = ","

        try:
            return pd.read_csv(
                uploaded_file,
                sep=sep,
                engine="python",
                quoting=csv.QUOTE_MINIMAL,
                skipinitialspace=True,
                encoding="utf-8",
                on_bad_lines="skip",
            )
        finally:
            uploaded_file.seek(0)

    if name.endswith((".xlsx", ".xlsm", ".xltx", ".xltm")):
        return pd.read_excel(uploaded_file, engine="openpyxl")

    if name.endswith(".xls"):
        raise ValueError("Fișierele .xls nu sunt suportate. Convertește la .xlsx sau exportă CSV.")

    raise ValueError("Format neacceptat. Folosește CSV sau XLSX.")

def prepare_year_df(uploaded_file, year: int, exclude_gifts: bool, sep_choice: str | None) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame(columns=["year","sku","produs","vanzari_nete","cost","profit","marja_pct"])
    raw = load_table(uploaded_file, sep_choice)
    produs_col, vanz_col, cost_col = find_cols(raw)
    df = raw[[produs_col, vanz_col, cost_col]].copy()
    df.columns = ["produs", "vanzari_nete", "cost"]

    df["vanzari_nete"] = df["vanzari_nete"].map(clean_number)
    df["cost"] = df["cost"].map(clean_number)
    df["sku"] = df["produs"].map(extract_sku)

    if exclude_gifts:
        df = df[~df["produs"].map(is_gift)]
    df = df.dropna(subset=["sku"])

    agg = (
        df.groupby("sku", as_index=False)
          .agg(vanzari_nete=("vanzari_nete","sum"),
               cost=("cost","sum"),
               produs=("produs","first"))
    )
    agg["profit"] = agg["vanzari_nete"] - agg["cost"]
    agg["marja_pct"] = np.where(agg["vanzari_nete"]>0, agg["profit"]/agg["vanzari_nete"], np.nan)
    agg["year"] = year
    return agg[["year","sku","produs","vanzari_nete","cost","profit","marja_pct"]]

def kpi(df_year: pd.DataFrame, label: str):
    s = float(df_year["vanzari_nete"].sum())
    c = float(df_year["cost"].sum())
    p = float(df_year["profit"].sum())
    m = (p/s) if s>0 else float("nan")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"{label} • Vânzări nete", f"{s:,.0f} RON")
    col2.metric(f"{label} • Cost", f"{c:,.0f} RON")
    col3.metric(f"{label} • Profit", f"{p:,.0f} RON")
    col4.metric(f"{label} • Marjă", "—" if np.isnan(m) else f"{m*100:,.2f} %")
    return s, c, p, m

def like_for_like_scaler(ref_year: int, today: dt.date) -> float:
    leap = ref_year % 4 == 0 and (ref_year % 100 != 0 or ref_year % 400 == 0)
    days_in_year = 366 if leap else 365
    doy = today.timetuple().tm_yday
    return min(doy / days_in_year, 1.0)

def ref_like_for_like(df_year: pd.DataFrame, year: int, enabled: bool) -> pd.DataFrame:
    ref = df_year.copy()
    if enabled and not ref.empty:
        scale = like_for_like_scaler(year, TODAY)
        ref[["vanzari_nete", "cost", "profit"]] = ref[["vanzari_nete", "cost", "profit"]] * scale
    return ref

# ──────────────────────────────────────────────────────────────────────────────
# UI – Upload & setări
# ──────────────────────────────────────────────────────────────────────────────
st.title("📊 Comparator Profit pe Produs – 2023 · 2024 · 2025 (YTD)")
st.write("Aplicația folosește doar **A (Produsul), D (Vânzări nete), E (Cost)**. SKU = ultimul text dintre paranteze în coloana A.")

with st.sidebar:
    st.header("Setări")
    exclude_gifts = st.toggle("Exclude produse-cadou (woorewards/freeproduct/cupon)", value=True)
    like_for_like = st.toggle("Compară YTD cu ani anteriori ajustați la aceeași perioadă (like-for-like)", value=True)
    csv_sep_choice = st.selectbox(
        "Separator CSV",
        options=["auto", "semicolon ;", "comma ,", "tab \\t", "pipe |"],
        index=0,
        help="Dacă auto dă eroare, alege manual separatorul."
    )
    st.markdown("---")
    st.subheader("Stoc la zi")
    st.caption("Încarcă un fișier cu **2 coloane**: `sku`, `stoc` (CSV/XLSX). Headerele pot fi recunoscute și case-insensitive.")
    stock_file = st.file_uploader("Stoc curent (sku, stoc)", type=["csv","xlsx","xlsm","xltx","xltm"], key="stock")

c1, c2, c3 = st.columns(3)
f2023 = c1.file_uploader("Raport 2023 (.csv/.xlsx)", type=["csv","xlsx","xlsm","xltx","xltm"])
f2024 = c2.file_uploader("Raport 2024 (.csv/.xlsx)", type=["csv","xlsx","xlsm","xltx","xltm"])
f2025 = c3.file_uploader("Raport 2025 până azi (.csv/.xlsx)", type=["csv","xlsx","xlsm","xltx","xltm"])
# Uploader stoc în zona principală (opțional, mai vizibil)
c4 = st.container()
stock_file_main = c4.file_uploader("🧱 Stoc curent (2 coloane: sku, stoc) – CSV/XLSX", type=["csv","xlsx","xlsm","xltx","xltm"], key="stock_main")

# Folosește fișierul din main dacă există; altfel pe cel din sidebar (stock_file)
stock_file = stock_file_main or stock_file

if not any([f2023, f2024, f2025]):
    st.info("Încărcă cel puțin un raport.")
    st.stop()

try:
    df2023 = prepare_year_df(f2023, 2023, exclude_gifts, csv_sep_choice) if f2023 else pd.DataFrame(columns=["year","sku","produs","vanzari_nete","cost","profit","marja_pct"])
    df2024 = prepare_year_df(f2024, 2024, exclude_gifts, csv_sep_choice) if f2024 else pd.DataFrame(columns=df2023.columns)
    df2025 = prepare_year_df(f2025, 2025, exclude_gifts, csv_sep_choice) if f2025 else pd.DataFrame(columns=df2023.columns)
except Exception as e:
    st.error(f"Eroare la încărcare: {e}")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# KPI pe ani
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("KPI pe ani")
if not df2023.empty: kpi(df2023, "2023")
if not df2024.empty: kpi(df2024, "2024")
if not df2025.empty: kpi(df2025, "2025 YTD")

# ──────────────────────────────────────────────────────────────────────────────
# Comparație 2025 YTD vs 2024 & 2023 (Top 100 scăderi/creșteri)
# ──────────────────────────────────────────────────────────────────────────────
if not df2025.empty and (not df2024.empty or not df2023.empty):
    ref24 = ref_like_for_like(df2024, 2024, like_for_like) if not df2024.empty else pd.DataFrame(columns=df2025.columns)
    ref23 = ref_like_for_like(df2023, 2023, like_for_like) if not df2023.empty else pd.DataFrame(columns=df2025.columns)

    label24 = "2024 (ajustat)" if like_for_like else "2024 (an întreg)"
    label23 = "2023 (ajustat)" if like_for_like else "2023 (an întreg)"

    st.markdown("---")
    st.subheader(f"Comparație {label24} & {label23} vs 2025 YTD")

    base = df2025[["sku","produs","profit","marja_pct"]].rename(columns={"profit":"profit_2025","marja_pct":"marja_2025"})
    comp = base.merge(
        ref24[["sku","profit","marja_pct"]].rename(columns={"profit":"profit_2024","marja_pct":"marja_2024"}),
        on="sku", how="outer"
    ).merge(
        ref23[["sku","profit","marja_pct"]].rename(columns={"profit":"profit_2023","marja_pct":"marja_2023"}),
        on="sku", how="outer"
    )

    # Denumiri lipsă din 2024/2023
    for ref_names in [ref24, ref23]:
        names = ref_names[["sku","produs"]].drop_duplicates()
        comp = comp.merge(names, on="sku", how="left", suffixes=("", "_y"))
        comp["produs"] = comp["produs"].fillna(comp.pop("produs_y"))

    for col in ["profit_2025","profit_2024","profit_2023"]:
        if col not in comp: comp[col] = 0.0
    comp[["profit_2025","profit_2024","profit_2023"]] = comp[["profit_2025","profit_2024","profit_2023"]].fillna(0.0)

    comp["delta_vs_2024"] = comp["profit_2025"] - comp["profit_2024"]
    comp["delta_vs_2023"] = comp["profit_2025"] - comp["profit_2023"]
    comp["delta_worst"]   = comp[["delta_vs_2024","delta_vs_2023"]].min(axis=1)
    comp["delta_best"]    = comp[["delta_vs_2024","delta_vs_2023"]].max(axis=1)

    metric_choice = st.radio(
        "Metrică pentru grafice",
        ["Δ vs 2024", "Δ vs 2023", "Worst (cea mai mare scădere)", "Best (cea mai mare creștere)"],
        horizontal=True
    )
    metric_map = {
        "Δ vs 2024": "delta_vs_2024",
        "Δ vs 2023": "delta_vs_2023",
        "Worst (cea mai mare scădere)": "delta_worst",
        "Best (cea mai mare creștere)": "delta_best",
    }
    sel_col = metric_map[metric_choice]

    worst100 = comp.sort_values("delta_worst").head(100)
    best100  = comp.sort_values("delta_best", ascending=False).head(100)

    t1, t2 = st.tabs(["⬇️ Top 100 scăderi de profit (vs 24/23)", "⬆️ Top 100 creșteri de profit (vs 24/23)"])

    with t1:
        left, right = st.columns([2,1])
        with left:
            show_w = worst100[[
                "sku","produs",
                "profit_2023","profit_2024","profit_2025",
                "delta_vs_2023","delta_vs_2024","delta_worst"
            ]].copy()
            show_w[[c for c in show_w.columns if c.startswith("profit") or c.startswith("delta")]] = \
                show_w[[c for c in show_w.columns if c.startswith("profit") or c.startswith("delta")]].round(2)
            st.dataframe(show_w, use_container_width=True, height=520)
            download_link_df(show_w, "top_100_scaderi_profit_25_vs_24_23.csv", "⬇️ Descarcă CSV")
        with right:
            st.bar_chart(worst100.set_index("sku")[sel_col])

    with t2:
        left, right = st.columns([2,1])
        with left:
            show_b = best100[[
                "sku","produs",
                "profit_2023","profit_2024","profit_2025",
                "delta_vs_2023","delta_vs_2024","delta_best"
            ]].copy()
            show_b[[c for c in show_b.columns if c.startswith("profit") or c.startswith("delta")]] = \
                show_b[[c for c in show_b.columns if c.startswith("profit") or c.startswith("delta")]].round(2)
            st.dataframe(show_b, use_container_width=True, height=520)
            download_link_df(show_b, "top_100_cresteri_profit_25_vs_24_23.csv", "⬆️ Descarcă CSV")
        with right:
            st.bar_chart(best100.set_index("sku")[sel_col])

# ──────────────────────────────────────────────────────────────────────────────
# Comparație 2024 vs 2023 (Top 100 scăderi/creșteri)
# ──────────────────────────────────────────────────────────────────────────────
if not df2024.empty and not df2023.empty:
    st.markdown("---")
    st.subheader("Comparație 2024 vs 2023")

    comp_2423 = (
        df2024[["sku","produs","profit"]].rename(columns={"profit":"profit_2024"})
        .merge(df2023[["sku","profit"]].rename(columns={"profit":"profit_2023"}), on="sku", how="outer")
        .fillna({"profit_2024":0.0,"profit_2023":0.0})
    )
    # completează denumiri pentru SKU doar din 2023
    names23 = df2023[["sku","produs"]].drop_duplicates()
    comp_2423 = comp_2423.merge(names23, on="sku", how="left", suffixes=("","_y"))
    comp_2423["produs"] = comp_2423["produs"].fillna(comp_2423.pop("produs_y"))

    comp_2423["delta_24_vs_23"] = comp_2423["profit_2024"] - comp_2423["profit_2023"]

    worst100_2423 = comp_2423.sort_values("delta_24_vs_23").head(100)
    best100_2423  = comp_2423.sort_values("delta_24_vs_23", ascending=False).head(100)

    t1, t2 = st.tabs(["⬇️ Top 100 scăderi (2024 vs 2023)", "⬆️ Top 100 creșteri (2024 vs 2023)"])
    with t1:
        left, right = st.columns([2,1])
        with left:
            view_w = worst100_2423[["sku","produs","profit_2023","profit_2024","delta_24_vs_23"]].round(2)
            st.dataframe(view_w, use_container_width=True, height=520)
            download_link_df(view_w, "top_100_scaderi_profit_24_vs_23.csv", "⬇️ Descarcă CSV")
        with right:
            st.bar_chart(worst100_2423.set_index("sku")["delta_24_vs_23"])

    with t2:
        left, right = st.columns([2,1])
        with left:
            view_b = best100_2423[["sku","produs","profit_2023","profit_2024","delta_24_vs_23"]].round(2)
            st.dataframe(view_b, use_container_width=True, height=520)
            download_link_df(view_b, "top_100_cresteri_profit_24_vs_23.csv", "⬆️ Descarcă CSV")
        with right:
            st.bar_chart(best100_2423.set_index("sku")["delta_24_vs_23"])

# ──────────────────────────────────────────────────────────────────────────────
# Stoc la zi: „Best-sellers fără stoc acum”
# ──────────────────────────────────────────────────────────────────────────────
def load_stock_df(stock_file, sep_choice: str | None) -> pd.DataFrame:
    if stock_file is None:
        return pd.DataFrame(columns=["sku","stoc"])
    name = stock_file.name.lower()
    df = None
    if name.endswith(".csv"):
        df = load_table(stock_file, sep_choice)
    elif name.endswith((".xlsx",".xlsm",".xltx",".xltm")):
        df = pd.read_excel(stock_file, engine="openpyxl")
    else:
        raise ValueError("Format stoc neacceptat. Folosește CSV/XLSX.")

    # încearcă să identifice coloanele sku/stoc
    cols = [c for c in df.columns]
    low = [str(c).strip().lower() for c in cols]

    def pick(name_candidates, fallback_idx):
        for i, nm in enumerate(low):
            if any(c == nm for c in name_candidates) or any(c in nm for c in name_candidates):
                return cols[i]
        return cols[fallback_idx] if fallback_idx < len(cols) else None

    c_sku  = pick(["sku","cod","cod produs","product code"], 0)
    c_stoc = pick(["stoc","stock","qty","cantitate"], 1)
    if c_sku is None or c_stoc is None:
        raise ValueError("Fișierul de stoc trebuie să aibă 2 coloane: sku, stoc (în orice ordine).")
    out = df[[c_sku, c_stoc]].copy()
    out.columns = ["sku","stoc"]
    out["sku"] = out["sku"].astype(str).str.strip().str.upper()
    out["stoc"] = out["stoc"].map(clean_number)
    return out

if stock_file is not None:
    try:
        stock_df = load_stock_df(stock_file, csv_sep_choice)
    except Exception as e:
        st.error(f"Eroare la încărcarea stocului: {e}")
        stock_df = pd.DataFrame(columns=["sku","stoc"])
else:
    stock_df = pd.DataFrame(columns=["sku","stoc"])

if not stock_df.empty and (not df2025.empty or not df2024.empty or not df2023.empty):
    st.markdown("---")
    st.subheader("🔥 Best-sellers fără stoc acum")

    # „s-a vândut bine în trecut” = profit sau vânzări > 0 în 2023/2024
    hist = pd.concat([
        df2023[["sku","profit"]].rename(columns={"profit":"p23"}) if not df2023.empty else pd.DataFrame(columns=["sku","p23"]),
        df2024[["sku","profit"]].rename(columns={"profit":"p24"}) if not df2024.empty else pd.DataFrame(columns=["sku","p24"])
    ], axis=0).groupby("sku", as_index=False).sum()

    # „s-a vândut și anul acesta” = vânzări/profit > 0 în 2025 YTD
    sold25 = df2025[["sku","produs","vanzari_nete","profit"]].rename(columns={"vanzari_nete":"sales25","profit":"p25"}) if not df2025.empty else pd.DataFrame(columns=["sku","produs","sales25","p25"])

    # join cu stoc
    view = (
        sold25.merge(hist, on="sku", how="left")
              .merge(stock_df, on="sku", how="left")
    )
    view[["p23","p24"]] = view[["p23","p24"]].fillna(0.0)
    view["hist_profit"] = view["p23"] + view["p24"]

    # filtre: vândut în trecut (profit pozitiv) & vândut în 2025 & stoc == 0
    out_of_stock = view[
        (view["hist_profit"] > 0) &
        ((view["sales25"] > 0) | (view["p25"] > 0)) &
        (view["stoc"].fillna(0) <= 0)
    ].copy()

    if out_of_stock.empty:
        st.success("Nu există best-sellers fără stoc conform criteriilor (au vândut în 2023/2024 și 2025).")
    else:
        # adaugă denumire din 2025; dacă lipsește, încearcă din 2024/2023
        if out_of_stock["produs"].isna().any():
            names24 = df2024[["sku","produs"]].drop_duplicates()
            names23 = df2023[["sku","produs"]].drop_duplicates()
            out_of_stock = out_of_stock.merge(names24, on="sku", how="left", suffixes=("","_24"))
            out_of_stock["produs"] = out_of_stock["produs"].fillna(out_of_stock.pop("produs_24"))
            out_of_stock = out_of_stock.merge(names23, on="sku", how="left", suffixes=("","_23"))
            out_of_stock["produs"] = out_of_stock["produs"].fillna(out_of_stock.pop("produs_23"))

        cols = ["sku","produs","hist_profit","sales25","p25","stoc"]
        out = out_of_stock[cols].copy()
        out = out.sort_values(["hist_profit","sales25","p25"], ascending=[False, False, False])
        out[["hist_profit","sales25","p25","stoc"]] = out[["hist_profit","sales25","p25","stoc"]].round(2)

        st.dataframe(out, use_container_width=True, height=520)
        st.markdown("👉 **Sugestie:** readu urgent în stoc aceste SKU-uri (au istoric bun și tracțiune în 2025).")
        download_link_df(out, "best_sellers_fara_stoc.csv", "⬇️ Descarcă CSV")

# ──────────────────────────────────────────────────────────────────────────────
# Detaliu pe an & Concluzii (tabele brute)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Detaliu pe an")

tab23, tab24, tab25 = st.tabs(["2023", "2024", "2025 YTD"])

def show_year_tab(container, df, year_label):
    with container:
        if df.empty:
            st.info(f"Nu există date pentru {year_label}.")
            return
        view = df.copy()
        view["marja_%"] = (view["marja_pct"] * 100).round(2)
        view = view.drop(columns=["marja_pct"])
        for c in ("vanzari_nete","cost","profit"):
            view[c] = view[c].round(2)
        st.dataframe(view[["sku","produs","vanzari_nete","cost","profit","marja_%"]], use_container_width=True, height=460)
        download_link_df(view, f"detaliu_{year_label.replace(' ','_')}.csv", f"⬇️ Descarcă {year_label} CSV")

show_year_tab(tab23, df2023, "2023")
show_year_tab(tab24, df2024, "2024")
show_year_tab(tab25, df2025, "2025_YTD")
