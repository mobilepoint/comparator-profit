# -*- coding: utf-8 -*-
import base64
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
# Pune automat azi; dacă vrei o dată fixă (ex. 2025-09-24), comentează linia de mai jos și setează manual.
TODAY = dt.date.today()
GIFT_KEYWORDS = ("woorewards-freeproduct", "freeproduct", "cupon")

# Afișează versiunea de Python / pachete (te ajută la debug în Cloud)
st.caption(f"Python: {sys.version.split()[0]} • Streamlit: {st.__version__}")

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def download_link_df(df: pd.DataFrame, filename: str, label: str) -> None:
    """Buton de download CSV pentru un DataFrame."""
    csv = df.to_csv(index=False).encode("utf-8-sig")
    href = f"data:text/csv;base64,{base64.b64encode(csv).decode()}"
    st.markdown(f'<a href="{href}" download="{filename}">{label}</a>', unsafe_allow_html=True)

def clean_number(val) -> float:
    """Transformă valori cu formatare EU/US (puncte/virgule) în float; NaN -> 0."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 0.0
    if isinstance(val, (int, float, np.floating, np.integer)):
        return float(val)
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return 0.0
    s = s.replace("\xa0", "").replace(" ", "")
    if "." in s and "," in s:
        # ex: 1.234,56 -> 1234.56
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            # ex: 1,234.56 -> 1234.56
            s = s.replace(",", "")
    else:
        if "," in s and "." not in s:
            s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return 0.0

def extract_sku(name: str) -> str | None:
    """SKU = ultimul text dintre paranteze, la finalul denumirii."""
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
    """Returnează (col_produs, col_vanzari_nete, col_cost). Match pe nume, fallback pe poziții A/D/E."""
    cols = list(df.columns)
    lower = [str(c).strip().lower() for c in cols]

    def pick(cands: tuple[str, ...] | list[str], fallback_index: int) -> str | None:
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
def load_table(uploaded_file) -> pd.DataFrame:
    """Încarcă CSV sau XLSX (openpyxl). .xls nu e suportat (convertește la .xlsx)."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file, sep=None, engine="python")
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file)
    if name.endswith((".xlsx", ".xlsm", ".xltx", ".xltm")):
        return pd.read_excel(uploaded_file, engine="openpyxl")
    if name.endswith(".xls"):
        raise ValueError("Fișierele .xls nu sunt suportate. Convertește la .xlsx sau exportă CSV.")
    raise ValueError("Format neacceptat. Folosește CSV sau XLSX.")

def prepare_year_df(uploaded_file, year: int, exclude_gifts: bool) -> pd.DataFrame:
    """Normalizează rapoartele: Produs (A), Vânzări nete (D), Cost (E) → agregare pe SKU."""
    if uploaded_file is None:
        return pd.DataFrame(columns=["year","sku","produs","vanzari_nete","cost","profit","marja_pct"])

    raw = load_table(uploaded_file)
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
    agg["marja_pct"] = np.where(agg["vanzari_nete"] > 0, agg["profit"] / agg["vanzari_nete"], np.nan)
    agg["year"] = year
    return agg[["year","sku","produs","vanzari_nete","cost","profit","marja_pct"]]

def kpi(df_year: pd.DataFrame, label: str) -> tuple[float,float,float,float|float]:
    s = float(df_year["vanzari_nete"].sum())
    c = float(df_year["cost"].sum())
    p = float(df_year["profit"].sum())
    m = (p / s) if s > 0 else float("nan")
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

# ──────────────────────────────────────────────────────────────────────────────
# UI – Upload & setări
# ──────────────────────────────────────────────────────────────────────────────
st.title("📊 Comparator Profit pe Produs – 2023 · 2024 · 2025 (YTD)")
st.write("Încarcă rapoartele **exact cu headerele din exemplu**. Folosim doar **A (Produsul)**, **D (Vânzări nete)**, **E (Cost)**.")

with st.sidebar:
    st.header("Setări")
    exclude_gifts = st.toggle("Exclude produse-cadou (woorewards/freeproduct/cupon)", value=True)
    like_for_like = st.toggle("Compară 2025 YTD cu 2024 ajustat la aceeași perioadă", value=True)

c1, c2, c3 = st.columns(3)
f2023 = c1.file_uploader("Raport 2023 (.csv/.xlsx)", type=["csv","xlsx","xlsm","xltx","xltm"])
f2024 = c2.file_uploader("Raport 2024 (.csv/.xlsx)", type=["csv","xlsx","xlsm","xltx","xltm"])
f2025 = c3.file_uploader("Raport 2025 până azi (.csv/.xlsx)", type=["csv","xlsx","xlsm","xltx","xltm"])

if not any([f2023, f2024, f2025]):
    st.info("Încărcă cel puțin un raport.")
    st.stop()

try:
    df2023 = prepare_year_df(f2023, 2023, exclude_gifts) if f2023 else pd.DataFrame(columns=["year","sku","produs","vanzari_nete","cost","profit","marja_pct"])
    df2024 = prepare_year_df(f2024, 2024, exclude_gifts) if f2024 else pd.DataFrame(columns=df2023.columns)
    df2025 = prepare_year_df(f2025, 2025, exclude_gifts) if f2025 else pd.DataFrame(columns=df2023.columns)
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
# Comparație 2025 YTD vs 2024 și 2023 (cu opțiune like-for-like pentru ambii ani)
# ──────────────────────────────────────────────────────────────────────────────
if not df2025.empty and (not df2024.empty or not df2023.empty):

    def ref_like_for_like(df_year: pd.DataFrame, year: int, enabled: bool) -> pd.DataFrame:
        ref = df_year.copy()
        if enabled and not ref.empty:
            scale = like_for_like_scaler(year, TODAY)
            ref[["vanzari_nete", "cost", "profit"]] = ref[["vanzari_nete", "cost", "profit"]] * scale
        return ref

    ref24 = ref_like_for_like(df2024, 2024, like_for_like) if not df2024.empty else pd.DataFrame(columns=df2025.columns)
    ref23 = ref_like_for_like(df2023, 2023, like_for_like) if not df2023.empty else pd.DataFrame(columns=df2025.columns)

    label24 = "2024 (ajustat)" if like_for_like else "2024 (an întreg)"
    label23 = "2023 (ajustat)" if like_for_like else "2023 (an întreg)"

    st.markdown("---")
    st.subheader(f"Comparație {label24} & {label23} vs 2025 YTD")

    base = df2025[["sku", "produs", "profit", "marja_pct"]].rename(
        columns={"profit": "profit_2025", "marja_pct": "marja_2025"}
    )
    comp = (
        base.merge(
            ref24[["sku","profit","marja_pct"]].rename(columns={"profit":"profit_2024","marja_pct":"marja_2024"}),
            on="sku", how="outer"
        )
        .merge(
            ref23[["sku","profit","marja_pct"]].rename(columns={"profit":"profit_2023","marja_pct":"marja_2023"}),
            on="sku", how="outer"
        )
    )

    # completează denumiri lipsă
    names24 = ref24[["sku","produs"]].drop_duplicates()
    names23 = ref23[["sku","produs"]].drop_duplicates()
    comp = comp.merge(names24, on="sku", how="left", suffixes=("", "_y"))
    comp["produs"] = comp["produs"].fillna(comp.pop("produs_y"))
    comp = comp.merge(names23, on="sku", how="left", suffixes=("", "_z"))
    comp["produs"] = comp["produs"].fillna(comp.pop("produs_z"))

    # NaN → 0 pentru profit
    for col in ["profit_2025","profit_2024","profit_2023"]:
        if col not in comp: comp[col] = 0.0
    comp[["profit_2025","profit_2024","profit_2023"]] = comp[["profit_2025","profit_2024","profit_2023"]].fillna(0.0)

    # delte
    comp["delta_vs_2024"] = comp["profit_2025"] - comp["profit_2024"]
    comp["delta_vs_2023"] = comp["profit_2025"] - comp["profit_2023"]
    comp["delta_worst"]   = comp[["delta_vs_2024","delta_vs_2023"]].min(axis=1)  # cea mai mare scădere
    comp["delta_best"]    = comp[["delta_vs_2024","delta_vs_2023"]].max(axis=1)  # cea mai mare creștere

    # Selector metrica pentru grafice
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

    # TOP 100 scăderi (ordonăm crescător după delta_worst)
    worst100 = comp.sort_values("delta_worst").head(100)

    # TOP 100 creșteri (ordonăm descrescător după delta_best)
    best100  = comp.sort_values("delta_best", ascending=False).head(100)

    # Afișare tabele + grafice
    t1, t2 = st.tabs(["⬇️ Top 100 scăderi de profit", "⬆️ Top 100 creșteri de profit"])

    with t1:
        left, right = st.columns([2,1])
        with left:
            st.markdown(f"**Top 100 scăderi** – 2025 YTD vs {label24}/{label23}")
            show_w = worst100[[
                "sku","produs",
                "profit_2023","profit_2024","profit_2025",
                "delta_vs_2023","delta_vs_2024","delta_worst"
            ]].copy()
            show_w[[c for c in show_w.columns if c.startswith("profit") or c.startswith("delta")]] = \
                show_w[[c for c in show_w.columns if c.startswith("profit") or c.startswith("delta")]].round(2)
            st.dataframe(show_w, use_container_width=True, height=520)
            download_link_df(show_w, "top_100_scaderi_profit.csv", "⬇️ Descarcă CSV")
        with right:
            st.bar_chart(worst100.set_index("sku")[sel_col])

    with t2:
        left, right = st.columns([2,1])
        with left:
            st.markdown(f"**Top 100 creșteri** – 2025 YTD vs {label24}/{label23}")
            show_b = best100[[
                "sku","produs",
                "profit_2023","profit_2024","profit_2025",
                "delta_vs_2023","delta_vs_2024","delta_best"
            ]].copy()
            show_b[[c for c in show_b.columns if c.startswith("profit") or c.startswith("delta")]] = \
                show_b[[c for c in show_b.columns if c.startswith("profit") or c.startswith("delta")]].round(2)
            st.dataframe(show_b, use_container_width=True, height=520)
            download_link_df(show_b, "top_100_cresteri_profit.csv", "⬆️ Descarcă CSV")
        with right:
            st.bar_chart(best100.set_index("sku")[sel_col])

    # Interpretare rapidă
    st.markdown("### Interpretare rapidă")
    bullets = []
    tot25, tot24, tot23 = comp["profit_2025"].sum(), comp["profit_2024"].sum(), comp["profit_2023"].sum()
    if tot25 < tot24: bullets.append("• **Profit 2025 < 2024** – vezi tabul „scăderi”.")
    if tot25 < tot23: bullets.append("• **Profit 2025 < 2023** – vezi coloana Δ vs 2023.")
    lost24 = comp[(comp["profit_2024"] > 0) & (comp["profit_2025"] == 0)].nlargest(10, "profit_2024")
    lost23 = comp[(comp["profit_2023"] > 0) & (comp["profit_2025"] == 0)].nlargest(10, "profit_2023")
    if not lost24.empty: bullets.append("• **SKU profitabile în 2024 dar lipsă în 2025** – readu pe stoc/listare.")
    if not lost23.empty: bullets.append("• **SKU profitabile în 2023 dar lipsă în 2025** – potențiale reintroduceri.")
    st.write("\n".join(bullets) if bullets else "• 2025 YTD este comparabil cu 2023/2024.")

# ──────────────────────────────────────────────────────────────────────────────
# Detaliu & Concluzii
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Detaliu pe an & Concluzii")

tab23, tab24, tab25, tabConc = st.tabs(["2023", "2024", "2025 YTD", "Concluzii & Sugestii"])

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

with tabConc:
    colA, colB = st.columns(2)

    with colA:
        st.markdown("#### Produse profitabile în 2024 care **lipsesc în 2025**")
        if not df2025.empty and not df2024.empty:
            lost = (
                df2024[df2024["profit"] > 0]
                .merge(df2025[["sku"]], on="sku", how="left", indicator=True)
                .query("_merge == 'left_only'")
                .sort_values("profit", ascending=False)
                .head(10)
            )
            if lost.empty:
                st.success("Nicio dispariție notabilă.")
            else:
                view = lost[["sku","produs","profit","marja_pct"]].copy()
                view["marja_%"] = (view.pop("marja_pct") * 100).round(2)
                view["profit"] = view["profit"].round(2)
                st.dataframe(view, use_container_width=True)
                st.markdown("👉 **Sugestie:** readu pe stoc/listare și rulează campanii pe aceste SKU-uri.")
        else:
            st.info("Încarcă 2024 și 2025 pentru această analiză.")

    with colB:
        st.markdown("#### Produse cu **marjă în scădere** (2025 vs 2024)")
        if not df2025.empty and not df2024.empty:
            both = df2025.merge(df2024, on="sku", suffixes=("_25","_24"))
            if not both.empty:
                both["delta_marja_pp"] = (both["marja_pct_25"] - both["marja_pct_24"]) * 100
                drops = both.sort_values("delta_marja_pp").head(10)
                show = drops[["sku","produs_25","marja_pct_24","marja_pct_25","delta_marja_pp"]].copy()
                show = show.rename(columns={"produs_25":"produs","marja_pct_24":"marja_2024_%","marja_pct_25":"marja_2025_%","delta_marja_pp":"Δ marjă pp"})
                show[["marja_2024_%","marja_2025_%","Δ marjă pp"]] = show[["marja_2024_%","marja_2025_%","Δ marjă pp"]].applymap(lambda x: round(float(x), 2))
                st.dataframe(show, use_container_width=True)
                st.markdown("👉 **Sugestie:** ajustează prețul/renegociază costurile; atenție la promoțiile prea agresive.")
            else:
                st.success("Nu s-au găsit produse comune pentru comparație.")
        else:
            st.info("Încarcă 2024 și 2025 pentru această analiză.")

    st.markdown("#### Winners 2025 (profit mare)")
    if not df2025.empty:
        winners = df2025.sort_values(["profit","marja_pct"], ascending=[False, False]).head(10)
        v = winners[["sku","produs","vanzari_nete","profit","marja_pct"]].copy()
        v["marja_%"] = (v.pop("marja_pct") * 100).round(2)
        v[["vanzari_nete","profit"]] = v[["vanzari_nete","profit"]].round(2)
        st.dataframe(v, use_container_width=True)
        st.markdown("👉 **Sugestie:** scalează bugetele (Ads), oferte speciale, vizibilitate pe site.")
    else:
        st.info("Încarcă raportul 2025.")

    st.markdown("#### Profit negativ în 2025")
    if not df2025.empty:
        negative = df2025[df2025["profit"] < 0].sort_values("profit").head(10)
        if negative.empty:
            st.success("Nu ai produse cu profit negativ în 2025.")
        else:
            n = negative[["sku","produs","vanzari_nete","profit","marja_pct"]].copy()
            n["marja_%"] = (n.pop("marja_pct") * 100).round(2)
            n[["vanzari_nete","profit"]] = n[["vanzari_nete","profit"]].round(2)
            st.dataframe(n, use_container_width=True)
            st.markdown("👉 **Sugestie:** mărește prețul sau oprește vânzarea până calibrezi costul/prețul.")
