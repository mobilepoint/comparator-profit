# -*- coding: utf-8 -*-
import base64
import datetime as dt
import math
import re

import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Comparator Profit pe Produs", layout="wide")
TODAY = dt.date(2025, 9, 24)  # dacă vrei automat, setează la: dt.date.today()
GIFT_KEYWORDS = ("woorewards-freeproduct", "freeproduct", "cupon")

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def download_link_df(df: pd.DataFrame, filename: str, label: str) -> None:
    csv = df.to_csv(index=False).encode("utf-8-sig")
    href = f'data:text/csv;base64,{base64.b64encode(csv).decode()}'
    st.markdown(f'<a href="{href}" download="{filename}">{label}</a>', unsafe_allow_html=True)

def clean_number(val):
    """Încearcă să convertească string-uri cu . și , (format EU sau US) în float."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 0.0
    if isinstance(val, (int, float, np.number)):
        return float(val)
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return 0.0
    s = s.replace("\xa0", "").replace(" ", "")
    # Dacă are atât . cât și , – determină care e separatorul zecimal
    if "." in s and "," in s:
        if s.rfind(",") > s.rfind("."):
            # format EU: 1.234,56
            s = s.replace(".", "").replace(",", ".")
        else:
            # format US cu virgule de mii: 1,234.56
            s = s.replace(",", "")
    else:
        # doar virgulă → zecimală
        if "," in s and "." not in s:
            s = s.replace(",", ".")
        # doar punct → rămâne așa
    try:
        return float(s)
    except Exception:
        return 0.0

def extract_sku(name: str):
    """SKU = ultimul text dintre paranteze la finalul denumirii."""
    if not isinstance(name, str):
        return None
    m = re.search(r"\(([^()]*)\)\s*$", name.strip())
    return m.group(1).strip().upper() if m else None

def is_gift(name: str) -> bool:
    if not isinstance(name, str):
        return False
    n = name.lower()
    return any(k in n for k in GIFT_KEYWORDS)

def find_cols(df: pd.DataFrame):
    """
    Returnează (produs_col, vanzari_col, cost_col).
    Folosește mai întâi nume, apoi fallback pe poziții (A, D, E).
    """
    cols = list(df.columns)
    lower = [str(c).strip().lower() for c in cols]

    def pick(cands, fallback_index):
        for i, nm in enumerate(lower):
            if any(c in nm for c in cands):
                return cols[i]
        return cols[fallback_index] if fallback_index < len(cols) else None

    produs = pick(["produsul", "produs", "denumire"], 0)
    vanz   = pick(["vânzări nete", "vanzari nete", "incasari", "încasări"], 3)
    cost   = pick(["costul bunurilor", "cost", "cogs"], 4)
    if produs is None or vanz is None or cost is None:
        raise ValueError("Nu am găsit coloanele necesare (A/D/E). Verifică headerele.")
    return produs, vanz, cost

@st.cache_data(show_spinner=False)
def load_table(uploaded_file) -> pd.DataFrame:
    """Încarcă CSV/XLSX. Nu acceptă .xls (vechi) – convertește la .xlsx."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        try:
            # încercăm auto-delimiter
            return pd.read_csv(uploaded_file, sep=None, engine="python")
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file)
    if name.endswith((".xlsx", ".xlsm", ".xltx", ".xltm")):
        return pd.read_excel(uploaded_file, engine="openpyxl")
    if name.endswith(".xls"):
        raise ValueError("Fișierele .xls nu sunt suportate. Te rog convertește la .xlsx sau exportă CSV.")
    raise ValueError("Format neacceptat. Folosește CSV sau XLSX.")

def prepare_year_df(uploaded_file, year: int, exclude_gifts: bool) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame(columns=["year","sku","produs","vanzari_nete","cost","profit","marja_pct"])
    raw = load_table(uploaded_file)
    produs_col, vanz_col, cost_col = find_cols(raw)
    df = raw[[produs_col, vanz_col, cost_col]].copy()
    df.columns = ["produs", "vanzari_nete", "cost"]
    # curățare
    df["vanzari_nete"] = df["vanzari_nete"].map(clean_number)
    df["cost"] = df["cost"].map(clean_number)
    df["sku"] = df["produs"].map(extract_sku)

    if exclude_gifts:
        df = df[~df["produs"].map(is_gift)]

    df = df.dropna(subset=["sku"])
    # agregare pe SKU
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
    s = df_year["vanzari_nete"].sum()
    c = df_year["cost"].sum()
    p = df_year["profit"].sum()
    m = (p/s) if s>0 else np.nan
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

with st.sidebar:
    st.header("Setări")
    exclude_gifts = st.toggle("Exclude produse-cadou (woorewards/freeproduct/cupon)", value=True)
    like_for_like = st.toggle("Compară 2025 YTD cu 2024 ajustat la aceeași perioadă", value=True)
    st.caption("Aplicația folosește doar coloanele: A=Produsul, D=Vânzări nete, E=Cost.")

c1, c2, c3 = st.columns(3)
f2023 = c1.file_uploader("Raport 2023 (.csv/.xlsx)", type=["csv","xlsx","xlsm","xltx","xltm"])
f2024 = c2.file_uploader("Raport 2024 (.csv/.xlsx)", type=["csv","xlsx","xlsm","xltx","xltm"])
f2025 = c3.file_uploader("Raport 2025 până azi (.csv/.xlsx)", type=["csv","xlsx","xlsm","xltx","xltm"])

if not any([f2023, f2024, f2025]):
    st.info("Încărcă cel puțin un raport pentru a începe.")
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
# Comparație 2025 YTD vs 2024
# ──────────────────────────────────────────────────────────────────────────────
if not df2025.empty and not df2024.empty:
    ref = df2024.copy()
    label = "2024 (ajustat)" if like_for_like else "2024 (an întreg)"
    if like_for_like:
        scale = like_for_like_scaler(2024, TODAY)
        ref[["vanzari_nete","cost","profit"]] *= scale

    st.markdown("---")
    st.subheader(f"Comparație {label} vs 2025 YTD")

    comp = (
        df2025[["sku","produs","profit","marja_pct"]].rename(columns={"profit":"profit_2025","marja_pct":"marja_2025"})
        .merge(ref[["sku","profit","marja_pct"]].rename(columns={"profit":"profit_2024","marja_pct":"marja_2024"}), on="sku", how="outer")
        .fillna({"profit_2025":0.0,"profit_2024":0.0})
    )
    # păstrează denumiri lipsă (SKU doar în 2024)
    names24 = ref[["sku","produs"]].drop_duplicates()
    comp = comp.merge(names24, on="sku", how="left", suffixes=("","_y"))
    comp["produs"] = comp["produs"].fillna(comp.pop("produs_y"))

    comp["delta_profit"] = comp["profit_2025"] - comp["profit_2024"]
    worst = comp.sort_values("delta_profit").head(25)

    left, right = st.columns([2,1])
    with left:
        st.markdown("**Top 25 scăderi de profit (SKU)** – 2025 YTD vs " + label)
        show = worst[["sku","produs","profit_2024","profit_2025","delta_profit"]].copy()
        show[["profit_2024","profit_2025","delta_profit"]] = show[["profit_2024","profit_2025","delta_profit"]].round(2)
        st.dataframe(show, use_container_width=True, height=500)
        download_link_df(show, "top_scaderi_profit.csv", "⬇️ Descarcă CSV")
    with right:
        st.bar_chart(worst.set_index("sku")["delta_profit"])

    # interpretări rapide
    st.markdown("### Interpretare rapidă")
    bullets = []
    if comp["profit_2025"].sum() < comp["profit_2024"].sum():
        bullets.append("• **Profit total în scădere** – concentrează-te pe SKU-urile din tabelul de mai sus.")
    lost = comp[(comp["profit_2024"]>0) & (comp["profit_2025"]==0)].sort_values("profit_2024", ascending=False).head(10)
    if not lost.empty:
        bullets.append("• **SKU profitabile în 2024 dar lipsă în 2025** – verifică stocul, listarea și campaniile.")
    new_bad = comp[(comp["profit_2024"]==0) & (comp["profit_2025"]<0)].head(10)
    if not new_bad.empty:
        bullets.append("• **SKU noi cu profit negativ în 2025** – revizuiește costurile/prețurile sau oprește promoțiile agresive.")
    st.write("\n".join(bullets) if bullets else "• Nu se disting scăderi structurale majore din aceste fișiere.")

# ──────────────────────────────────────────────────────────────────────────────
# Tabele detaliate & Concluzii
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Detaliu pe an & Concluzii")

tab_det23, tab_det24, tab_det25, tab_conc = st.tabs(["2023", "2024", "2025 YTD", "Concluzii & Sugestii"])

def show_year_tab(container, df, year_label):
    with container:
        if df.empty:
            st.info(f"Nu există date pentru {year_label}.")
            return
        view = df.copy()
        view["marja_%"] = (view["marja_pct"]*100).round(2)
        view = view.drop(columns=["marja_pct"])
        for c in ["vanzari_nete","cost","profit"]:
            view[c] = view[c].round(2)
        st.dataframe(view[["sku","produs","vanzari_nete","cost","profit","marja_%"]], use_container_width=True, height=480)
        download_link_df(view, f"detaliu_{year_label.replace(' ','_')}.csv", f"⬇️ Descarcă {year_label} CSV")

show_year_tab(tab_det23, df2023, "2023")
show_year_tab(tab_det24, df2024, "2024")
show_year_tab(tab_det25, df2025, "2025_YTD")

with tab_conc:
    colA, colB = st.columns(2)

    # Lipsă în 2025
    with colA:
        st.markdown("#### Produse profitabile în 2024 care **lipsesc în 2025**")
        if not df2025.empty and not df2024.empty:
            lost = (
                df2024[df2024["profit"]>0]
                .merge(df2025[["sku"]], on="sku", how="left", indicator=True)
                .query("_merge == 'left_only'")
                .sort_values("profit", ascending=False)
                .head(10)
            )
            if lost.empty:
                st.success("Nicio dispariție notabilă.")
            else:
                view = lost[["sku","produs","profit","marja_pct"]].copy()
                view["marja_%"] = (view.pop("marja_pct")*100).round(2)
                view["profit"] = view["profit"].round(2)
                st.dataframe(view, use_container_width=True)
                st.markdown("👉 **Sugestie:** readu pe stoc/listare și rulează campanii pe aceste SKU-uri.")
        else:
            st.info("Încarcă 2024 și 2025 pentru această analiză.")

    # Marjă în scădere
    with colB:
        st.markdown("#### Produse cu **marjă în scădere** (2025 vs 2024)")
        if not df2025.empty and not df2024.empty:
            both = df2025.merge(df2024, on="sku", suffixes=("_25","_24"))
            if not both.empty:
                both["delta_marja_pp"] = (both["marja_pct_25"] - both["marja_pct_24"])*100
                drops = both.sort_values("delta_marja_pp").head(10)
                show = drops[["sku","produs_25","marja_pct_24","marja_pct_25","delta_marja_pp"]].copy()
                show[["marja_pct_24","marja_pct_25","delta_marja_pp"]] = show[["marja_pct_24","marja_pct_25","delta_marja_pp"]]*np.array([100,100,1])
                show = show.rename(columns={"produs_25":"produs","marja_pct_24":"marja_2024_%","marja_pct_25":"marja_2025_%","delta_marja_pp":"Δ marjă pp"})
                show[["marja_2024_%","marja_2025_%","Δ marjă pp"]] = show[["marja_2024_%","marja_2025_%","Δ marjă pp"]].round(2)
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
        v["marja_%"] = (v.pop("marja_pct")*100).round(2)
        v[["vanzari_nete","profit"]] = v[["vanzari_nete","profit"]].round(2)
        st.dataframe(v, use_container_width=True)
        st.markdown("👉 **Sugestie:** scalează bugetele pe acești SKU (Ads, oferte speciale, vizibilitate pe site).")
    else:
        st.info("Încarcă raportul 2025.")

    st.markdown("#### Profit negativ în 2025")
    if not df2025.empty:
        negative = df2025[df2025["profit"] < 0].sort_values("profit").head(10)
        if negative.empty:
            st.success("Nu ai produse cu profit negativ în 2025.")
        else:
            n = negative[["sku","produs","vanzari_nete","profit","marja_pct"]].copy()
            n["marja_%"] = (n.pop("marja_pct")*100).round(2)
            n[["vanzari_nete","profit"]] = n[["vanzari_nete","profit"]].round(2)
            st.dataframe(n, use_container_width=True)
            st.markdown("👉 **Sugestie:** mărește prețul sau oprește vânzarea acestor SKU până calibrezi costul/prețul.")
