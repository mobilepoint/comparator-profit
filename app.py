import io
import math
import base64
import datetime as dt
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------
# Config & helpers
# ----------------------
st.set_page_config(page_title="Profit pe produs – Analyzer", layout="wide")

TODAY = dt.date(2025, 9, 24)  # dacă vrei, schimbă în dt.date.today()
YEAR_LIST = [2023, 2024, 2025]

GIFT_KEYWORDS = ("woorewards-freeproduct", "freeproduct", "cupon")

def _download_link(df: pd.DataFrame, filename: str, label: str) -> str:
    csv = df.to_csv(index=False).encode("utf-8-sig")
    b64 = base64.b64encode(csv).decode()
    return f'<a href="data:text/csv;base64,{b64}" download="{filename}">{label}</a>'

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only 3 columns we need: Produsul (A), Vânzări nete (D), Cost (E).
       We try by header names first; fallback to positions A/D/E."""
    cols = [c for c in df.columns]
    cols_lower = [str(c).strip().lower() for c in cols]

    def find_col(candidates, fallback_idx):
        # search by text
        for i, name in enumerate(cols_lower):
            for cand in candidates:
                if cand in name:
                    return cols[i]
        # fallback by index (0-based)
        if fallback_idx < len(cols):
            return cols[fallback_idx]
        raise ValueError(f"Nu am găsit coloana {candidates} și fallback-ul {fallback_idx} nu există.")

    col_prod = find_col(["produsul", "produs", "denumire"], 0)
    col_net  = find_col(["vanzari nete", "vânzări nete", "incasari", "încasări"], 3)
    col_cost = find_col(["costul bunurilor vandute", "costul bunurilor vândute", "cost", "cogs"], 4)

    out = df[[col_prod, col_net, col_cost]].copy()
    out.columns = ["produs", "vanzari_nete", "cost"]
    return out

def extract_sku(name: str) -> Optional[str]:
    """SKU = ultimul text dintre paranteze rotunde, la finalul stringului."""
    if not isinstance(name, str):
        return None
    name = name.strip()
    # căutăm ultima paranteză închisă
    import re
    m = re.search(r"\(([^()]*)\)\s*$", name)
    if not m:
        return None
    sku = m.group(1).strip()
    if not sku:
        return None
    return sku.upper()

def is_gift(name: str) -> bool:
    if not isinstance(name, str):
        return False
    n = name.lower()
    return any(k in n for k in GIFT_KEYWORDS)

def load_any(file) -> pd.DataFrame:
    suffix = (file.name.split(".")[-1] or "").lower()
    if suffix == "csv":
        df = pd.read_csv(file)
    elif suffix in ("xlsx", "xlsm", "xltx", "xltm"):
        df = pd.read_excel(file, engine="openpyxl")
    else:
        st.error(f"Format neacceptat: .{suffix}. Te rog CSV sau XLSX.")
        raise ValueError("Unsupported format")
    return df

def prep_year_df(uploaded_file, year: int, exclude_gifts: bool) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame(columns=["year", "sku", "produs", "vanzari_nete", "cost", "profit", "marja_pct"])
    raw = load_any(uploaded_file)
    df = normalize_columns(raw)
    # curățări numerice (în caz de separatori de mii / virgule)
    def to_num(x):
        if pd.isna(x):
            return 0.0
        if isinstance(x, (int, float, np.number)):
            return float(x)
        s = str(x).replace("\xa0", "").replace(" ", "").replace(",", ".")
        try:
            return float(s)
        except:
            return 0.0
    df["vanzari_nete"] = df["vanzari_nete"].map(to_num)
    df["cost"] = df["cost"].map(to_num)
    df["sku"] = df["produs"].map(extract_sku)

    if exclude_gifts:
        df = df[~df["produs"].map(is_gift)]

    # aruncăm rândurile fără SKU (nu le putem agrega)
    df = df.dropna(subset=["sku"])
    # agregare pe SKU (dacă există denumiri multiple pentru același SKU)
    agg = df.groupby("sku", as_index=False).agg(
        vanzari_nete=("vanzari_nete", "sum"),
        cost=("cost", "sum"),
        # păstrăm o denumire reprezentativă (cea cu vânzări maxime)
        produs=("produs", lambda s: s.iloc[np.argmax(df.loc[s.index, "vanzari_nete"].values)])
    )
    agg["profit"] = agg["vanzari_nete"] - agg["cost"]
    agg["marja_pct"] = np.where(agg["vanzari_nete"] > 0, agg["profit"] / agg["vanzari_nete"], np.nan)
    agg["year"] = year
    return agg[["year", "sku", "produs", "vanzari_nete", "cost", "profit", "marja_pct"]]

def kpi_block(df_year: pd.DataFrame, title: str):
    total_sales = df_year["vanzari_nete"].sum()
    total_cost  = df_year["cost"].sum()
    total_profit= df_year["profit"].sum()
    margin = (total_profit / total_sales) if total_sales > 0 else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{title} • Vânzări nete", f"{total_sales:,.0f} RON")
    c2.metric(f"{title} • Cost", f"{total_cost:,.0f} RON")
    c3.metric(f"{title} • Profit", f"{total_profit:,.0f} RON")
    c4.metric(f"{title} • Marjă", f"{margin*100:,.2f} %" if not math.isnan(margin) else "—")

    return dict(vanzari=total_sales, cost=total_cost, profit=total_profit, marja=margin)

def same_period_scaler(ref_year: int, today: dt.date) -> float:
    """Raportul zilelor trecute din an / total zile; pentru a ajusta 2024 la YTD comparabil cu 2025."""
    is_leap = (ref_year % 4 == 0 and (ref_year % 100 != 0 or ref_year % 400 == 0))
    days_in_year = 366 if is_leap else 365
    doy = today.timetuple().tm_yday
    return min(doy / days_in_year, 1.0)

# ----------------------
# UI
# ----------------------
st.title("📉 Profit pe produs – comparație 2023 · 2024 · 2025 (YTD)")

st.markdown(
    "Încarcă rapoartele anuale (cu aceleași headere ca în exemplu). "
    "**Folosim doar coloanele A (Produsul), D (Vânzări nete) și E (Cost).**"
)

with st.sidebar:
    st.header("Setări")
    exclude_gifts = st.toggle("Exclude produse-cadou (woorewards/freeproduct/cupon)", value=True)
    like_for_like = st.toggle("Compară 2025 YTD cu 2024 ajustat la aceeași perioadă (like-for-like)", value=True)

    st.caption("Formate acceptate: CSV, XLSX (recomandat).")

c2023, c2024, c2025 = st.columns(3)
f2023 = c2023.file_uploader("Raport 2023", type=["csv", "xlsx", "xlsm", "xltx", "xltm"])
f2024 = c2024.file_uploader("Raport 2024", type=["csv", "xlsx", "xlsm", "xltx", "xltm"])
f2025 = c2025.file_uploader("Raport 2025 (până azi)", type=["csv", "xlsx", "xlsm", "xltx", "xltm"])

if not any([f2023, f2024, f2025]):
    st.info("Aștept fișierele…")
    st.stop()

df_2023 = prep_year_df(f2023, 2023, exclude_gifts) if f2023 else pd.DataFrame(columns=["year","sku","produs","vanzari_nete","cost","profit","marja_pct"])
df_2024 = prep_year_df(f2024, 2024, exclude_gifts) if f2024 else pd.DataFrame(columns=df_2023.columns)
df_2025 = prep_year_df(f2025, 2025, exclude_gifts) if f2025 else pd.DataFrame(columns=df_2023.columns)

all_df = pd.concat([df_2023, df_2024, df_2025], ignore_index=True)

# ----------------------
# KPI pe ani
# ----------------------
st.subheader("KPI pe ani")
kpi2023 = kpi_block(df_2023, "2023") if not df_2023.empty else None
kpi2024 = kpi_block(df_2024, "2024") if not df_2024.empty else None
kpi2025 = kpi_block(df_2025, "2025 YTD") if not df_2025.empty else None

# YoY 2025 vs 2024
if not df_2025.empty and not df_2024.empty:
    ref_2024 = df_2024.copy()
    label = "2024 (ajustat)" if like_for_like else "2024 (an întreg)"
    if like_for_like:
        scaler = same_period_scaler(2024, TODAY)
        ref_2024[["vanzari_nete","cost","profit"]] *= scaler

    st.markdown("---")
    st.subheader(f"Comparație {label} vs 2025 YTD")

    # Top „tragere în jos” profit – contribuții SKU
    comp = (
        df_2025[["sku", "produs", "profit"]].rename(columns={"profit":"profit_2025"})
        .merge(ref_2024[["sku", "profit"]].rename(columns={"profit":"profit_2024"}), on="sku", how="outer")
        .fillna({"profit_2025":0.0, "profit_2024":0.0})
    )
    # păstrăm o denumire pentru SKU-urile care există doar în 2024
    if "produs" not in comp or comp["produs"].isna().any():
        names_2024 = df_2024[["sku","produs"]].drop_duplicates()
        comp = comp.merge(names_2024, on="sku", how="left", suffixes=("","_y"))
        comp["produs"] = comp["produs"].fillna(comp.pop("produs_y"))

    comp["delta_profit"] = comp["profit_2025"] - comp["profit_2024"]
    worst = comp.sort_values("delta_profit").head(25)

    cL, cR = st.columns([2,1])
    with cL:
        st.markdown("**Top 25 scăderi de profit (SKU)** – 2025 YTD vs " + label)
        st.dataframe(
            worst.assign(
                profit_2024=lambda d: d["profit_2024"].round(2),
                profit_2025=lambda d: d["profit_2025"].round(2),
                delta_profit=lambda d: d["delta_profit"].round(2)
            )[["sku","produs","profit_2024","profit_2025","delta_profit"]],
            use_container_width=True, height=500
        )
        st.markdown(_download_link(worst[["sku","produs","profit_2024","profit_2025","delta_profit"]],
                                   "top_scaderi_profit.csv", "⬇️ Descarcă CSV"), unsafe_allow_html=True)
    with cR:
        st.bar_chart(worst.set_index("sku")["delta_profit"])

    # Rezumat cauze probabile (din datele disponibile)
    st.markdown("### Interpretare rapidă")
    bullets = []
    if comp["profit_2025"].sum() < comp["profit_2024"].sum():
        bullets.append("• **Profit total în scădere** – concentrarea pe SKU-urile cu cele mai mari scăderi de profit din tabelul de mai sus.")
    # SKU pierdute (există în 2024, lipsesc în 2025)
    lost = comp[(comp["profit_2024"]>0) & (comp["profit_2025"]==0)].sort_values("profit_2024", ascending=False).head(10)
    if not lost.empty:
        bullets.append(f"• **SKU dispărute în 2025**: {', '.join(lost['sku'].head(5))} … – verifică stoc, preț, listare, campanii.")
    # SKU noi neprofitabile
    new_bad = comp[(comp["profit_2024"]==0) & (comp["profit_2025"]<0)].head(10)
    if not new_bad.empty:
        bullets.append(f"• **SKU noi cu profit negativ**: {', '.join(new_bad['sku'].head(5))} – verifică costurile și prețurile.")
    # SKU cu marjă în scădere (doar pentru cele prezente în ambele)
    both = df_2025.merge(df_2024, on="sku", suffixes=("_25","_24"))
    if not both.empty:
        both["marja_drop"] = (both["marja_pct_25"] - both["marja_pct_24"])
        margedown = both.nsmallest(10, "marja_drop")
        if (margedown["marja_drop"]<0).any():
            bullets.append("• **Marje în scădere** pe unele SKU-uri – vezi tabelul detaliat mai jos.")
    if bullets:
        st.write("\n".join(bullets))
    else:
        st.write("• Profitul 2025 YTD este comparabil sau mai bun; nu se disting scăderi structurale majore din aceste fișiere.")

# ----------------------
# Tabele detaliate pe an
# ----------------------
st.markdown("---")
st.subheader("Detaliu pe an (agregat pe SKU)")

tabs = st.tabs(["2023", "2024", "2025 YTD"])
for t,df,y in zip(tabs, [df_2023, df_2024, df_2025], YEAR_LIST):
    with t:
        if df.empty:
            st.info(f"Nu există date pentru {y}.")
        else:
            view = df.copy()
            view["vanzari_nete"] = view["vanzari_nete"].round(2)
            view["cost"] = view["cost"].round(2)
            view["profit"] = view["profit"].round(2)
            view["marja_pct"] = (view["marja_pct"]*100).round(2)
            st.dataframe(view[["sku","produs","vanzari_nete","cost","profit","marja_pct"]]
                         .rename(columns={"marja_pct":"marja_%"}),
                         use_container_width=True, height=480)
            st.markdown(_download_link(view, f"detaliu_{y}.csv", f"⬇️ Descarcă {y} CSV"), unsafe_allow_html=True)

# ----------------------
# Sfaturi de acțiune (în baza datelor actuale)
# ----------------------
st.markdown("---")
st.subheader("Recomandări acționabile")

rec = []
# 1) Repornire SKU-uri dispărute
if not df_2025.empty and not df_2024.empty:
    comp2 = (
        df_2025[["sku", "profit"]].rename(columns={"profit":"p25"})
        .merge(df_2024[["sku", "profit"]].rename(columns={"profit":"p24"}), on="sku", how="outer")
        .fillna(0.0)
    )
    lost = comp2[(comp2["p24"]>0) & (comp2["p25"]==0)].sort_values("p24", ascending=False).head(10)
    if not lost.empty:
        rec.append("• **Readu în ofertă / evidențiază** SKU-urile care au adus profit în 2024 dar lipsesc în 2025 (top listă mai sus).")

# 2) Ajustare preț acolo unde marja a scăzut puternic
if not df_2025.empty and not df_2024.empty:
    both = df_2025.merge(df_2024, on="sku", suffixes=("_25","_24"))
    if not both.empty:
        both["marja_drop_pp"] = (both["marja_pct_25"] - both["marja_pct_24"])*100
        drop_list = both.nsmallest(10, "marja_drop_pp")
        if (drop_list["marja_drop_pp"] < 0).any():
            rec.append("• **Revizuiește prețurile/costurile** pe SKU-urile cu scădere mare de marjă (%).")

# 3) Focus campanii pe SKU cu profit/marjă bună
if not df_2025.empty:
    winners = df_2025.sort_values(["profit","marja_pct"], ascending=[False, False]).head(15)
    if not winners.empty:
        rec.append("• **Scalează bugetele** pe SKU-urile cu profit și marjă ridicate în 2025 YTD (top din tabel).")

# 4) Curățare feed / pagini
rec.append("• **Exclude din rapoarte** denumirile duplicate de cadou (deja bifat) și asigură-te că paginile produselor principale sunt cele indexate/promovate.")

st.write("\n".join(rec) if rec else "Nu există recomandări specifice fără mai multe date (ex. cantități, campanii, trafic).")
# ----------------------
# Concluzii & Sugestii (tab clar)
# ----------------------
st.markdown("---")
st.subheader("📌 Concluzii & Sugestii")

tab1, tab2, tab3, tab4 = st.tabs([
    "Produse lipsă în 2025",
    "Marjă în scădere",
    "Top winners 2025",
    "Profit negativ"
])

# 1) Produse lipsă în 2025
with tab1:
    if not df_2025.empty and not df_2024.empty:
        lost = (
            df_2024[df_2024["profit"] > 0]
            .merge(df_2025[["sku"]], on="sku", how="left", indicator=True)
            .query("_merge == 'left_only'")
            .sort_values("profit", ascending=False)
            .head(10)
        )
        if lost.empty:
            st.success("Nu există produse profitabile în 2024 care lipsesc în 2025.")
        else:
            st.warning("Top 10 produse profitabile în 2024 care lipsesc în 2025:")
            st.dataframe(lost[["sku","produs","profit","marja_pct"]], use_container_width=True)
            st.markdown("👉 Sugestie: readu aceste produse în ofertă, verifică stocul și campaniile.")
    else:
        st.info("Ai nevoie de ambele rapoarte 2024 și 2025 pentru această analiză.")

# 2) Marjă în scădere
with tab2:
    if not df_2025.empty and not df_2024.empty:
        both = df_2025.merge(df_2024, on="sku", suffixes=("_25","_24"))
        both["delta_marja_pp"] = (both["marja_pct_25"] - both["marja_pct_24"]) * 100
        drops = both.sort_values("delta_marja_pp").head(10)
        if drops.empty:
            st.success("Nu s-au găsit produse cu marjă în scădere.")
        else:
            st.error("Top 10 produse cu scădere de marjă:")
            st.dataframe(drops[["sku","produs_25","marja_pct_24","marja_pct_25","delta_marja_pp"]],
                         use_container_width=True)
            st.markdown("👉 Sugestie: ajustează prețul sau renegociază costurile pentru aceste produse.")
    else:
        st.info("Ai nevoie de ambele rapoarte 2024 și 2025 pentru această analiză.")

# 3) Winners 2025
with tab3:
    if not df_2025.empty:
        winners = df_2025.sort_values("profit", ascending=False).head(10)
        st.success("Top 10 produse cu cel mai mare profit în 2025:")
        st.dataframe(winners[["sku","produs","vanzari_nete","profit","marja_pct"]], use_container_width=True)
        st.markdown("👉 Sugestie: scalează promovarea acestor SKU-uri (bugete Ads, oferte speciale, vizibilitate pe site).")
    else:
        st.info("Nu există date pentru 2025.")

# 4) Profit negativ
with tab4:
    if not df_2025.empty:
        negative = df_2025[df_2025["profit"] < 0].sort_values("profit").head(10)
        if negative.empty:
            st.success("Nu ai produse cu profit negativ în 2025.")
        else:
            st.error("Top 10 produse cu profit negativ în 2025:")
            st.dataframe(negative[["sku","produs","vanzari_nete","profit","marja_pct"]], use_container_width=True)
            st.markdown("👉 Sugestie: renunță la aceste produse sau ajustează imediat prețul.")
    else:
        st.info("Nu există date pentru 2025.")

