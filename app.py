"""E-commerce Customer Behaviour Analytics - Streamlit dashboard
HCLTech Career Shaper Programme | Project No. 51
Run locally:  streamlit run app.py   (dataset.csv must be in the same folder)
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

st.set_page_config(page_title="E-commerce Customer Behaviour Analytics", page_icon="🛒", layout="wide")

PAGES = ["Home", "Category", "Product", "Cart", "Checkout", "Confirmation"]
COL = {"blue": "#1F4E79", "teal": "#2A9D8F", "amber": "#E9A23B", "red": "#C8553D", "grey": "#8D99AE"}


@st.cache_data
def load_data(path="dataset.csv"):
    d = pd.read_csv(path)
    d["page_type"] = pd.Categorical(d.page_type.str.strip(), PAGES, ordered=True)
    d["device_type"] = d.device_type.str.strip().str.title()
    d["flag_purchase_without_cart"] = ((d.purchased == 1) & (d.add_to_cart == 0)).astype(int)
    d["flag_confirmation_no_purchase"] = ((d.page_type == "Confirmation") & (d.purchased == 0)).astype(int)
    d["flag_purchase_on_early_page"] = ((d.purchased == 1) & d.page_type.isin(["Home", "Category", "Product"])).astype(int)
    d["flag_any_inconsistency"] = d[[c for c in d.columns if c.startswith("flag_")]].max(axis=1)
    d["time_bucket"] = pd.cut(d.time_on_page_s, [0, 60, 120, 180, 240, 300],
                              labels=["0-60s", "61-120s", "121-180s", "181-240s", "241-300s"])
    d["session_outcome"] = np.select(
        [(d.purchased == 1) & (d.add_to_cart == 1), (d.purchased == 1) & (d.add_to_cart == 0), d.add_to_cart == 1],
        ["Cart + Purchase", "Purchase (no cart)", "Cart abandoned"], "Browse only")
    return d


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0, 0
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return p, max(0, c - h), min(1, c + h)


def rate_table(df, by):
    rows = []
    for g, x in df.groupby(by, observed=True):
        for m in ["add_to_cart", "purchased"]:
            p, lo, hi = wilson(x[m].sum(), len(x))
            rows.append({by: str(g), "metric": "Add-to-cart" if m == "add_to_cart" else "Purchase",
                         "rate": p, "lo": lo, "hi": hi, "n": len(x)})
    return pd.DataFrame(rows)


def ci_bar(t, by, title):
    fig = px.bar(t, x=by, y="rate", color="metric", barmode="group", text=t.rate.map("{:.1%}".format),
                 color_discrete_map={"Add-to-cart": COL["teal"], "Purchase": COL["amber"]}, title=title)
    for m, tr in zip(["Add-to-cart", "Purchase"], fig.data):
        s = t[t.metric == m]
        tr.error_y = dict(type="data", symmetric=False, array=(s.hi - s.rate).values, arrayminus=(s.rate - s.lo).values)
    fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Rate (95% Wilson CI)", xaxis_title="", legend_title="")
    return fig


try:
    data = load_data()
except FileNotFoundError:
    st.error("dataset.csv not found. Keep dataset.csv in the same folder as app.py.")
    st.stop()

# ---------------- sidebar ----------------
st.sidebar.header("Filters")
dev = st.sidebar.multiselect("Device", sorted(data.device_type.unique()), default=sorted(data.device_type.unique()))
pg = st.sidebar.multiselect("Page type", PAGES, default=PAGES)
tmin, tmax = st.sidebar.slider("Time on page (s)", 5, 300, (5, 300))
clean_only = st.sidebar.checkbox("Exclude logically inconsistent sessions", value=False,
                                 help="Removes sessions flagged by the validation rules (purchase without cart, etc.)")
df = data[data.device_type.isin(dev) & data.page_type.isin(pg) & data.time_on_page_s.between(tmin, tmax)]
if clean_only:
    df = df[df.flag_any_inconsistency == 0]
st.sidebar.caption(f"{len(df)} of {len(data)} sessions selected")
st.sidebar.caption("Data: HCLTech Career Shaper dataset.csv (500 sessions). user_id is pseudonymous (DPDP Act 2023).")

# ---------------- header + KPIs ----------------
st.title("🛒 E-commerce Customer Behaviour Analytics")
st.caption("HCLTech Career Shaper Programme · III Year Industry Project No. 51 · Funnel, user journeys and purchase drivers")
if df.empty:
    st.warning("No sessions match the current filters.")
    st.stop()

n = len(df); cart = int(df.add_to_cart.sum()); purch = int(df.purchased.sum())
both = int(((df.add_to_cart == 1) & (df.purchased == 1)).sum())
k = st.columns(5)
k[0].metric("Sessions", f"{n}")
k[1].metric("Add-to-cart rate", f"{cart / n:.1%}", help="Reference target: 35%")
k[2].metric("Cart → purchase", f"{(both / cart if cart else 0):.1%}", help="Reference target: 68%")
k[3].metric("Purchase rate (flag)", f"{purch / n:.1%}")
k[4].metric("Inconsistent sessions", f"{df.flag_any_inconsistency.mean():.1%}")

tabs = st.tabs(["Funnel", "Page & device", "Time on page", "Data quality", "Statistical tests", "Data"])

# ---------------- funnel ----------------
with tabs[0]:
    fig = go.Figure(go.Funnel(y=["Sessions", "Added to cart", "Cart + purchased"], x=[n, cart, both],
                              textinfo="value+percent previous", marker_color=[COL["blue"], COL["teal"], COL["amber"]]))
    fig.update_layout(title="Conversion funnel", height=380)
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(fig, width="stretch")
    oc = df.session_outcome.value_counts().reindex(["Browse only", "Cart abandoned", "Cart + Purchase", "Purchase (no cart)"]).fillna(0).reset_index()
    oc.columns = ["outcome", "sessions"]
    c2.plotly_chart(px.bar(oc, x="sessions", y="outcome", orientation="h", title="Session outcome mix",
                           color="outcome", color_discrete_sequence=[COL["grey"], COL["teal"], COL["amber"], COL["red"]]).update_layout(showlegend=False, yaxis_title="", height=380), width="stretch")
    st.info("Add-to-cart matches the 35% reference, but cart → purchase is far below the 68% reference. "
            "Sessions marked purchased without an add-to-cart are shown separately because they contradict the funnel logic.")

# ---------------- page & device ----------------
with tabs[1]:
    c1, c2 = st.columns(2)
    c1.plotly_chart(ci_bar(rate_table(df, "page_type"), "page_type", "By page type"), width="stretch")
    c2.plotly_chart(ci_bar(rate_table(df, "device_type"), "device_type", "By device"), width="stretch")
    hm = st.radio("Heatmap metric", ["purchased", "add_to_cart"], horizontal=True)
    pv = df.pivot_table(index="page_type", columns="device_type", values=hm, aggfunc="mean", observed=True)
    st.plotly_chart(px.imshow(pv, text_auto=".0%", color_continuous_scale="YlOrBr", aspect="auto",
                              title=f"{hm} rate: page type × device (small cells, interpret with care)"), width="stretch")

# ---------------- time ----------------
with tabs[2]:
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.histogram(df, x="time_on_page_s", nbins=30, title="Time on page distribution",
                                 color_discrete_sequence=[COL["blue"]]), width="stretch")
    out = st.radio("Compare by", ["purchased", "add_to_cart"], horizontal=True)
    c2.plotly_chart(px.box(df, x=out, y="time_on_page_s", color=out, title=f"Time on page by {out}",
                           color_discrete_sequence=[COL["grey"], COL["amber"]]).update_layout(showlegend=False), width="stretch")
    tb = df.groupby("time_bucket", observed=True).agg(sessions=("session_id", "count"), add_to_cart=("add_to_cart", "mean"), purchased=("purchased", "mean"))
    st.dataframe(tb.style.format({"add_to_cart": "{:.1%}", "purchased": "{:.1%}"}), width="stretch")

# ---------------- data quality ----------------
with tabs[3]:
    dq = pd.DataFrame({"rule": ["Purchase without add-to-cart", "Confirmation page but no purchase", "Purchase on Home/Category/Product page"],
                       "sessions": [df.flag_purchase_without_cart.sum(), df.flag_confirmation_no_purchase.sum(), df.flag_purchase_on_early_page.sum()]})
    dq["share"] = dq.sessions / n
    st.plotly_chart(px.bar(dq, x="sessions", y="rule", orientation="h", text=dq.share.map("{:.1%}".format),
                           color_discrete_sequence=[COL["red"]], title="Logical inconsistencies").update_layout(yaxis_title=""), width="stretch")
    st.write(f"**{int(df.flag_any_inconsistency.sum())} of {n} selected sessions ({df.flag_any_inconsistency.mean():.1%})** break at least one rule. "
             "They are flagged, not deleted; use the sidebar checkbox to exclude them.")

# ---------------- stats ----------------
with tabs[4]:
    rows = []
    for col in ["page_type", "device_type"]:
        for t in ["add_to_cart", "purchased"]:
            ct = pd.crosstab(df[col], df[t])
            if ct.shape[0] > 1 and ct.shape[1] > 1:
                chi, p, dof, _ = stats.chi2_contingency(ct)
                rows.append([f"{col} vs {t}", "Chi-square", f"{chi:.2f} (df={dof})", round(p, 3), "Yes" if p < .05 else "No"])
    for t in ["add_to_cart", "purchased"]:
        a, b = df[df[t] == 1].time_on_page_s, df[df[t] == 0].time_on_page_s
        if len(a) > 1 and len(b) > 1:
            u = stats.mannwhitneyu(a, b)
            rows.append([f"time_on_page_s vs {t}", "Mann-Whitney U", f"{u.statistic:.0f}", round(u.pvalue, 3), "Yes" if u.pvalue < .05 else "No"])
    st.dataframe(pd.DataFrame(rows, columns=["Hypothesis", "Test", "Statistic", "p-value", "Significant at 0.05?"]), width="stretch", hide_index=True)
    st.caption("On the full dataset no test is significant, and cross-validated models score ROC-AUC 0.47–0.52 (no better than chance). "
               "Tests here recompute on the filtered selection, so very small selections have little power.")

# ---------------- data ----------------
with tabs[5]:
    st.dataframe(df.drop(columns=["time_bucket"]), width="stretch", hide_index=True)
    st.download_button("Download filtered data (CSV)", df.to_csv(index=False).encode(), "filtered_sessions.csv", "text/csv")
