"""Lebanon After Dark: interactive dashboard of night-time lights by district, 2012-2026.
Data: NASA/NOAA VIIRS monthly night-time lights, aggregated to Lebanon's 26 districts.
"""
import difflib
import json
import re
import urllib.request
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------- setup
st.set_page_config(page_title="Lebanon After Dark", layout="centered")

HERE = Path(__file__).parent
DATA_FILE = HERE / "lebanon_lights_monthly_clean.csv"
GEO_FILE = HERE / "lebanon_districts.geojson"
GEO_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_LBN_2.json"

NAVY = "#1F3A5F"
MAROON = "#8C1C3A"
GRAY = "#B8BEC6"
INK = "#2B2B2B"
MUTED = "#6B7280"
CFG = {"displayModeBar": False, "responsive": True}
FONT = dict(family="Georgia, 'Times New Roman', serif", color=INK, size=13)


def style(fig, height=380, legend=False):
    fig.update_layout(
        height=height, font=FONT, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=10, r=10, t=30, b=10), showlegend=legend, hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        hoverlabel=dict(bgcolor="white", font=FONT),
    )
    fig.update_xaxes(showgrid=False, linecolor="#D1D5DB", ticks="outside", tickcolor="#D1D5DB")
    fig.update_yaxes(gridcolor="#EEF0F3", zeroline=False)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", config=CFG)


# ---------------------------------------------------------------- data
@st.cache_data
def load():
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    df["year"] = df["date"].dt.year
    return df


@st.cache_data
def load_geo():
    """Local file first, then download (works on Colab and Streamlit Cloud).
    Pieces that share a district name are merged into one shape, so every piece gets colored."""
    try:
        if GEO_FILE.exists():
            geo = json.loads(GEO_FILE.read_text())
        else:
            with urllib.request.urlopen(GEO_URL, timeout=30) as r:
                geo = json.loads(r.read().decode())
    except Exception:
        return None
    props = geo["features"][0]["properties"]
    key = next((k for k in ("NAME_2", "shapeName", "name", "NAME") if k in props), None)
    if key is None:
        return geo
    merged = {}
    for f in geo["features"]:
        name, g = f["properties"][key], f["geometry"]
        if not g:
            continue
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        if name not in merged:
            merged[name] = {"type": "Feature", "properties": {key: name},
                            "geometry": {"type": "MultiPolygon", "coordinates": []}}
        merged[name]["geometry"]["coordinates"].extend(polys)
    return {"type": "FeatureCollection", "features": list(merged.values())}


def norm(s):
    return re.sub(r"[^a-z]", "", str(s).lower())


def match_geo(geo, districts):
    """Map each district in the data to a feature name in the GeoJSON."""
    keys = [k for k in ("NAME_2", "shapeName", "name", "NAME") if k in geo["features"][0]["properties"]]
    key = keys[0]
    names = [f["properties"][key] for f in geo["features"]]
    by_norm = {norm(n): n for n in names}
    out, left = {}, []
    for d in districts:                      # exact matches first
        if norm(d) in by_norm:
            out[d] = by_norm.pop(norm(d))
        else:
            left.append(d)
    for d in left:                           # then closest remaining name, one-to-one
        close = difflib.get_close_matches(norm(d), list(by_norm), n=1, cutoff=0.5)
        if close:
            out[d] = by_norm.pop(close[0])
    return key, out


df = load()
LAST_YEAR = int(df["year"].max())
LAST_MONTH = int(df.loc[df["year"] == LAST_YEAR, "date"].dt.month.max())
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WINDOW = f"Jan–{MONTH_NAMES[LAST_MONTH - 1]}"

# national series
national = df.groupby("date")["light_total"].sum()
nat_base = national[national.index.year == 2019].mean()
nat_idx = national / nat_base * 100
nat_smooth = nat_idx.rolling(12, center=True, min_periods=6).mean()
nat_year = national.groupby(national.index.year).mean()

# same-months comparison so the partial last year is fair
win = df[df["date"].dt.month <= LAST_MONTH]
piv = win.pivot_table(index="district", columns="year", values="light_total", aggfunc="mean")
level = piv.div(piv[2019], axis=0) * 100
nat_win = piv.sum()

drop = (nat_year[2022] / nat_year[2019] - 1) * 100
trough = nat_idx.idxmin()
now_pct = nat_win[LAST_YEAR] / nat_win[2019] * 100

# district smoothed index
df = df.sort_values(["district", "date"])
df["idx"] = df["light_total"] / df.groupby("district")["light_total"].transform(
    lambda s: s[df.loc[s.index, "year"] == 2019].mean()) * 100
df["idx_smooth"] = df.groupby("district")["idx"].transform(lambda s: s.rolling(12, center=True, min_periods=6).mean())

# ---------------------------------------------------------------- header
st.markdown(
    """
    <style>
    html, body, [class*="css"] {font-family: Georgia, 'Times New Roman', serif;}
    h1, h2, h3 {color: #1F3A5F;}
    .kicker {color:#8C1C3A; font-size:0.85rem; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:-0.6rem;}
    .note {color:#6B7280; font-size:0.85rem;}
    div[data-testid="stMetricValue"] {color:#1F3A5F;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<p class="kicker">Lebanon from space, 2012 to 2026</p>', unsafe_allow_html=True)
st.title("Lebanon After Dark")
st.markdown(
    "Every night a satellite photographs Lebanon. When the power goes out, the country goes dark, "
    "and the satellite sees it. Here is how much light Lebanon lost, and how much has come back, district by district."
)

c1, c2, c3 = st.columns(3)
c1.metric("Light lost, 2019 to 2022", f"{abs(drop):.0f}%")
c2.metric(f"Darkest month ({nat_idx[trough]:.0f}% of 2019)", f"{trough:%b %Y}")
c3.metric(f"Light back today, % of 2019 ({WINDOW} {LAST_YEAR})", f"{now_pct:.0f}%")

# ---------------------------------------------------------------- 1. national story
st.header("1. The fall and the slow return")
st.markdown("Total night-time light across Lebanon, as a percentage of the 2019 average.")

fig = go.Figure()
fig.add_trace(go.Scatter(x=nat_idx.index, y=nat_idx, mode="lines", line=dict(color=GRAY, width=1),
                         name="Monthly", hovertemplate="%{x|%b %Y}: %{y:.0f}% of 2019<extra></extra>"))
fig.add_trace(go.Scatter(x=nat_smooth.index, y=nat_smooth, mode="lines", line=dict(color=NAVY, width=2.5),
                         name="12-month average", hovertemplate="%{x|%b %Y}: %{y:.0f}% (12-month avg)<extra></extra>"))
fig.add_hline(y=100, line=dict(color=MUTED, width=1, dash="dot"))
events = [
    ("2019-10-01", "Oct 2019<br>financial crisis", 0, -50),
    ("2021-08-01", "Aug 2021<br>fuel shortages", -75, 0),
    (f"{trough:%Y-%m-%d}", f"{trough:%b %Y}<br>darkest month", 0, 45),
    ("2024-10-01", "Autumn 2024<br>conflict in the south", -20, -75),
]
for d, label, ax, ay in events:
    d = pd.Timestamp(d)
    y = nat_idx.get(d, nat_smooth.get(d, 50))
    fig.add_annotation(x=d, y=y, text=label, showarrow=True, arrowhead=0, arrowcolor=MUTED,
                       ax=ax, ay=ay, font=dict(size=11, color=INK),
                       bgcolor="rgba(255,255,255,0.85)")
fig.update_yaxes(title=None, ticksuffix="%", range=[0, 130])
show(style(fig, 420, legend=True))
st.markdown(
    f'<p class="note">The decline started in late 2019 but the real break came in August 2021, when fuel ran out. '
    f'At the bottom, Lebanon gave off about {nat_idx[trough]:.0f}% of its normal light.</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- 2. map
st.header("2. Find your district")
year = st.select_slider("Year", options=list(range(2012, LAST_YEAR + 1)), value=LAST_YEAR)
st.markdown(f"Light in {year} as a % of 2019 ({WINDOW} of each year, so years compare fairly). "
            "The darker the district, the closer it is to its 2019 light.")

map_df = level[[year]].rename(columns={year: "pct"}).reset_index()
map_df["pct"] = map_df["pct"].round(0)
scale = [[0, "#F2E8EB"], [0.6, MAROON], [1, "#4A0B1D"]]
geo = load_geo()
drawn = False
if geo is not None:
    try:
        key, mapping = match_geo(geo, map_df["district"])
        map_df["geo_name"] = map_df["district"].map(mapping)
        fig = px.choropleth(map_df.dropna(subset=["geo_name"]), geojson=geo, locations="geo_name",
                            featureidkey=f"properties.{key}", color="pct", hover_name="district",
                            color_continuous_scale=scale, range_color=[0, 100],
                            custom_data=["district", "pct"])
        fig.update_traces(marker_line_color="white", marker_line_width=1,
                          hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]:.0f}% of 2019 light<extra></extra>")
        fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator")
        fig.update_layout(
            dragmode=False,
            coloraxis_colorbar=dict(title=dict(text="% of 2019 light", side="top"), ticksuffix="%",
                                    orientation="h", x=0.5, xanchor="center", y=-0.02, yanchor="top",
                                    len=0.9, thickness=10, tickvals=[0, 25, 50, 75, 100],
                                    ticktext=["0%", "25%", "50%", "75%", "100%+"]),
        )
        fig = style(fig, 620)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=70))
        st.plotly_chart(fig, width="stretch", config={**CFG, "scrollZoom": False, "staticPlot": False})
        drawn = True
        missing = sorted(set(map_df["district"]) - set(mapping))
        all_geo = {f["properties"][key] for f in geo["features"]}
        empty = sorted(all_geo - set(mapping.values()))
        if missing or empty:
            st.caption("Data districts not on the map: " + (", ".join(missing) or "none")
                       + " | Map areas without data: " + (", ".join(empty) or "none"))
    except Exception:
        drawn = False
if not drawn:
    st.info("District boundaries could not be loaded, so the ranking is shown instead.")
    bar = map_df.sort_values("pct")
    fig = go.Figure(go.Bar(x=bar["pct"], y=bar["district"], orientation="h",
                           marker_color=[MAROON if v < 100 else NAVY for v in bar["pct"]],
                           hovertemplate="%{y}: %{x:.0f}% of 2019<extra></extra>"))
    fig.add_vline(x=100, line=dict(color=MUTED, dash="dot"))
    fig.update_xaxes(ticksuffix="%")
    show(style(fig, 640))

# ---------------------------------------------------------------- 3. recovery, two lenses
st.header("3. Who came back, who is still dark")
lens = st.radio("View", ["% of 2019 light", "Light still missing"], horizontal=True)

if lens == "% of 2019 light":
    dd = pd.DataFrame({"low": level[2022], "now": level[LAST_YEAR]}).sort_values("now")
    fig = go.Figure()
    for d, r in dd.iterrows():
        fig.add_trace(go.Scatter(x=[r["low"], r["now"]], y=[d, d], mode="lines",
                                 line=dict(color=GRAY, width=2), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=dd["low"], y=dd.index, mode="markers", name="2022 (bottom)",
                             marker=dict(color=MAROON, size=10, line=dict(color="white", width=2)),
                             hovertemplate="%{y}, 2022: %{x:.0f}% of 2019<extra></extra>"))
    fig.add_trace(go.Scatter(x=dd["now"], y=dd.index, mode="markers", name=f"{LAST_YEAR} ({WINDOW})",
                             marker=dict(color=NAVY, size=10, line=dict(color="white", width=2)),
                             hovertemplate=f"%{{y}}, {LAST_YEAR}: %{{x:.0f}}% of 2019<extra></extra>"))
    fig.add_vline(x=100, line=dict(color=MUTED, width=1, dash="dot"),
                  annotation_text="2019 level", annotation_position="top")
    fig.update_xaxes(ticksuffix="%", range=[0, 125])
    show(style(fig, 720, legend=True))
    best, worst = dd["now"].idxmax(), dd["now"].idxmin()
    st.markdown(f'<p class="note">{best} is back to {dd.loc[best, "now"]:.0f}% of its 2019 light. '
                f'{worst} is at {dd.loc[worst, "now"]:.0f}%. Each line shows how far a district climbed back from its 2022 low.</p>',
                unsafe_allow_html=True)
else:
    gap = (piv[2019] - piv[LAST_YEAR]).clip(lower=0)
    share = (gap / gap.sum() * 100).sort_values()
    top3 = share.tail(3)
    fig = go.Figure(go.Bar(x=share, y=share.index, orientation="h",
                           marker_color=[MAROON if d in top3.index else NAVY for d in share.index],
                           hovertemplate="%{y}: %{x:.1f}% of Lebanon's missing light<extra></extra>"))
    fig.update_xaxes(ticksuffix="%", title="Share of Lebanon's missing light")
    show(style(fig, 720))
    st.markdown(f'<p class="note">In percentage terms small rural districts look worst. In absolute terms the gap is '
                f'concentrated in Mount Lebanon: {", ".join(top3.index[::-1])} alone account for '
                f'{top3.sum():.0f}% of the light Lebanon has not recovered.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------- 4. district explorer
st.header("4. Explore one district")
districts = sorted(df["district"].unique())
pick = st.selectbox("District", districts, index=districts.index("Beirut") if "Beirut" in districts else 0)
sub = df[df["district"] == pick]

share19 = piv.loc[pick, 2019] / nat_win[2019] * 100
low_row = sub.loc[sub["idx_smooth"].idxmin()]
k1, k2, k3 = st.columns(3)
k1.metric("Share of national light, 2019", f"{share19:.1f}%")
k2.metric(f"Lowest point, {low_row['date']:%b %Y}", f"{low_row['idx_smooth']:.0f}%")
k3.metric(f"{LAST_YEAR} vs 2019", f"{level.loc[pick, LAST_YEAR]:.0f}%")

fig = go.Figure()
fig.add_trace(go.Scatter(x=nat_smooth.index, y=nat_smooth, mode="lines", line=dict(color=GRAY, width=2),
                         name="Lebanon", hovertemplate="Lebanon %{x|%b %Y}: %{y:.0f}%<extra></extra>"))
fig.add_trace(go.Scatter(x=sub["date"], y=sub["idx_smooth"], mode="lines", line=dict(color=NAVY, width=2.5),
                         name=pick, hovertemplate=f"{pick} %{{x|%b %Y}}: %{{y:.0f}}%<extra></extra>"))
fig.add_hline(y=100, line=dict(color=MUTED, width=1, dash="dot"))
fig.update_yaxes(ticksuffix="%", title=None)
show(style(fig, 380, legend=True))
st.caption("12-month average, 2019 = 100%.")

# ---------------------------------------------------------------- 5. regions
st.header("5. Compare governorates")
gov = df.groupby(["governorate", "date"])["light_total"].sum().reset_index()
gbase = gov[gov["date"].dt.year == 2019].groupby("governorate")["light_total"].mean()
gov["idx"] = gov["light_total"] / gov["governorate"].map(gbase) * 100
gov["smooth"] = gov.groupby("governorate")["idx"].transform(lambda s: s.rolling(12, center=True, min_periods=6).mean())
govs = sorted(gov["governorate"].unique())
hl = st.selectbox("Highlight", govs, index=govs.index("South") if "South" in govs else 0)

fig = go.Figure()
for g in govs:
    if g == hl:
        continue
    s = gov[gov["governorate"] == g]
    fig.add_trace(go.Scatter(x=s["date"], y=s["smooth"], mode="lines", line=dict(color=GRAY, width=1.2),
                             name=g, showlegend=False, hovertemplate=f"{g} %{{x|%b %Y}}: %{{y:.0f}}%<extra></extra>"))
s = gov[gov["governorate"] == hl].dropna(subset=["smooth"])
fig.add_trace(go.Scatter(x=s["date"], y=s["smooth"], mode="lines", line=dict(color=MAROON, width=3),
                         name=hl, hovertemplate=f"{hl} %{{x|%b %Y}}: %{{y:.0f}}%<extra></extra>"))
fig.add_annotation(x=s["date"].iloc[-1], y=s["smooth"].iloc[-1], text=f"<b>{hl}</b>", showarrow=False,
                   xanchor="left", xshift=6, font=dict(color=INK))
fig.add_hline(y=100, line=dict(color=MUTED, width=1, dash="dot"))
fig.update_yaxes(ticksuffix="%", title=None)
show(style(fig, 400))
st.caption("12-month average, 2019 = 100%. Gray lines are the other governorates.")

# ---------------------------------------------------------------- method + data
with st.expander("How we built this"):
    st.markdown(f"""
- **Data:** VIIRS monthly night-time lights (NASA/NOAA), summed over each of Lebanon's 26 districts, Jan 2012 to {MONTH_NAMES[LAST_MONTH-1]} {LAST_YEAR}.
- **Cleaning:** four districts (Batroun, Jezzine, Minieh-Danieh, Nabatiyeh) were split across two governorates in the raw boundaries. We merged the pieces and assigned each district to the governorate holding most of its area.
- **Comparisons:** every figure is relative to the 2019 average, the last year before the crisis. When comparing years we use the same months ({WINDOW}) so {LAST_YEAR} is not penalised for being incomplete.
- **Smoothing:** a 12-month moving average removes seasonal swings and one-off spikes (for example June 2025 in Baalbeck and Hermel, likely harvest fires).
- **Limits:** satellites see outdoor light (streets, signs, facades), not electricity use. Indoor lights running on solar batteries are mostly invisible, so night lights are a proxy for public power and activity, not a meter.
- **Boundaries:** GADM 4.1 district boundaries.
""")
    out = level.round(1).reset_index()
    out.columns = ["district"] + [f"{c} (% of 2019)" for c in out.columns[1:]]
    st.dataframe(out, width="stretch", hide_index=True)
    st.download_button("Download this table (CSV)", out.to_csv(index=False), "lebanon_lights_by_district.csv")

st.markdown('<p class="note">MSBA 325 Data Visualization project, American University of Beirut, Fall 2026-2027.</p>',
            unsafe_allow_html=True)
