# app.py
# 🌸 Lotus Architecture Lab — Nine-Fold Geometry & Unity Index
# Author: Sir Brahmam Labs (New Era High School)
#
# Inspired by: "The Architecture of Unity: Mathematical and Spiritual Symbolism of Nine-Fold Geometry in the Bahá’í Lotus Temple"
# Core ideas implemented: nonagon plan, 9-petal rose curve, equal-angle entrances, light & sound uniformity, Unity Index.

import io
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st

st.set_page_config(page_title="Lotus Architecture Lab — Nine-Fold Geometry", layout="wide")

# ======================
# Sidebar controls
# ======================
st.sidebar.title("Controls")
R = st.sidebar.slider("Nonagon radius (m, model scale)", 10.0, 60.0, 30.0, 1.0)
petal_a = st.sidebar.slider("Rose-curve scale a (m)", 5.0, 60.0, 30.0, 1.0)
petal_n = st.sidebar.number_input("Rose petals (n)", min_value=3, max_value=15, value=9, step=1)
theta_samples = st.sidebar.slider("Curve resolution (points)", 360, 4000, 1200, 60)

st.sidebar.markdown("---")
st.sidebar.subheader("Light / Sound model (educational)")
sensor_radius = st.sidebar.slider("Sensor ring radius rₛ (m)", 1.0, 0.95*R, 0.6*R, 1.0)
power_light = st.sidebar.slider("Relative lamp power (each entrance)", 0.1, 5.0, 1.0, 0.1)
power_sound = st.sidebar.slider("Relative source power (center voice)", 0.1, 5.0, 1.0, 0.1)
falloff_p = st.sidebar.selectbox("Inverse-distance falloff exponent", [1.0, 1.5, 2.0, 3.0], index=2)
noise = st.sidebar.slider("Small randomness (±%)", 0.0, 10.0, 0.0, 0.5)

st.sidebar.markdown("---")
export_csv = st.sidebar.checkbox("Enable CSV downloads", value=True)
st.sidebar.caption("Tip: Keep parameters simple to demonstrate symmetry & fairness.")

# ======================
# Helpers
# ======================
def nonagon_vertices(R):
    # 9 equal vertices on a circle
    verts = []
    for k in range(9):
        ang = 2*math.pi*k/9.0
        verts.append((R*math.cos(ang), R*math.sin(ang)))
    return np.array(verts)

def rose_curve(a, n, samples):
    # r = a * |cos(n*theta)| creates n petals in [0, 2π]
    th = np.linspace(0, 2*np.pi, samples)
    r = a * np.abs(np.cos(n*th))
    x = r*np.cos(th); y = r*np.sin(th)
    return th, x, y

def sensor_ring(radius, samples=360):
    th = np.linspace(0, 2*np.pi, samples, endpoint=False)
    x = radius*np.cos(th); y = radius*np.sin(th)
    return th, x, y

def entrance_angles():
    # 9 equal entrances → angles at 0, 40°, 80°, ..., 320° (radians)
    return np.array([2*math.pi*k/9.0 for k in range(9)])

def add_noise(arr, pct):
    if pct <= 0:
        return arr
    rng = np.random.default_rng(42)
    return arr * (1.0 + (rng.random(len(arr))*2 - 1)*(pct/100.0))

def light_distribution(sensor_pts, entrances_r, P=1.0, p=2.0, jitter_pct=0.0):
    # Entrances placed on a circle of radius R at equal angles; light sources at each entrance.
    th_e = entrance_angles()
    ex = entrances_r*np.cos(th_e); ey = entrances_r*np.sin(th_e)
    if jitter_pct>0:
        ex = add_noise(ex, jitter_pct); ey = add_noise(ey, jitter_pct)
    intens = []
    for (sx, sy) in sensor_pts:
        dists = np.sqrt((sx-ex)**2 + (sy-ey)**2)
        # prevent division by zero
        dists = np.clip(dists, 1e-6, None)
        I = np.sum(P/(dists**p))
        intens.append(I)
    return np.array(intens)

def sound_distribution(sensor_pts, center=(0.0,0.0), P=1.0, p=2.0):
    cx, cy = center
    intens = []
    for (sx, sy) in sensor_pts:
        d = math.hypot(sx-cx, sy-cy)
        d = max(d, 1e-6)
        intens.append(P/(d**p))
    return np.array(intens)

def coef_var(x):
    x = np.array(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x)==0: return np.nan
    mu = x.mean()
    if mu==0: return np.nan
    return x.std(ddof=1)/mu

def unity_index(cv_light, cv_sound, order=9, reflections=9):
    # 4-part score: symmetry (rotations+reflections), light uniformity, sound uniformity
    # Scale to ~[0,4], lower CV = better.
    # Symmetry score (max 2): exact order-9 → 2.0 (1 for rotations, 1 for reflections)
    sym = 0.0
    sym += 1.0 if order==9 else min(order/9.0, 1.0)
    sym += 1.0 if reflections==9 else min(reflections/9.0, 1.0)

    # Light/Sound (max 2): map CV in [0, 0.05] to [1, 0]; clamp
    def score_from_cv(cv):
        if not np.isfinite(cv): return 0.0
        if cv <= 0.05: return 1.0 - (cv/0.05)
        return 0.0
    L = score_from_cv(cv_light)
    S = score_from_cv(cv_sound)
    return round(sym + L + S, 2)

# ======================
# Geometry tab
# ======================
st.title("🌸 Lotus Architecture Lab — Nine-Fold Geometry")

tab1, tab2, tab3, tab4 = st.tabs([
    "Geometry (Plan)",
    "Light Balance",
    "Sound Balance",
    "Unity Index & Export"
])

# ---- Tab 1: Geometry
with tab1:
    st.subheader("Nonagon base & 9-petal rose curve")
    # Nonagon
    V = nonagon_vertices(R)
    V_closed = np.vstack([V, V[0]])
    fig_geo = go.Figure()

    fig_geo.add_trace(go.Scatter(
        x=V_closed[:,0], y=V_closed[:,1],
        mode="lines+markers", name="Nonagon", line=dict(width=2)
    ))
    # Rose curve
    th, rx, ry = rose_curve(petal_a, petal_n, theta_samples)
    fig_geo.add_trace(go.Scatter(
        x=rx, y=ry, mode="lines", name=f"Rose curve r=a|cos({petal_n}θ)|", line=dict(width=2)
    ))
    # Unity center
    fig_geo.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers", name="Unity Center", marker=dict(size=10, symbol="x")
    ))

    fig_geo.update_yaxes(scaleanchor="x", scaleratio=1)
    fig_geo.update_layout(height=600, title="Plan view", legend_orientation="h")
    st.plotly_chart(fig_geo, use_container_width=True)

    st.markdown(
        "- **Rotational symmetry**: order 9 (40° steps), **reflective symmetry**: 9 mirror axes.\n"
        "- **Unity Center** at (0,0): equal radius to all entrances.\n"
        "- **Rose curve** illustrates lotus-like boundary from a single simple equation."
    )

# ---- Prepare sensor ring & entrances for both models
th_s, sx, sy = sensor_ring(sensor_radius, samples=360)
sensor_pts = list(zip(sx, sy))

# ---- Tab 2: Light Balance
with tab2:
    st.subheader("Equal-angle light distribution from 9 entrances")
    I = light_distribution(sensor_pts, entrances_r=R, P=power_light, p=falloff_p, jitter_pct=noise)
    cvL = coef_var(I)

    dfL = pd.DataFrame({"theta_deg": np.degrees(th_s), "Intensity": I})
    figL = px.line(dfL, x="theta_deg", y="Intensity", title=f"Light intensity on sensor ring (CV={cvL:.3f})")
    st.plotly_chart(figL, use_container_width=True)

    st.caption("Lower coefficient of variation (CV) ⇒ more uniform light around the ring.")

# ---- Tab 3: Sound Balance
with tab3:
    st.subheader("Center voice distribution (acoustic symmetry)")
    # Sound from center to ring is perfectly symmetric; CV≈0 (numerical roundoff)
    Snd = sound_distribution(sensor_pts, center=(0.0,0.0), P=power_sound, p=falloff_p)
    cvS = coef_var(Snd)

    dfS = pd.DataFrame({"theta_deg": np.degrees(th_s), "Intensity": Snd})
    figS = px.line(dfS, x="theta_deg", y="Intensity", title=f"Sound intensity on sensor ring (CV={cvS:.3f})")
    st.plotly_chart(figS, use_container_width=True)

    st.caption("Symmetry makes sound almost uniform at equal radius from center (idealized model).")

# ---- Tab 4: Unity Index & Export
with tab4:
    st.subheader("Unity Index (0–4)")
    score = unity_index(cvL, cvS, order=9, reflections=9)
    cols = st.columns(2)
    with cols[0]:
        st.metric("Unity Index", f"{score}/4.00")
        st.write(
            f"- Symmetry (rotations + reflections): 2.00\n"
            f"- Light uniformity score: {('%.2f' % (1.0 - min(cvL/0.05,1.0)))}\n"
            f"- Sound uniformity score: {('%.2f' % (1.0 - min(cvS/0.05,1.0)))}"
        )
    with cols[1]:
        st.write("**Notes**")
        st.write(
            "This educational index rewards order-9 symmetry and low variation in light/sound around a ring. "
            "Values near 4.00 indicate near-perfect harmony."
        )

    if export_csv:
        st.markdown("### Downloads")
        lbuf = io.StringIO(); dfL.to_csv(lbuf, index=False)
        sbuf = io.StringIO(); dfS.to_csv(sbuf, index=False)
        st.download_button("Download Light CSV", lbuf.getvalue(), file_name="light_ring.csv", mime="text/csv")
        st.download_button("Download Sound CSV", sbuf.getvalue(), file_name="sound_ring.csv", mime="text/csv")

st.markdown("---")
st.caption("Lotus geometry, nine-fold symmetry, and balance modeled for classroom exploration.")
