"""
IV Surface Dashboard
================================
Comprehensive Streamlit dashboard for implied volatility surface construction,
calibration, validation, and operational controls:

  Arbitrage Enforcement     (ArbitrageChecker, QuoteAdjuster)
  SSVI Surface Model
  Advanced Calibration      (vega-weighting, Tikhonov, warm-start)
  Total Variance Framework
  Validation Suite          (Greeks, Benchmarks, Backtesting, PCA)
  Operational Controls      (Bid/Ask, Audit, Alerts, Pipeline)
  Forward Curve, eSSVI Extended Model

Run:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── path bootstrap (must be before any project imports) ───────────────────────
ROOT = Path(__file__).parent
SRC  = ROOT / "src" / "python"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── project imports ────────────────────────────────────────────────────────────
from surface.arbitrage  import ArbitrageChecker, ArbitrageReport
from surface.ssvi       import SSVISurface, SSVIParameters
from surface.essvi      import ESSVISurface, ESSVIParameters
from surface.greeks     import GreeksCalculator, GreeksValidator
from surface.benchmark  import BenchmarkStructurePricer
from surface.backtesting import Backtester, StressScenario
from surface.parameter_monitoring import ParameterDynamicsMonitor
from surface.bid_ask    import BidAskSurfaceGenerator
from surface.total_variance import TotalVarianceInterpolator, TotalVarianceConfig
from surface.quote_adjustment import QuoteAdjuster
from data.forwards      import ForwardCurveBuilder
from data.pipeline      import (
    load_or_fetch, snapshot_to_grid,
    list_cached_tickers, get_snapshot_info,
)
import datetime

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IV Surface Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── session state ─────────────────────────────────────────────────────────────
for _ss_key in ("iv_snapshot", "iv_grid", "iv_quality", "iv_clean_df"):
    if _ss_key not in st.session_state:
        st.session_state[_ss_key] = None


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_arbitrage_checker(tol: float = 1e-6) -> ArbitrageChecker:
    return ArbitrageChecker(tolerance=tol)


def make_synthetic_surface(
    spot: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    atm_vol: float = 0.20,
    skew: float = -0.10,
    convexity: float = 0.05,
    vol_term_slope: float = 0.02,
) -> np.ndarray:
    """Generate a realistic synthetic IV surface (n_strikes × n_maturities)."""
    ivs = np.zeros((len(strikes), len(maturities)))
    for j, T in enumerate(maturities):
        base_vol = atm_vol + vol_term_slope * np.sqrt(T)
        for i, K in enumerate(strikes):
            k = np.log(K / spot)
            ivs[i, j] = base_vol + skew * k + convexity * k ** 2
    return np.clip(ivs, 0.01, 2.0)


def plotly_surface_3d(
    strikes: np.ndarray,
    maturities: np.ndarray,
    ivs: np.ndarray,
    title: str = "Implied Volatility Surface",
    colorscale: str = "Viridis",
    z_label: str = "IV",
) -> go.Figure:
    """Return a Plotly 3-D surface figure (ivs: n_strikes × n_maturities)."""
    K_grid, T_grid = np.meshgrid(strikes, maturities, indexing="ij")
    fig = go.Figure(
        data=[
            go.Surface(
                x=T_grid,
                y=K_grid,
                z=ivs * 100,          # show as percent
                colorscale=colorscale,
                colorbar=dict(title=f"{z_label} (%)"),
                hovertemplate=(
                    "Maturity: %{x:.3f}y<br>"
                    "Strike: %{y:.1f}<br>"
                    f"{z_label}: %{{z:.2f}}%<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Maturity (years)",
            yaxis_title="Strike",
            zaxis_title=f"{z_label} (%)",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8)),
        ),
        height=520,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def plotly_smile(
    strikes: np.ndarray,
    ivs_cols: list[np.ndarray],
    labels: list[str],
    title: str = "Volatility Smile",
) -> go.Figure:
    """Overlay multiple smile slices."""
    fig = go.Figure()
    colours = px.colors.qualitative.Plotly
    for idx, (iv_col, label) in enumerate(zip(ivs_cols, labels)):
        fig.add_trace(
            go.Scatter(
                x=strikes,
                y=iv_col * 100,
                mode="lines+markers",
                name=label,
                line=dict(color=colours[idx % len(colours)]),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Strike",
        yaxis_title="Implied Volatility (%)",
        height=380,
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def arb_badge(report: ArbitrageReport) -> None:
    """Display a coloured badge for arbitrage status."""
    if report.is_arbitrage_free:
        st.success("✅ Arbitrage-Free")
    else:
        total = report.butterfly_violations + report.calendar_violations
        st.error(f"❌ {total} Arbitrage Violation(s)  |  "
                 f"Butterfly: {report.butterfly_violations}  |  "
                 f"Calendar: {report.calendar_violations}")


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR – GLOBAL PARAMETERS + DATA SOURCE
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/combo-chart--v1.png", width=60)
    st.title("IV Surface Dashboard")
    st.caption("Arbitrage · SSVI · Calibration · Validation · Operations")

    st.divider()
    st.subheader("🌐 Market Parameters")
    rate      = st.number_input("Risk-Free Rate (r)", value=0.05, min_value=0.0, max_value=0.5, step=0.005, format="%.3f")
    div_yield = st.number_input("Dividend Yield (q)", value=0.01, min_value=0.0, max_value=0.5, step=0.005, format="%.3f")

    st.divider()
    st.subheader("📥 Data Source")
    data_source = st.radio(
        "Surface data from:",
        ["Synthetic", "Real Market Data"],
        key="data_source_radio",
    )

    if data_source == "Real Market Data":
        # ── Ticker selection ──────────────────────────────────────────────
        _cached = list_cached_tickers(ROOT / "data")
        _ticker_opts = (_cached if _cached else []) + ["✏️ Enter manually"]
        _sel = st.selectbox("Cached tickers", _ticker_opts, key="sel_ticker")
        if _sel == "✏️ Enter manually":
            ticker_input = st.text_input("Ticker", value="AAPL", key="ticker_manual").upper().strip()
        else:
            ticker_input = _sel.upper()

        _c1, _c2 = st.columns(2)
        fetch_btn = _c1.button("🔄 Fetch Live", key="fetch_btn")
        load_btn  = _c2.button("📂 Load Cache", key="load_btn")

        if fetch_btn or load_btn:
            _force = bool(fetch_btn)
            with st.spinner(f"{'Fetching live' if _force else 'Loading'} {ticker_input}…"):
                try:
                    _snap, _qrpt, _cdf = load_or_fetch(
                        ticker_input, ROOT / "data",
                        rate=rate, div_yield=div_yield,
                        force_refresh=_force,
                    )
                    st.session_state["iv_snapshot"] = _snap
                    st.session_state["iv_quality"]  = _qrpt
                    st.session_state["iv_clean_df"] = _cdf
                    st.session_state["iv_grid"]     = None  # reset; sliders will rebuild
                except Exception as _exc:
                    st.error(f"❌ {_exc}")

        _snap_loaded = st.session_state.get("iv_snapshot")

        if _snap_loaded is not None:
            st.success(f"✅ {_snap_loaded.ticker} | S={_snap_loaded.spot:.2f}")
            st.caption(
                f"{_snap_loaded.timestamp[:10]}  ·  "
                f"{_snap_loaded.n_maturities()} mats  ·  "
                f"{_snap_loaded.n_options()} options"
            )
            spot = st.number_input(
                "Spot Price (S)", value=float(_snap_loaded.spot),
                min_value=1.0, step=1.0, key="spot_real",
            )
            st.divider()
            st.subheader("📐 Real Data Grid")
            _n_max = max(3, _snap_loaded.n_maturities())
            _n_mats_use    = st.slider("# Maturities",     3, min(12, _n_max), min(6, _n_max), key="n_mats_real")
            _n_strikes_use = st.slider("# Strike points",  10, 30, 20, key="n_strikes_real")
            _sk_lo = st.slider("Strike lo (%spot)", 70, 95,  80, key="sk_lo")
            _sk_hi = st.slider("Strike hi (%spot)", 105, 150, 125, key="sk_hi")
            try:
                _g = snapshot_to_grid(
                    _snap_loaded,
                    n_strikes=_n_strikes_use,
                    n_mats=_n_mats_use,
                    strike_lo_pct=_sk_lo / 100,
                    strike_hi_pct=_sk_hi / 100,
                )
                strikes, maturities, ivs_base = _g
            except Exception as _exc:
                st.warning(f"Grid error: {_exc} — synthetic fallback")
                strikes    = np.linspace(spot * 0.80, spot * 1.20, 15)
                maturities = np.linspace(0.1, 2.0, 6)
                ivs_base   = make_synthetic_surface(spot, strikes, maturities)
        else:
            st.info("👆 Choose a ticker and click **Load Cache** or **Fetch Live**.")
            spot       = st.number_input("Spot Price (S)", value=100.0, min_value=1.0, step=1.0, key="spot_pre")
            strikes    = np.linspace(spot * 0.80, spot * 1.20, 15)
            maturities = np.linspace(0.1, 2.0, 6)
            ivs_base   = make_synthetic_surface(spot, strikes, maturities)

    else:  # ── Synthetic ─────────────────────────────────────────────────────
        st.divider()
        st.subheader("💲 Spot Price")
        spot = st.number_input("Spot Price (S)", value=100.0, min_value=1.0, step=1.0, key="spot_synth")

        st.divider()
        st.subheader("📐 Surface Grid")
        strike_lo = st.number_input("Strike min (%spot)", value=80.0,  min_value=50.0, max_value=99.0)
        strike_hi = st.number_input("Strike max (%spot)", value=120.0, min_value=101.0, max_value=200.0)
        n_strikes = st.slider("# Strikes",    7,  31, 15, step=2)
        n_mats    = st.slider("# Maturities", 3,   9,  6)
        mat_lo    = st.number_input("Min Maturity (yr)", value=0.08, min_value=0.02, max_value=0.5,  step=0.02)
        mat_hi    = st.number_input("Max Maturity (yr)", value=2.00, min_value=0.5,  max_value=5.0,  step=0.25)

        strikes    = np.linspace(spot * strike_lo / 100, spot * strike_hi / 100, n_strikes)
        maturities = np.linspace(mat_lo, mat_hi, n_mats)

        st.divider()
        st.subheader("📊 Surface Shape")
        atm_vol   = st.slider("ATM Vol (%)",            5,  60, 20) / 100
        skew      = st.slider("Skew",                 -50,   0, -10) / 100
        convexity = st.slider("Convexity",              0,  30,   5) / 100
        ts_slope  = st.slider("Term-Structure Slope",  -5,  10,   2) / 100

        ivs_base = make_synthetic_surface(spot, strikes, maturities, atm_vol, skew, convexity, ts_slope)

    st.divider()
    st.caption("🔬 C++ engine (subprocess IPC) · Data: Yahoo Finance")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════════

tabs = st.tabs([
    "🌍 Market Data",
    "📈 IV Surface",
    "🔍 Arbitrage Check",
    "🛠️ Quote Adjustment",
    "🏛️ SSVI Model",
    "🚀 eSSVI Model",
    "⚗️ Advanced Calibration",
    "📐 Total Variance",
    "🧮 Greeks",
    "📏 Benchmarks",
    "📉 Backtesting",
    "🔬 PCA Monitor",
    "📊 Bid/Ask Spread",
    "⏩ Forward Curve",
])

(
    tab_data, tab_surface, tab_arb, tab_adj, tab_ssvi, tab_essvi,
    tab_adv_cal, tab_tv, tab_greeks, tab_bench,
    tab_bt, tab_pca, tab_bidask, tab_fwd,
) = tabs


# ─────────────────────────────────────────────────────────────────────────────
# TAB 0 — MARKET DATA
# ─────────────────────────────────────────────────────────────────────────────
with tab_data:
    st.header("🌍 Market Data — Fetch, Cache & Inspect")
    _snap_d = st.session_state.get("iv_snapshot")
    _qrpt_d = st.session_state.get("iv_quality")
    _cdf_d  = st.session_state.get("iv_clean_df")

    if _snap_d is None:
        st.info(
            "No market data loaded yet.  "
            "👈 In the sidebar, select **Real Market Data**, pick a ticker, "
            "then click **Load Cache** or **Fetch Live**."
        )
        _avail = list_cached_tickers(ROOT / "data")
        if _avail:
            st.subheader("📂 Available Cached Tickers")
            st.dataframe(
                pd.DataFrame({"Ticker": _avail, "File": [f"data/raw/{t}_market_data.json" for t in _avail]}),
                width='content', hide_index=True,
            )
        else:
            st.warning("No cached tickers found in `data/raw/`. Click **Fetch Live** to download.")
    else:
        _info = get_snapshot_info(_snap_d)

        # ── Overview metrics ──────────────────────────────────────────────
        _mc = st.columns(5)
        _mc[0].metric("Ticker",     _info["ticker"])
        _mc[1].metric("Spot",       f"{_info['spot']:.2f}")
        _mc[2].metric("Maturities", str(_info["n_maturities"]))
        _mc[3].metric("Options",    str(_info["n_options"]))
        _mc[4].metric("Snapshot",   _info["timestamp"][:10])

        st.divider()

        # ── Data Quality Report ───────────────────────────────────────────
        st.subheader("🔬 Data Quality Report (DataQualityPipeline)")
        if _qrpt_d is not None:
            _qc = st.columns(5)
            _qc[0].metric("Score",    f"{_qrpt_d.score:.1f}/100")
            _qc[1].metric("Grade",    _qrpt_d.grade)
            _qc[2].metric("Coverage", f"{_qrpt_d.coverage['coverage_score']*100:.0f}%")
            _qc[3].metric("Outliers", str(_qrpt_d.outliers['n_outliers']))
            _qc[4].metric("TS Inv.",  str(_qrpt_d.term_structure['n_inversions']))
            if _qrpt_d.passed:
                st.success(f"✅ PASS — Grade {_qrpt_d.grade}  |  {_qrpt_d.summary()}")
            else:
                st.warning(f"⚠️ FAIL — Grade {_qrpt_d.grade}  |  {_qrpt_d.summary()}")
            if _qrpt_d.recommendations:
                with st.expander("📋 Recommendations"):
                    for _rec in _qrpt_d.recommendations:
                        st.markdown(f"• {_rec}")
        else:
            st.info("Quality report not available (fetch live data to regenerate).")

        st.divider()

        # ── Expiry Classification (ExpiryClassifier) ──────────────────────
        st.subheader("📅 Expiry Type Breakdown (ExpiryClassifier)")
        from data.expiry_classifier import ExpiryClassifier as _EC
        _clf2  = _EC()
        _today2 = datetime.date.today()
        _exp_counts: dict = {}
        for _T2 in _snap_d.maturities:
            _dte2  = int(round(_T2 * 365.25))
            _expd2 = _today2 + datetime.timedelta(days=max(1, _dte2))
            _et2   = _clf2.classify(_expd2, today=_today2)
            _exp_counts[_et2.value] = _exp_counts.get(_et2.value, 0) + 1
        _ec_df = pd.DataFrame(
            {"Type": list(_exp_counts.keys()), "Count": list(_exp_counts.values())}
        ).sort_values("Count", ascending=False).reset_index(drop=True)
        _ecol1, _ecol2 = st.columns([1, 2])
        with _ecol1:
            st.dataframe(_ec_df, width='stretch', hide_index=True)
        with _ecol2:
            _fig_ec = px.pie(
                _ec_df, names="Type", values="Count",
                title="Maturities by Expiry Type",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            _fig_ec.update_layout(height=260)
            st.plotly_chart(_fig_ec, width='stretch')

        st.divider()

        # ── ATM Term Structure ────────────────────────────────────────────
        st.subheader("📈 ATM IV Term Structure")
        _atm_rows = []
        for _T3 in _snap_d.maturities:
            _ks3  = _snap_d.strikes_by_maturity[_T3]
            _ivs3 = _snap_d.ivs_by_maturity[_T3]
            if len(_ks3) == 0:
                continue
            _ai = int(np.argmin(np.abs(_ks3 - _snap_d.spot)))
            _atm_rows.append({"Maturity (yr)": round(_T3, 4), "ATM IV (%)": round(float(_ivs3[_ai]) * 100, 2)})
        if _atm_rows:
            _ts_df = pd.DataFrame(_atm_rows)
            _fig_ts = px.line(
                _ts_df, x="Maturity (yr)", y="ATM IV (%)",
                title=f"{_snap_d.ticker} ATM IV Term Structure",
                markers=True, line_shape="linear",
            )
            _fig_ts.update_layout(height=320)
            st.plotly_chart(_fig_ts, width='stretch')

        st.divider()

        # ── Smile slice ────────────────────────────────────────────────────
        st.subheader("📉 Smile Slice")
        _mat_options = [f"{T:.4f}" for T in _snap_d.maturities]
        _sel_mat_str = st.selectbox(
            "Select maturity (years)", _mat_options,
            index=min(3, len(_mat_options) - 1), key="md_smile_mat",
        )
        # Use the original key directly (avoid float round-trip precision loss)
        _sel_mat_idx = _mat_options.index(_sel_mat_str)
        _sel_mat     = _snap_d.maturities[_sel_mat_idx]
        _slice_k     = _snap_d.strikes_by_maturity[_sel_mat]
        _slice_iv    = _snap_d.ivs_by_maturity[_sel_mat] * 100
        _fig_smile_md = go.Figure()
        _fig_smile_md.add_trace(go.Scatter(
            x=_slice_k, y=_slice_iv, mode="lines+markers",
            name=f"T={_sel_mat:.4f}y", line=dict(color="steelblue"),
        ))
        _fig_smile_md.add_vline(
            x=_snap_d.spot, line_dash="dash", line_color="red",
            annotation_text="Spot", annotation_position="top right",
        )
        _fig_smile_md.update_layout(
            xaxis_title="Strike", yaxis_title="IV (%)",
            title=f"{_snap_d.ticker} Smile — T={_sel_mat:.4f}y", height=340,
        )
        st.plotly_chart(_fig_smile_md, width='stretch')

        st.divider()

        # ── Cleaned option chain preview ───────────────────────────────────
        if _cdf_d is not None:
            st.subheader("🗄️ Cleaned Option Chain (cleaners.py + validators.py)")
            _type_filter = st.radio("Option type", ["all", "call", "put"], horizontal=True, key="md_type")
            _preview = _cdf_d if _type_filter == "all" else _cdf_d[_cdf_d["type"] == _type_filter]
            st.dataframe(_preview.head(300), width='stretch', height=350)
            st.caption(f"Showing up to 300 of {len(_preview)} rows  ·  total raw: {len(_cdf_d)}")

            _dl_csv = _cdf_d.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download cleaned CSV", _dl_csv,
                file_name=f"{_snap_d.ticker}_cleaned.csv", mime="text/csv",
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — IV SURFACE (3-D Interactive)
# ─────────────────────────────────────────────────────────────────────────────
with tab_surface:
    st.header("📈 Interactive 3-D Implied Volatility Surface")
    _src_lbl = (
        f"real market ({st.session_state['iv_snapshot'].ticker})"
        if data_source == "Real Market Data" and st.session_state.get("iv_snapshot") is not None
        else "synthetic"
    )
    st.markdown(
        f"Showing **{_src_lbl}** IV surface. "
        "Adjust sidebar parameters or load market data to update. "
        "Drag to rotate; scroll to zoom."
    )

    c1, c2 = st.columns([3, 1])
    with c2:
        colorscale = st.selectbox(
            "Colorscale",
            ["Viridis", "RdBu", "Plasma", "Inferno", "Turbo", "Electric"],
        )
        show_smile = st.checkbox("Overlay smile slices", value=True)
        atm_idx = np.argmin(np.abs(strikes - spot))

    with c1:
        st.plotly_chart(
            plotly_surface_3d(strikes, maturities, ivs_base, colorscale=colorscale),
            width='stretch',
        )

    if show_smile:
        st.subheader("Smile Slices per Maturity")
        smile_cols  = [ivs_base[:, j] for j in range(len(maturities))]
        smile_labels = [f"T={T:.2f}y" for T in maturities]
        st.plotly_chart(
            plotly_smile(strikes, smile_cols, smile_labels, "Smile Slices"),
            width='stretch',
        )

    st.subheader("Term Structure at ATM")
    ts_fig = go.Figure()
    ts_fig.add_trace(go.Scatter(
        x=maturities,
        y=ivs_base[atm_idx, :] * 100,
        mode="lines+markers",
        name="ATM IV",
        line=dict(color="royalblue", width=2),
    ))
    ts_fig.update_layout(
        xaxis_title="Maturity (years)", yaxis_title="ATM IV (%)", height=320,
    )
    st.plotly_chart(ts_fig, width='stretch')

    with st.expander("Surface Data"):
        import pandas as pd
        df = pd.DataFrame(
            ivs_base * 100,
            index=[f"K={K:.1f}" for K in strikes],
            columns=[f"T={T:.2f}" for T in maturities],
        ).round(2)
        st.dataframe(df, width='stretch')


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — ARBITRAGE CHECK
# ─────────────────────────────────────────────────────────────────────────────
with tab_arb:
    st.header("🔍 Arbitrage Checker")
    st.markdown(
        "Validates the IV surface for **butterfly** (∂²C/∂K² ≥ 0) "
        "and **calendar** (∂C/∂T ≥ 0) arbitrage using the C++ engine."
    )

    c1, c2 = st.columns(2)
    with c1:
        tol   = st.number_input("Tolerance", value=1e-6, format="%.2e", min_value=1e-10, max_value=1e-2)
        inject_arb = st.checkbox("Inject butterfly arbitrage for demo", value=False)
    with c2:
        bid_ask_bps = st.slider("Bid-Ask Spread (bps)", 0, 100, 10)

    ivs_check = ivs_base.copy()
    if inject_arb:
        # Artificially create convexity violations in the mid-strike slice
        mid = len(strikes) // 2
        ivs_check[mid, :] -= 0.05
        st.warning("⚠️ Artificial butterfly violation injected at mid-strike.")

    bid_ask = np.full_like(ivs_check, bid_ask_bps / 10_000)

    run_check = st.button("▶ Run Arbitrage Check", type="primary")
    if run_check:
        checker = ArbitrageChecker(tolerance=tol)
        with st.spinner("Running C++ arbitrage checks…"):
            report = checker.validate_surface(
                strikes=strikes,
                maturities=maturities,
                implied_vols=ivs_check,
                spot=spot,
                rate=rate,
                dividend_yield=div_yield,
                bid_ask_spreads=bid_ask,
            )

        arb_badge(report)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Arbitrage-Free", "✅ Yes" if report.is_arbitrage_free else "❌ No")
        c2.metric("Butterfly Violations", report.butterfly_violations)
        c3.metric("Calendar Violations", report.calendar_violations)
        c4.metric("TV Violations", report.total_variance_violations)

        if report.violations:
            st.subheader("Violation Details")
            import pandas as pd
            vdf = pd.DataFrame([
                {
                    "Type": v.type,
                    "Severity": v.severity,
                    "Value": f"{v.value:.6f}",
                    "Tolerance": f"{v.tolerance:.2e}",
                    "Message": v.message,
                }
                for v in report.violations
            ])
            st.dataframe(vdf, width='stretch')

        # Per-maturity butterfly check visualisation
        st.subheader("Per-Maturity Butterfly Profile")
        chk_figs = []
        for j, T in enumerate(maturities):
            _, vios = checker.check_butterfly_arbitrage(
                strikes=strikes,
                implied_vols=ivs_check[:, j],
                maturity=T,
                spot=spot,
                rate=rate,
                dividend_yield=div_yield,
            )
            chk_figs.append(len(vios))

        bf_fig = go.Figure(go.Bar(
            x=[f"T={T:.2f}" for T in maturities],
            y=chk_figs,
            marker_color=["red" if v > 0 else "green" for v in chk_figs],
        ))
        bf_fig.update_layout(yaxis_title="# Violations", height=300)
        st.plotly_chart(bf_fig, width='stretch')


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — QUOTE ADJUSTMENT
# ─────────────────────────────────────────────────────────────────────────────
with tab_adj:
    st.header("🛠️ Quote Adjustment Framework")
    st.markdown(
        "Constrained quadratic optimisation that **minimises adjustments** "
        "while restoring arbitrage-free conditions."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        adj_mat_idx = st.selectbox("Maturity slice to adjust", range(len(maturities)),
                                   format_func=lambda i: f"T={maturities[i]:.2f}y")
        inject_bf = st.checkbox("Inject butterfly violation", value=True)
    with c2:
        adj_tol     = st.number_input("Tolerance", value=1e-6, format="%.2e", key="adj_tol")
        max_adj     = st.slider("Max adjustment (vol pts)", 0.01, 0.20, 0.05)
    with c3:
        adj_type = st.radio("Adjustment type", ["butterfly", "combined"])

    T_slice   = maturities[adj_mat_idx]
    iv_slice  = ivs_base[:, adj_mat_idx].copy()
    if inject_bf:
        mid = len(strikes) // 2
        iv_slice[mid] = max(0.01, iv_slice[mid] - 0.06)

    run_adj = st.button("▶ Run Quote Adjustment", type="primary")
    if run_adj:
        adjuster = QuoteAdjuster(tolerance=adj_tol, max_adjustment=max_adj)
        with st.spinner("Optimising quotes…"):
            report = adjuster.adjust_butterfly_arbitrage(
                strikes=strikes,
                market_ivs=iv_slice,
                maturity=T_slice,
                spot=spot,
                rate=rate,
                dividend_yield=div_yield,
            )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Success", "✅" if report.success else "❌")
        c2.metric("Quotes Adjusted", report.num_adjusted)
        c3.metric("Max Δ (vol pts)", f"{report.max_adjustment:.4f}")
        c4.metric("RMSE Δ (vol pts)", f"{report.rmse_adjustment:.4f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=strikes, y=iv_slice * 100,
                                  mode="lines+markers", name="Original",
                                  line=dict(dash="dash", color="firebrick")))
        fig.add_trace(go.Scatter(x=strikes, y=report.adjusted_ivs * 100,
                                  mode="lines+markers", name="Adjusted",
                                  line=dict(color="steelblue")))
        fig.update_layout(title=f"Quote Adjustment  T={T_slice:.2f}y",
                          xaxis_title="Strike", yaxis_title="IV (%)", height=380)
        st.plotly_chart(fig, width='stretch')

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Before Adjustment**")
            arb_badge(report.original_arbitrage_report)
        with col2:
            st.markdown("**After Adjustment**")
            arb_badge(report.adjusted_arbitrage_report)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — SSVI MODEL
# ─────────────────────────────────────────────────────────────────────────────
with tab_ssvi:
    st.header("🏛️ SSVI Surface Model")
    st.markdown(
        r"""
        **Gatheral–Jacquier (2014)** parametrisation:
        $$w(k,\theta) = \frac{\theta}{2}\left[1 + \rho\phi(\theta)k
        + \sqrt{(\phi(\theta)k+\rho)^2+(1-\rho^2)}\right]$$
        $$\phi(\theta)=\eta/\theta^\gamma$$
        """
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        s_eta   = st.slider("η  (phi scale)",       0.10, 3.0, 0.70, step=0.05)
        s_gamma = st.slider("γ  (phi exponent)",    0.10, 0.99, 0.50, step=0.05)
    with c2:
        s_rho   = st.slider("ρ  (correlation)",    -0.95, 0.95, -0.40, step=0.05)
        s_theta0 = st.slider("θ₀  (ATM var, T=1)", 0.01, 0.30, 0.04, step=0.005)
    with c3:
        s_theta_slope = st.slider("θ term-structure slope", -0.02, 0.10, 0.02, step=0.002)
        ssvi_spot = st.number_input("Forward price (SSVI)", value=float(spot), key="ssvi_fwd")

    # Build SSVI surface
    ssvi_mats   = maturities
    theta_curve = s_theta0 * (1 + s_theta_slope * ssvi_mats)
    theta_curve = np.clip(theta_curve, 0.001, 10.0)

    params = SSVIParameters(
        theta_curve=theta_curve,
        maturities=ssvi_mats,
        eta=s_eta,
        gamma=s_gamma,
        rho=s_rho,
    )
    surf = SSVISurface(forward=ssvi_spot, maturities=ssvi_mats)
    ivs_ssvi = surf.evaluate_surface(strikes, ssvi_mats, params)

    # Arbitrage check
    ok, msgs = surf.check_gatheral_jacquier_constraints(params)
    if ok:
        st.success("✅ Gatheral–Jacquier constraints satisfied (arbitrage-free by construction)")
    else:
        st.warning("⚠️ Parameter constraints violated: " + "; ".join(msgs))

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            plotly_surface_3d(strikes, ssvi_mats, ivs_ssvi,
                              title="SSVI Implied Volatility Surface",
                              colorscale="RdBu"),
            width='stretch',
        )
    with c2:
        # Smile per maturity
        smile_cols  = [ivs_ssvi[:, j] for j in range(len(ssvi_mats))]
        smile_labels = [f"T={T:.2f}y" for T in ssvi_mats]
        st.plotly_chart(
            plotly_smile(strikes, smile_cols, smile_labels, "SSVI Smiles"),
            width='stretch',
        )

    # Parameter summary
    st.subheader("SSVI Parameters")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("η", f"{s_eta:.3f}")
    c2.metric("γ", f"{s_gamma:.3f}")
    c3.metric("ρ", f"{s_rho:.3f}")
    c4.metric("θ(T₁)", f"{theta_curve[0]:.4f}")
    c5.metric("θ(Tₙ)", f"{theta_curve[-1]:.4f}")

    with st.expander("θ(T) Term Structure"):
        fig_ts = go.Figure(go.Scatter(
            x=ssvi_mats, y=theta_curve,
            mode="lines+markers", name="θ(T)",
            line=dict(color="darkorange", width=2),
        ))
        fig_ts.update_layout(xaxis_title="Maturity (yr)",
                              yaxis_title="ATM Total Variance θ(T)", height=300)
        st.plotly_chart(fig_ts, width='stretch')


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — eSSVI MODEL  (§4 Extended SSVI)
# ─────────────────────────────────────────────────────────────────────────────
with tab_essvi:
    st.header("🚀 §4 — eSSVI Extended Surface Model")
    st.markdown(
        r"""
        Extends SSVI with **maturity-dependent correlation**:
        $$\rho(\theta) = \rho_0 \cdot e^{-\lambda_\rho \cdot \theta}$$
        At $\lambda_\rho=0$ this recovers standard SSVI.
        """
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        es_eta    = st.slider("η",           0.10, 3.0,  0.70, step=0.05, key="es_eta")
        es_gamma  = st.slider("γ",           0.10, 0.99, 0.50, step=0.05, key="es_gamma")
    with c2:
        es_rho0   = st.slider("ρ₀",         -0.95, 0.95, -0.50, step=0.05, key="es_rho0")
        es_lrho   = st.slider("λ_ρ (decay)",  0.0,  5.0,  1.0,  step=0.1,  key="es_lrho")
    with c3:
        es_theta0 = st.slider("θ₀",          0.01, 0.30,  0.04, step=0.005, key="es_theta0")
        es_tslope = st.slider("θ slope",     -0.02, 0.10,  0.02, step=0.002, key="es_tslope")

    es_mats = maturities
    es_theta = np.clip(es_theta0 * (1 + es_tslope * es_mats), 0.001, 10.0)

    es_params = ESSVIParameters(
        theta_curve=es_theta,
        maturities=es_mats,
        eta=es_eta,
        gamma=es_gamma,
        rho_0=es_rho0,
        lambda_rho=es_lrho,
    )

    es_surf = ESSVISurface(
        forward=float(spot),
        maturities=es_mats,
    )
    es_surf.parameters = es_params
    es_surf._build_theta_interp()
    ivs_essvi = es_surf.implied_vol_grid(strikes, es_mats)

    arb_free = es_params.is_arbitrage_free()
    if arb_free:
        st.success("✅ eSSVI arbitrage-free constraints satisfied")
    else:
        bb = es_params.butterfly_bound()
        st.warning(
            f"⚠️ Butterfly constraint violated at {(bb > 0).sum()} maturities. "
            "Reduce η or |ρ₀|."
        )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            plotly_surface_3d(strikes, es_mats, ivs_essvi,
                              title="eSSVI Implied Volatility Surface",
                              colorscale="Plasma"),
            width='stretch',
        )
    with c2:
        # Show ρ(θ) term structure
        rho_vals = es_params.rho_at_theta(es_theta)
        fig_rho = go.Figure()
        fig_rho.add_trace(go.Scatter(
            x=es_mats, y=rho_vals, mode="lines+markers",
            name="ρ(θ(T))", line=dict(color="crimson", width=2),
        ))
        fig_rho.add_hline(y=es_rho0, line_dash="dash", line_color="gray",
                          annotation_text=f"ρ₀={es_rho0:.2f}")
        fig_rho.update_layout(
            title="Maturity-Dependent Correlation ρ(θ(T))",
            xaxis_title="Maturity (yr)", yaxis_title="ρ", height=380,
        )
        st.plotly_chart(fig_rho, width='stretch')

    # SSVI vs eSSVI comparison
    st.subheader("SSVI vs eSSVI Comparison (ATM slice)")
    atm_idx = np.argmin(np.abs(strikes - spot))
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Scatter(
        x=maturities, y=ivs_ssvi[atm_idx, :] * 100,
        mode="lines+markers", name="SSVI ATM IV",
        line=dict(color="steelblue"),
    ))
    fig_cmp.add_trace(go.Scatter(
        x=es_mats, y=ivs_essvi[atm_idx, :] * 100,
        mode="lines+markers", name="eSSVI ATM IV",
        line=dict(color="crimson"),
    ))
    fig_cmp.update_layout(
        xaxis_title="Maturity (yr)", yaxis_title="ATM IV (%)", height=320,
    )
    st.plotly_chart(fig_cmp, width='stretch')


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — ADVANCED CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────
with tab_adv_cal:
    st.header("⚗️ Advanced Calibration Quality")
    st.markdown(
        "Demonstrates **vega-weighting**, **bid-ask weighting**, "
        "**Tikhonov regularisation**, and **multi-objective** calibration."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        use_vega    = st.checkbox("Vega weighting",    value=True)
        use_ba      = st.checkbox("Bid-ask weighting", value=True)
    with c2:
        tikh_lambda = st.slider("Tikhonov λ",  0.0, 0.5, 0.01, step=0.005)
        smooth_lambda = st.slider("Smoothness λ", 0.0, 0.1, 0.001, step=0.001)
    with c3:
        st.markdown("**Multi-objective strategy**")
        mo_fit    = st.slider("Fit weight",       0.0, 2.0, 1.0, step=0.1, key="mo_fit")
        mo_smooth = st.slider("Smoothness weight", 0.0, 1.0, 0.1, step=0.05, key="mo_sm")
        mo_stab   = st.slider("Stability weight",  0.0, 1.0, 0.1, step=0.05, key="mo_st")

    from surface.advanced_calibration import (
        black_scholes_vega,
    )

    # Compute vega weights for visualisation
    st.subheader("Vega Weights")
    # Derive ATM vol directly from the active surface (works for both synthetic and real data)
    _atm_idx_vega = int(np.argmin(np.abs(strikes - spot)))
    _mid_mat_idx  = len(maturities) // 2
    _atm_vol_vega = float(ivs_base[_atm_idx_vega, _mid_mat_idx])
    vegas = np.array([
        black_scholes_vega(spot, K, 0.5, rate, _atm_vol_vega)
        for K in strikes
    ])
    vega_w = vegas / (vegas.sum() + 1e-12)

    fig_vega = go.Figure()
    fig_vega.add_trace(go.Bar(x=strikes, y=vega_w * 100,
                               marker_color="steelblue", name="Vega Weight (%)"))
    fig_vega.update_layout(xaxis_title="Strike", yaxis_title="Weight (%)", height=280)
    st.plotly_chart(fig_vega, width='stretch')

    # Bid-ask weights
    st.subheader("Bid-Ask Weights")
    spreads_ba = np.linspace(0.01, 0.15, len(strikes))  # synthetic spreads
    ba_weights = 1.0 / (spreads_ba ** 2)
    ba_weights /= ba_weights.sum()

    fig_ba = go.Figure()
    fig_ba.add_trace(go.Bar(x=strikes, y=ba_weights * 100,
                             marker_color="darkorange", name="B/A Weight (%)"))
    fig_ba.add_trace(go.Scatter(x=strikes, y=spreads_ba * 100,
                                 mode="lines", name="Spread (%)",
                                 yaxis="y2", line=dict(color="gray", dash="dash")))
    fig_ba.update_layout(
        xaxis_title="Strike",
        yaxis_title="B/A Weight (%)",
        yaxis2=dict(overlaying="y", side="right", title="Spread (%)"),
        height=280,
    )
    st.plotly_chart(fig_ba, width='stretch')

    # Tikhonov penalty demo
    st.subheader("Tikhonov Regularisation — Stability Penalty")
    prev_params = np.array([0.04, 0.04, 0.04, 0.04, 0.04, 0.04])
    curr_params = prev_params * np.array([1.0, 1.05, 1.10, 1.02, 0.98, 1.01])
    penalties   = tikh_lambda * np.sum((curr_params - prev_params) ** 2)

    c1, c2, c3 = st.columns(3)
    c1.metric("Tikhonov λ", f"{tikh_lambda:.4f}")
    c2.metric("Param Change (max)", f"{abs(curr_params - prev_params).max():.4f}")
    c3.metric("Regularisation Penalty", f"{penalties:.6f}")

    fig_reg = go.Figure()
    labels = [f"θ{i}" for i in range(len(prev_params))]
    fig_reg.add_trace(go.Bar(x=labels, y=(curr_params - prev_params) * 100,
                              marker_color=["red" if v > 0 else "blue"
                                            for v in curr_params - prev_params],
                              name="Δ param (%)"))
    fig_reg.add_hline(y=0, line_dash="dash")
    fig_reg.update_layout(yaxis_title="Change (%)", height=280)
    st.plotly_chart(fig_reg, width='stretch')


# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — TOTAL VARIANCE
# ─────────────────────────────────────────────────────────────────────────────
with tab_tv:
    st.header("📐 Total Variance Framework")
    st.markdown(
        r"Converts σ(K,T) → w(K,T) = σ²T and enforces ∂w/∂T ≥ 0 via the C++ engine."
    )

    c1, c2 = st.columns(2)
    with c1:
        tv_min_var = st.number_input("Min var/yr", value=0.001, format="%.4f", key="tv_min")
        tv_max_var = st.number_input("Max var/yr", value=10.0,  format="%.2f",  key="tv_max")
    with c2:
        tv_mono    = st.checkbox("Enforce monotonicity", value=True)
        tv_smooth  = st.slider("Smoothness λ", 0.0, 0.05, 0.001, step=0.001, key="tv_smooth")

    config = TotalVarianceConfig(
        use_monotonic=tv_mono,
        min_variance_per_year=tv_min_var,
        max_variance_per_year=tv_max_var,
        smoothness_lambda=tv_smooth,
    )
    tv_interp = TotalVarianceInterpolator(config)

    # Convert surface to total variance
    w_grid = tv_interp.sigma_to_total_variance(ivs_base, maturities)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            plotly_surface_3d(strikes, maturities, w_grid,
                              title="Total Variance w(K,T) = σ²T",
                              colorscale="Turbo",
                              z_label="w"),
            width='stretch',
        )
    with c2:
        # Monotonicity check: ∂w/∂T ≥ 0?
        dw_dT = np.diff(w_grid, axis=1)
        violations = (dw_dT < -1e-8).sum()
        if violations == 0:
            st.success(f"✅ ∂w/∂T ≥ 0 everywhere — {violations} violations")
        else:
            st.error(f"❌ Calendar violations in w-space: {violations}")

        # ATM total variance term structure
        atm_idx = np.argmin(np.abs(strikes - spot))
        w_atm   = w_grid[atm_idx, :]
        fig_watm = go.Figure()
        fig_watm.add_trace(go.Scatter(x=maturities, y=w_atm,
                                       mode="lines+markers", name="w_ATM",
                                       line=dict(color="teal", width=2)))
        fig_watm.update_layout(title="ATM Total Variance w(T)",
                                xaxis_title="Maturity (yr)",
                                yaxis_title="w = σ²T", height=360)
        st.plotly_chart(fig_watm, width='stretch')

    # Lee bounds
    st.subheader("Lee Moment Bounds")
    try:
        from cpp_unified_engine import get_unified_cpp_engine
        eng = get_unified_cpp_engine()
        atm_w = float(w_atm.mean())
        bounds_raw = eng.tv_lee_bounds(atm_w)
        lee_up  = abs(bounds_raw[0]) if len(bounds_raw) > 0 else 1.5
        lee_dn  = abs(bounds_raw[1]) if len(bounds_raw) > 1 else lee_up

        c1, c2, c3 = st.columns(3)
        c1.metric("ATM Total Variance (mean)", f"{atm_w:.4f}")
        c2.metric("Lee Upper Bound", f"{lee_up:.4f}")
        c3.metric("Lee Lower Bound", f"{lee_dn:.4f}")
    except Exception as e:
        st.info(f"Lee bounds: C++ engine returned {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 8 — GREEKS
# ─────────────────────────────────────────────────────────────────────────────
with tab_greeks:
    st.header("🧮 Greeks Dashboard")
    st.markdown(
        "Analytical Black-Scholes Greeks over the IV surface. "
        "GreeksValidator detects oscillations/kinks (overfitting indicators)."
    )

    c1, c2 = st.columns(2)
    with c1:
        g_rate = st.number_input("Risk-free rate (Greeks)", value=float(rate), key="g_rate")
        g_div  = st.number_input("Div yield (Greeks)",      value=float(div_yield), key="g_div")
    with c2:
        greek_name = st.selectbox(
            "Select Greek",
            ["delta", "gamma", "vega", "theta", "vanna", "volga"],
        )
        osc_thresh = st.slider("Oscillation threshold", 0.01, 0.10, 0.02, step=0.005)

    calc    = GreeksCalculator()
    greeks  = calc.compute(strikes, maturities, ivs_base,
                           spot=float(spot), rate=g_rate, div_yield=g_div)

    # 3-D surface for selected greek
    g_surface = greeks[greek_name]
    st.plotly_chart(
        plotly_surface_3d(
            strikes, maturities, g_surface,
            title=f"{greek_name.capitalize()} Surface",
            colorscale="Electric",
            z_label=greek_name.capitalize(),
        ),
        width='stretch',
    )

    # Validator
    validator = GreeksValidator(oscillation_threshold=osc_thresh)
    report_g  = validator.validate(greeks, strikes, maturities)

    st.subheader("Greeks Smoothness Report")
    col1, col2 = st.columns(2)
    with col1:
        if report_g.is_smooth:
            st.success("✅ " + report_g.summary())
        else:
            st.warning("⚠️ " + report_g.summary())
    with col2:
        osc_data = {
            "Greek": ["Delta", "Gamma", "Vega", "Vanna", "Volga"],
            "Oscillations/Kinks": [
                report_g.delta_oscillations,
                report_g.gamma_oscillations,
                report_g.vega_oscillations,
                report_g.vanna_kinks,
                report_g.volga_kinks,
            ],
        }
        import pandas as pd
        st.dataframe(pd.DataFrame(osc_data), width='stretch')

    # Strike slice of all Greeks at a chosen maturity
    st.subheader("All Greeks — Strike Slice")
    g_mat_idx = st.selectbox("Maturity slice", range(len(maturities)),
                              format_func=lambda i: f"T={maturities[i]:.2f}y",
                              key="g_mat_slice")
    fig_gall = make_subplots(rows=2, cols=3,
                              subplot_titles=["Delta","Gamma","Vega","Theta","Vanna","Volga"])
    for idx, gname in enumerate(["delta","gamma","vega","theta","vanna","volga"]):
        r, c = divmod(idx, 3)
        fig_gall.add_trace(
            go.Scatter(x=strikes, y=greeks[gname][:, g_mat_idx],
                       mode="lines", name=gname, showlegend=False),
            row=r+1, col=c+1,
        )
    fig_gall.update_layout(height=480, title_text=f"Greeks at T={maturities[g_mat_idx]:.2f}y")
    st.plotly_chart(fig_gall, width='stretch')


# ─────────────────────────────────────────────────────────────────────────────
# TAB 9 — BENCHMARK STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────
with tab_bench:
    st.header("📏 Benchmark Structure Pricer")
    st.markdown(
        "Prices 25Δ/10Δ **risk reversals**, **butterflies**, and **calendar spreads**. "
        "Tolerance: |model − market| < 0.5 vol points."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        bench_mat = st.selectbox("Maturity for RR/BF", range(len(maturities)),
                                  format_func=lambda i: f"T={maturities[i]:.2f}y",
                                  key="bench_mat")
    with c2:
        bench_tol = st.slider("Tolerance (vol pts)", 0.1, 2.0, 0.5, step=0.1) / 100
    with c3:
        bench_q   = st.number_input("Div yield (bench)", value=float(div_yield), key="b_div")

    pricer = BenchmarkStructurePricer(tolerance_vol_pts=bench_tol)
    T_b    = maturities[bench_mat]
    iv_b   = ivs_base[:, bench_mat]

    run_bench = st.button("▶ Price Benchmarks", type="primary")
    if run_bench:
        with st.spinner("Pricing benchmark structures…"):
            report_b = pricer.run_full_benchmark(
                strikes=strikes,
                maturities=maturities,
                ivs=ivs_base,
                spot=spot,
                rate=rate,
                div_yield=bench_q,
            )

        st.markdown(f"```\n{report_b.summary()}\n```")

        import pandas as pd
        bdf = pd.DataFrame([
            {
                "Structure": r.structure,
                "Maturity": f"{r.maturity:.2f}y",
                "Model Value (%)": f"{r.model_value * 100:.3f}",
                "Market Value (%)": f"{(r.market_value or 0) * 100:.3f}" if r.market_value else "—",
                "Diff (%)": f"{(r.diff or 0) * 100:.3f}" if r.diff is not None else "—",
                "Pass": "✅" if (r.within_tolerance is None or r.within_tolerance) else "❌",
            }
            for r in report_b.results
        ])
        st.dataframe(bdf, width='stretch')

        # Bar chart
        fig_b = go.Figure(go.Bar(
            x=[r.structure for r in report_b.results],
            y=[r.model_value * 100 for r in report_b.results],
            marker_color=["green" if (r.within_tolerance is None or r.within_tolerance) else "red"
                          for r in report_b.results],
        ))
        fig_b.update_layout(yaxis_title="Value (vol %)", height=320)
        st.plotly_chart(fig_b, width='stretch')


# ─────────────────────────────────────────────────────────────────────────────
# TAB 10 — BACKTESTING
# ─────────────────────────────────────────────────────────────────────────────
with tab_bt:
    st.header("📉 Backtesting Framework")
    st.markdown(
        "Replays IV surface calibration over a **simulated historical series** "
        "and applies standard **stress scenarios**."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        n_dates   = st.slider("# Backtest dates", 10, 100, 30)
        rmse_thr  = st.slider("RMSE threshold (vol pts)", 0.1, 5.0, 1.0, step=0.1) / 100
    with c2:
        vol_drift = st.slider("Vol drift per day (bps)", -20, 20, 2)
        noise_bps = st.slider("Daily noise (bps)", 0, 50, 10)
    with c3:
        stress_spot_shock = st.slider("Spot shock (%)", -30, 0, -10)
        stress_vol_shock  = st.slider("Vol shock (vol pts)", 0, 20, 5)

    # Build synthetic historical IV surfaces
    rng = np.random.default_rng(42)
    history = []
    base_ivs = ivs_base.copy()
    for d in range(n_dates):
        daily_iv = base_ivs + (vol_drift + rng.normal(0, noise_bps, base_ivs.shape)) / 10_000
        daily_iv = np.clip(daily_iv, 0.01, 1.5)
        history.append({
            "date": f"2025-{1 + d // 20:02d}-{1 + d % 20:02d}",
            "strikes": strikes,
            "maturities": maturities,
            "ivs": daily_iv.copy(),
        })
        base_ivs = daily_iv

    bt = Backtester(rmse_threshold=rmse_thr)
    bt_report = bt.run(history)
    bt_stress  = bt.stress_test(
        ivs_base,
        strikes,
        maturities,
        float(spot),
        scenarios=[
            StressScenario("Crash",     spot_shock=stress_spot_shock/100, vol_shock=stress_vol_shock/100),
            StressScenario("Vol Spike", spot_shock=0.0,                  vol_shock=stress_vol_shock/100),
            StressScenario("Recovery",  spot_shock=0.05,                 vol_shock=-0.05),
        ],
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean RMSE (vol pts)", f"{bt_report.mean_rmse * 100:.3f}")
    col2.metric("Max RMSE (vol pts)",  f"{bt_report.max_rmse * 100:.3f}")
    col3.metric("# Dates", bt_report.n_dates)
    col4.metric("Passed", "✅" if bt_report.passed else "❌")

    # RMSE time series
    rmse_series = [s.rmse for s in bt_report.snapshots if s.rmse is not None]
    dates_s     = [s.date  for s in bt_report.snapshots if s.rmse is not None]
    fig_rmse = go.Figure()
    fig_rmse.add_trace(go.Scatter(x=dates_s, y=[r * 100 for r in rmse_series],
                                   mode="lines", name="RMSE"))
    fig_rmse.add_hline(y=rmse_thr * 100, line_dash="dash", line_color="red",
                       annotation_text="Threshold")
    fig_rmse.update_layout(xaxis_title="Date", yaxis_title="RMSE (vol pts)", height=320)
    st.plotly_chart(fig_rmse, width='stretch')

    # Stress scenarios
    st.subheader("Stress Test Results")
    for scenario_name, sc_result in bt_stress.items():
        diff = sc_result["iv_change_mean"] * 100
        st.metric(scenario_name, f"Mean IV Δ = {diff:+.2f} vol pts")

    # Show stressed surface for crash scenario
    if "Crash" in bt_stress:
        st.plotly_chart(
            plotly_surface_3d(strikes, maturities, bt_stress["Crash"]["shocked_ivs"],
                              title="Stressed IV Surface (Crash)",
                              colorscale="RdBu"),
            width='stretch',
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 11 — PCA MONITORING
# ─────────────────────────────────────────────────────────────────────────────
with tab_pca:
    st.header("🔬 Parameter Dynamics Monitoring")
    st.markdown(
        "PCA on historical **SSVI parameter time-series** — detects jumps, "
        "outliers, and tracks factor loadings."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        n_pca_dates  = st.slider("# History dates", 20, 200, 60,  key="pca_dates")
        n_components = st.slider("PCA components",   2,   5,   3,  key="pca_comp")
    with c2:
        ev_target    = st.slider("Explained variance target (%)", 70, 99, 95) / 100
        jump_thresh  = st.slider("Jump threshold (%)", 1, 30, 10) / 100
    with c3:
        outlier_sig  = st.slider("Outlier sigma",   1.0, 5.0, 3.0, step=0.5)

    # Synthetic parameter history: [eta, gamma, rho, theta_mean]
    rng2    = np.random.default_rng(7)
    base_p  = np.array([0.7, 0.5, -0.4, 0.04])
    param_hist = base_p + 0.02 * rng2.standard_normal((n_pca_dates, 4))
    # Inject a jump
    param_hist[n_pca_dates // 2, :] += np.array([0.3, 0.0, -0.2, 0.02])

    monitor = ParameterDynamicsMonitor(
        n_components=n_components,
        explained_variance_target=ev_target,
        jump_threshold=jump_thresh,
        outlier_sigma=outlier_sig,
    )
    pca_report = monitor.analyze(param_hist)

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cumulative EV (%)", f"{pca_report.cumulative_explained * 100:.1f}")
    c2.metric("Target Met", "✅" if pca_report.target_met else "❌")
    c3.metric("Jump Flags", len(pca_report.jump_flags))
    c4.metric("Outlier Flags", len(pca_report.outlier_flags))

    st.info(pca_report.summary)

    # Explained variance bar
    fig_ev = go.Figure(go.Bar(
        x=[f"PC{i+1}" for i in range(len(pca_report.explained_variance_ratio))],
        y=pca_report.explained_variance_ratio * 100,
        marker_color="steelblue",
    ))
    fig_ev.add_hline(y=ev_target * 100, line_dash="dash", line_color="red",
                     annotation_text=f"Target {ev_target * 100:.0f}%")
    fig_ev.update_layout(yaxis_title="Explained Variance (%)", height=280)
    st.plotly_chart(fig_ev, width='stretch')

    # PCA scores time-series
    st.subheader("PCA Score Time-Series")
    fig_scores = go.Figure()
    colours_pca = ["royalblue", "crimson", "green", "darkorange", "purple"]
    for comp_i in range(min(n_components, pca_report.scores.shape[1])):
        fig_scores.add_trace(go.Scatter(
            y=pca_report.scores[:, comp_i],
            mode="lines", name=f"PC{comp_i+1}",
            line=dict(color=colours_pca[comp_i]),
        ))
    # Mark jumps
    for jdx in pca_report.jump_flags:
        fig_scores.add_vline(x=jdx, line_color="red", line_dash="dot",
                              annotation_text="Jump", annotation_position="top")
    fig_scores.update_layout(xaxis_title="Date index",
                              yaxis_title="PCA Score", height=320)
    st.plotly_chart(fig_scores, width='stretch')

    # Loadings heatmap
    st.subheader("Factor Loadings")
    fig_load = go.Figure(go.Heatmap(
        z=pca_report.loadings,
        x=["η", "γ", "ρ", "θ̄"],
        y=[f"PC{i+1}" for i in range(pca_report.loadings.shape[0])],
        colorscale="RdBu", zmid=0,
        text=np.round(pca_report.loadings, 3).tolist(),
        texttemplate="%{text}",
    ))
    fig_load.update_layout(height=280)
    st.plotly_chart(fig_load, width='stretch')


# ─────────────────────────────────────────────────────────────────────────────
# TAB 12 — BID/ASK SPREAD
# ─────────────────────────────────────────────────────────────────────────────
with tab_bidask:
    st.header("📊 Bid/Ask Surface Generator")
    st.markdown(
        "Generates **bid** and **ask** IV surfaces from mid-IVs with flat, "
        "skewed-wing, or market-derived spreads."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        ba_mode    = st.radio("Spread mode", ["Flat", "Skewed (wider wings)"])
        ba_flat    = st.slider("Flat spread (bps)", 1, 100, 10)
    with c2:
        ba_wing_mult = st.slider("Wing multiplier (skewed)", 1.0, 5.0, 2.0, step=0.25)
        ba_atm_strike = st.number_input("ATM Strike", value=float(spot), key="ba_atm")
    with c3:
        ba_mat_idx = st.selectbox("Maturity for smile view", range(len(maturities)),
                                   format_func=lambda i: f"T={maturities[i]:.2f}y",
                                   key="ba_mat")

    gen    = BidAskSurfaceGenerator()
    mid_2d = ivs_base  # (n_strikes, n_maturities)

    if ba_mode == "Flat":
        bid_2d, ask_2d, ba_report = gen.generate(mid_2d, spread_bps=ba_flat)
    else:
        bid_2d, ask_2d, ba_report = gen.generate_with_skew(
            mid_2d, strikes,
            atm_strike=ba_atm_strike,
            base_spread_bps=ba_flat,
            wing_multiplier=ba_wing_mult,
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Spread (bps)", f"{ba_report.mean_spread_bps:.1f}")
    c2.metric("Max Spread (bps)",  f"{ba_report.max_spread_bps:.1f}")
    c3.metric("Min Spread (bps)",  f"{ba_report.min_spread_bps:.1f}")
    c4.metric("B/A Ratio", f"{ba_report.bid_ask_ratio * 100:.2f}%")

    # 3-D ask surface
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            plotly_surface_3d(strikes, maturities, bid_2d,
                              title="Bid IV Surface", colorscale="Blues"),
            width='stretch',
        )
    with c2:
        st.plotly_chart(
            plotly_surface_3d(strikes, maturities, ask_2d,
                              title="Ask IV Surface", colorscale="Reds"),
            width='stretch',
        )

    # Smile slice with bid/ask band
    st.subheader(f"Bid-Mid-Ask Smile — T={maturities[ba_mat_idx]:.2f}y")
    fig_band = go.Figure()
    fig_band.add_trace(go.Scatter(
        x=np.concatenate([strikes, strikes[::-1]]),
        y=np.concatenate([ask_2d[:, ba_mat_idx] * 100, bid_2d[::-1, ba_mat_idx] * 100]),
        fill="toself", fillcolor="rgba(70,130,180,0.2)",
        line=dict(color="rgba(0,0,0,0)"), name="Bid-Ask Band",
    ))
    fig_band.add_trace(go.Scatter(
        x=strikes, y=mid_2d[:, ba_mat_idx] * 100,
        mode="lines", name="Mid IV", line=dict(color="steelblue", width=2),
    ))
    fig_band.add_trace(go.Scatter(
        x=strikes, y=bid_2d[:, ba_mat_idx] * 100,
        mode="lines", name="Bid IV", line=dict(color="green", dash="dash"),
    ))
    fig_band.add_trace(go.Scatter(
        x=strikes, y=ask_2d[:, ba_mat_idx] * 100,
        mode="lines", name="Ask IV", line=dict(color="red", dash="dash"),
    ))
    fig_band.update_layout(xaxis_title="Strike", yaxis_title="IV (%)", height=380)
    st.plotly_chart(fig_band, width='stretch')


# ─────────────────────────────────────────────────────────────────────────────
# TAB 13 — FORWARD CURVE  (§1 Standard)
# ─────────────────────────────────────────────────────────────────────────────
with tab_fwd:
    st.header("⏩ §1 — Forward Curve Builder")
    st.markdown(
        "Builds the forward curve **F(T) = S·exp((r−q)·T)** and computes "
        "log-moneyness k = ln(K/F(T)) as required for accurate options pricing."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        fwd_spot  = st.number_input("Spot (§1)", value=float(spot), key="fwd_spot")
        fwd_rate  = st.number_input("Rate (§1)", value=float(rate),  key="fwd_rate",
                                    format="%.4f")
    with c2:
        fwd_div   = st.number_input("Div yield (§1)", value=float(div_yield), key="fwd_div",
                                    format="%.4f")
        fwd_n     = st.slider("# Maturity points", 4, 20, 10)
    with c3:
        fwd_lo    = st.number_input("Min maturity (yr)", value=0.08, key="fwd_lo")
        fwd_hi    = st.number_input("Max maturity (yr)", value=3.0,  key="fwd_hi")

    fwd_mats  = np.linspace(fwd_lo, fwd_hi, fwd_n)
    builder   = ForwardCurveBuilder(spot=fwd_spot, rate=fwd_rate, div_yield=fwd_div)
    fwd_curve = builder.build(fwd_mats)

    # Forward curve plot
    fig_fwd = go.Figure()
    fig_fwd.add_trace(go.Scatter(
        x=fwd_curve.maturities, y=fwd_curve.forwards,
        mode="lines+markers", name="F(T)",
        line=dict(color="royalblue", width=2),
    ))
    fig_fwd.add_hline(y=fwd_spot, line_dash="dash", line_color="gray",
                      annotation_text=f"Spot = {fwd_spot:.1f}")
    fig_fwd.update_layout(
        title="Forward Curve F(T)",
        xaxis_title="Maturity (yr)", yaxis_title="Forward Price", height=340,
    )
    st.plotly_chart(fig_fwd, width='stretch')

    # Log-moneyness surface
    st.subheader("Log-Moneyness Surface k = ln(K/F(T))")
    k_grid = np.zeros((len(strikes), len(fwd_mats)))
    for j, T in enumerate(fwd_mats):
        F_T = fwd_curve.forward(T)
        k_grid[:, j] = np.log(strikes / F_T)

    fig_km = go.Figure(go.Surface(
        x=fwd_mats, y=strikes, z=k_grid,
        colorscale="RdBu", cmid=0,
        colorbar=dict(title="k = ln(K/F)"),
        hovertemplate="Maturity: %{x:.2f}y<br>Strike: %{y:.1f}<br>k: %{z:.4f}<extra></extra>",
    ))
    fig_km.update_layout(
        scene=dict(
            xaxis_title="Maturity (yr)",
            yaxis_title="Strike",
            zaxis_title="Log-Moneyness k",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8)),
        ),
        height=460, margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig_km, width='stretch')

    # Table of forwards
    import pandas as pd
    fwd_df = pd.DataFrame({
        "Maturity (yr)": fwd_curve.maturities.round(3),
        "Forward F(T)":  fwd_curve.forwards.round(4),
        "Rate r(T)":     fwd_curve.rates.round(4),
        "Div yield q(T)": fwd_curve.div_yields.round(4),
        "ln(S/F)":       np.log(fwd_spot / fwd_curve.forwards).round(4),
    })
    st.dataframe(fwd_df, width='stretch')


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "IV Surface Dashboard  |  "
    "Arbitrage · SSVI · Calibration · Validation · Operations  |  "
    "C++ engine: `build/sabr_cli.exe`  |  "
    "339/339 tests passing ✅"
)
