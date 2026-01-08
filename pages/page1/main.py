import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def euler_liquidation_model(
    hf_start: float = 1.0,
    hf_end: float = 0.80,
    steps: int = 100,
    LLTV: float = 0.91,
    max_bonus: float = 0.15,
    collateral_value: float = 100.0,
):
    """Return DataFrame + arrays for plotting (corrected profit/bad debt logic)."""
    hf = np.linspace(hf_start, hf_end, steps)
    
    # Core maths
    ltv = LLTV / hf                    # debt / collateral (ratio)
    buffer = 1.0 - ltv                 # residual collateral (ratio)
    slope = max_bonus / (1 - LLTV)
    raw_bonus = np.where(hf > LLTV, slope * (1 - hf), max_bonus)
    
    # CORRECTED: Effective bonus is capped at max_bonus (15%)
    # But it equals raw_bonus when raw_bonus < max_bonus
    eff_bonus = np.minimum(raw_bonus, max_bonus)
    
    # Profit = effective bonus (capped at 15%)
    profit = eff_bonus * collateral_value
    
    # Bad debt occurs when buffer < raw_bonus
    # Bad debt = raw_bonus - min(raw_bonus, buffer)
    bad_debt = np.maximum(0, (raw_bonus - buffer)) * collateral_value
    
    df = pd.DataFrame({
        "Health Factor": hf.round(4),
        "LTV (%)": (ltv * 100).round(2),
        "Buffer (%)": (buffer * 100).round(2),
        "Raw Bonus (%)": (raw_bonus * 100).round(2),
        "Effective Bonus (%)": (eff_bonus * 100).round(2),
        "Profit ($)": profit.round(2),
        "Bad Debt ($)": bad_debt.round(2),
    })
    
    # Return arrays for further analysis/plots
    return df, hf, raw_bonus, eff_bonus, profit, bad_debt, ltv, LLTV

def analyze_bad_debt(hf, raw_bonus, profit, bad_debt, ltv, LLTV):
    """Analyze bad debt details and return formatted strings."""
    nonzero_indices = np.where(bad_debt > 0)[0]
    
    if len(nonzero_indices) > 0:
        start_idx = nonzero_indices[0]
        max_idx = int(np.argmax(bad_debt))
        
        hf_start_bd = hf[start_idx]
        raw_bonus_start = raw_bonus[start_idx]
        ltv_start = ltv[start_idx]
        ltv_start_pct = ltv_start * 100
        lltv_pct = LLTV * 100
        ltv_minus_lltv = ltv_start - LLTV
        ltv_minus_lltv_pp = ltv_minus_lltv * 100
        
        bad_debt_info = {
            "has_bad_debt": True,
            "start_hf": hf_start_bd,
            "start_raw_bonus": raw_bonus_start * 100,
            "start_ltv": ltv_start_pct,
            "ltv_diff": ltv_minus_lltv_pp,
            "lltv_pct": lltv_pct,
            "max_hf": hf[max_idx],
            "max_bad_debt": bad_debt[max_idx],
            "profit_at_max": profit[max_idx]
        }
    else:
        bad_debt_info = {"has_bad_debt": False}
    
    return bad_debt_info

def create_plotly_chart(curves_data, use_ltv=False):
    """Create a clean Plotly chart with both profit and bad debt curves on the same chart."""
    fig = go.Figure()

    # Styling constants (matching page3)
    chart_bg = "#f0f0f0"
    grid_color = "#d9d9d9"

    # Define colors for different curves
    curve_colors = px.colors.qualitative.Set1

    for i, (name, data) in enumerate(curves_data.items()):
        base_color = curve_colors[i % len(curve_colors)]

        # Select x-axis data based on mode
        x_data = data['ltv'] * 100 if use_ltv else data['hf']

        # Add profit trace (solid line, green tones)
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=data['profit'],
                mode='lines',
                name=f'{name} - Profit',
                line=dict(color='green' if name == 'Current' else base_color, width=3),
                showlegend=True
            )
        )

        # Add bad debt trace (dashed line, red tones)
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=data['bad_debt'],
                mode='lines',
                name=f'{name} - Bad Debt',
                line=dict(color='red' if name == 'Current' else base_color, width=3, dash='dash'),
                showlegend=True
            )
        )

    # Set axis title based on mode
    x_axis_title = "LTV (%)" if use_ltv else "Health Factor"
    chart_title = "Profit and Bad Debt vs LTV" if use_ltv else "Profit and Bad Debt vs Health Factor"

    # Update layout
    fig.update_layout(
        title={"text": f"<span style=\"font-weight:normal\">{chart_title}</span>", "x": 0.48, "xanchor": "center"},
        title_font=dict(size=22),
        height=500,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        margin=dict(l=80, r=120, t=60, b=50),
        plot_bgcolor=chart_bg,
        paper_bgcolor=chart_bg,
    )

    fig.update_xaxes(
        title_text=x_axis_title,
        showgrid=True,
        gridcolor=grid_color,
        gridwidth=1,
        showline=True,
        linecolor=grid_color,
        linewidth=1
    )

    fig.update_yaxes(
        title_text="Amount ($)",
        showgrid=True,
        gridcolor=grid_color,
        gridwidth=1,
        showline=True,
        linecolor=grid_color,
        linewidth=1
    )

    # X-axis direction: HF is inverted (high=safe on left), LTV is normal (low=safe on left)
    if not use_ltv:
        fig.update_xaxes(autorange="reversed")

    # Add horizontal line at y=0 for reference
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.6)

    return fig

def render():
    """Render Euler Liquidation Factor page"""
    st.title("Euler Liquidation Factor")
    st.markdown("Buffer-capped liquidation-bonus model with bad debt analysis")
    
    # Initialize session state for curves (unique key for this page)
    if 'euler_curves' not in st.session_state:
        st.session_state.euler_curves = {}
    
    # Sidebar controls
    st.sidebar.markdown("### Model Parameters")

    # Create 2-column layout for parameters
    with st.sidebar:
        col1, col2 = st.columns(2)

        with col1:
            hf_start = st.number_input("HF Start", value=1.0, min_value=0.1, max_value=2.0, step=0.01, key="hf_start")
            steps = st.number_input("Steps", value=100, min_value=10, max_value=1000, step=10, key="steps")
            max_bonus = st.number_input("Max Bonus", value=0.15, min_value=0.01, max_value=0.5, step=0.01, key="max_bonus")

        with col2:
            hf_end = st.number_input("HF End", value=0.80, min_value=0.1, max_value=1.5, step=0.01, key="hf_end")
            LLTV = st.number_input("LLTV", value=0.91, min_value=0.1, max_value=0.99, step=0.01, key="lltv")
            collateral_value = st.number_input("Collateral ($)", value=100.0, min_value=1.0, max_value=10000.0, step=1.0, key="collateral")

        st.markdown("### Display Options")
        use_ltv = st.toggle("Show X-axis as LTV (%)", value=False, help="Toggle between Health Factor and LTV (%) on x-axis. HF = LLTV / LTV")
    
    # Curve management
    st.sidebar.markdown("### Curve Management")
    curve_name = st.sidebar.text_input("Curve Name", value=f"Curve {len(st.session_state.euler_curves) + 1}")
    
    if st.sidebar.button("Add Curve"):
        if curve_name and curve_name not in st.session_state.euler_curves:
            # Calculate model for this curve
            df, hf, raw_bonus, eff_bonus, profit, bad_debt, ltv, lltv_val = euler_liquidation_model(
                hf_start, hf_end, steps, LLTV, max_bonus, collateral_value
            )
            
            st.session_state.euler_curves[curve_name] = {
                'df': df,
                'hf': hf,
                'profit': profit,
                'bad_debt': bad_debt,
                'raw_bonus': raw_bonus,
                'ltv': ltv,
                'LLTV': lltv_val,
                'params': {
                    'hf_start': hf_start,
                    'hf_end': hf_end,
                    'steps': steps,
                    'LLTV': LLTV,
                    'max_bonus': max_bonus,
                    'collateral_value': collateral_value
                }
            }
            st.sidebar.success(f"Added curve: {curve_name}")
        else:
            st.sidebar.error("Curve name already exists or is empty")
    
    # Display saved curves
    if st.session_state.euler_curves:
        st.sidebar.markdown("### Saved Curves")
        for name in list(st.session_state.euler_curves.keys()):
            col1, col2 = st.sidebar.columns([3, 1])
            col1.write(name)
            if col2.button("×", key=f"remove_{name}"):
                del st.session_state.euler_curves[name]
                st.rerun()
    
    # Calculate current model
    df, hf, raw_bonus, eff_bonus, profit, bad_debt, ltv, lltv_val = euler_liquidation_model(
        hf_start, hf_end, steps, LLTV, max_bonus, collateral_value
    )
    
    # Create chart data including current parameters
    chart_data = {"Current": {
        'hf': hf,
        'ltv': ltv,
        'profit': profit,
        'bad_debt': bad_debt
    }}

    # Add saved curves to chart data
    for name, curve in st.session_state.euler_curves.items():
        chart_data[name] = {
            'hf': curve['hf'],
            'ltv': curve['ltv'],
            'profit': curve['profit'],
            'bad_debt': curve['bad_debt']
        }

    # Display chart
    if chart_data:
        fig = create_plotly_chart(chart_data, use_ltv=use_ltv)
        st.plotly_chart(fig, use_container_width=True)
    
    # Download option (minimal, centered)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Download Data as CSV", use_container_width=True):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"euler_liquidation_model_{curve_name.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True
            )