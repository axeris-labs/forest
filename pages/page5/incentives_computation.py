"""
Incentives Computation Module
Calculate relationships between capacity, budget, and incentive rate.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Styling constants
chart_bg = "#f0f0f0"
grid_color = "#d9d9d9"


def calculate_budget(capacity: float, rate: float, duration: int) -> float:
    """Calculate budget given capacity, rate, and duration."""
    return capacity * rate * (duration / 365)


def calculate_capacity(budget: float, rate: float, duration: int) -> float:
    """Calculate capacity given budget, rate, and duration."""
    if rate == 0:
        return 0
    return budget / (rate * (duration / 365))


def calculate_rate(budget: float, capacity: float, duration: int) -> float:
    """Calculate rate given budget, capacity, and duration."""
    if capacity == 0:
        return 0
    return budget / (capacity * (duration / 365))


def create_constraint_line_chart(
    fixed_var: str,
    fixed_value: float,
    var1_name: str,
    var1_range: tuple,
    var2_name: str,
    var2_range: tuple,
    duration: int
) -> go.Figure:
    """
    Create a chart showing valid combinations of two variables
    given a fixed third variable.
    """
    # Generate points along var1 axis
    var1_points = np.linspace(var1_range[0], var1_range[1], 500)
    var2_points = []

    # Calculate corresponding var2 values based on fixed variable
    for var1 in var1_points:
        if fixed_var == 'budget':
            if var1_name == 'capacity':
                var2 = calculate_rate(fixed_value, var1, duration)
            else:
                var2 = calculate_capacity(fixed_value, var1, duration)
        elif fixed_var == 'capacity':
            if var1_name == 'budget':
                var2 = calculate_rate(var1, fixed_value, duration)
            else:
                var2 = calculate_budget(fixed_value, var1, duration)
        else:  # fixed_var == 'rate'
            if var1_name == 'budget':
                var2 = calculate_capacity(var1, fixed_value, duration)
            else:
                var2 = calculate_budget(var1, fixed_value, duration)

        var2_points.append(var2)

    var2_points = np.array(var2_points)

    # Filter to viewing window
    mask = (var2_points >= var2_range[0]) & (var2_points <= var2_range[1])
    var1_filtered = var1_points[mask]
    var2_filtered = var2_points[mask]

    # Create hover text with all three values
    if fixed_var == 'budget':
        if var1_name == 'capacity':
            hover_text = [
                f"Capacity: {v1:,.0f}<br>Rate: {v2:.2%}<br>Budget: {fixed_value:,.0f}"
                for v1, v2 in zip(var1_filtered, var2_filtered)
            ]
        else:
            hover_text = [
                f"Rate: {v1:.2%}<br>Capacity: {v2:,.0f}<br>Budget: {fixed_value:,.0f}"
                for v1, v2 in zip(var1_filtered, var2_filtered)
            ]
    elif fixed_var == 'capacity':
        if var1_name == 'budget':
            hover_text = [
                f"Budget: {v1:,.0f}<br>Rate: {v2:.2%}<br>Capacity: {fixed_value:,.0f}"
                for v1, v2 in zip(var1_filtered, var2_filtered)
            ]
        else:
            hover_text = [
                f"Rate: {v1:.2%}<br>Budget: {v2:,.0f}<br>Capacity: {fixed_value:,.0f}"
                for v1, v2 in zip(var1_filtered, var2_filtered)
            ]
    else:  # fixed_var == 'rate'
        if var1_name == 'budget':
            hover_text = [
                f"Budget: {v1:,.0f}<br>Capacity: {v2:,.0f}<br>Rate: {fixed_value:.2%}"
                for v1, v2 in zip(var1_filtered, var2_filtered)
            ]
        else:
            hover_text = [
                f"Capacity: {v1:,.0f}<br>Budget: {v2:,.0f}<br>Rate: {fixed_value:.2%}"
                for v1, v2 in zip(var1_filtered, var2_filtered)
            ]

    # Create figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=var1_filtered,
        y=var2_filtered,
        mode='lines',
        line=dict(width=3, color='#1f77b4'),
        name='Valid Combinations',
        text=hover_text,
        hovertemplate='%{text}<extra></extra>'
    ))

    # Format axis labels
    if var2_name == 'rate':
        var2_label = "Annual Rate (%)"
        tickformat_y = ".1%"
    elif var2_name == 'capacity':
        var2_label = "Capacity (TVL)"
        tickformat_y = ",.0f"
    else:
        var2_label = "Budget"
        tickformat_y = ",.0f"

    if var1_name == 'rate':
        var1_label = "Annual Rate (%)"
        tickformat_x = ".1%"
    elif var1_name == 'capacity':
        var1_label = "Capacity (TVL)"
        tickformat_x = ",.0f"
    else:
        var1_label = "Budget"
        tickformat_x = ",.0f"

    # Format fixed variable for title
    if fixed_var == 'rate':
        fixed_str = f"Rate = {fixed_value:.2%}"
    elif fixed_var == 'capacity':
        fixed_str = f"Capacity = {fixed_value:,.0f}"
    else:
        fixed_str = f"Budget = {fixed_value:,.0f}"

    fig.update_layout(
        title={
            "text": f"<span style=\"font-weight:normal\">Valid Combinations for {fixed_str} (Duration: {duration} days)</span>",
            "x": 0.5,
            "xanchor": "center"
        },
        title_font=dict(size=18),
        xaxis_title=var1_label,
        yaxis_title=var2_label,
        plot_bgcolor=chart_bg,
        paper_bgcolor=chart_bg,
        height=500,
        margin=dict(l=80, r=80, t=60, b=50),
        showlegend=False
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=grid_color,
        gridwidth=1,
        showline=True,
        linecolor=grid_color,
        linewidth=1,
        tickformat=tickformat_x
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=grid_color,
        gridwidth=1,
        showline=True,
        linecolor=grid_color,
        linewidth=1,
        tickformat=tickformat_y
    )

    return fig


def render_sidebar():
    """Render sidebar controls for Incentives Computation."""
    st.markdown("### Settings")

    duration = st.number_input(
        "Campaign Duration (days)",
        value=7,
        min_value=1,
        max_value=365,
        step=1,
        key="comp_duration"
    )

    st.markdown("---")

    mode = st.radio(
        "Mode:",
        ["Single Variable", "Two Variables"],
        key="comp_mode"
    )

    st.markdown("---")

    if mode == "Single Variable":
        fixed_var = st.selectbox(
            "Fixed Variable:",
            ["Budget", "Capacity", "Rate"],
            key="comp_fixed_var"
        )

        if fixed_var == "Budget":
            fixed_value = st.number_input(
                "Budget:",
                value=100_000.0,
                min_value=0.0,
                step=10_000.0,
                format="%.0f",
                key="comp_budget_fixed"
            )
            st.caption(f"≈ ${fixed_value/1_000:.0f}K")
        elif fixed_var == "Capacity":
            fixed_value = st.number_input(
                "Capacity:",
                value=50_000_000.0,
                min_value=0.0,
                step=10_000_000.0,
                format="%.0f",
                key="comp_capacity_fixed"
            )
            st.caption(f"≈ ${fixed_value/1_000_000:.1f}M")
        else:  # Rate
            fixed_value = st.number_input(
                "Annual Rate (%):",
                value=5.0,
                min_value=0.01,
                max_value=100.0,
                step=0.5,
                format="%.2f",
                key="comp_rate_fixed"
            )
    else:  # Two Variables
        calc_for = st.selectbox(
            "Calculate:",
            ["Budget", "Capacity", "Rate"],
            key="comp_calc_for"
        )

    return duration, mode


def render_main_content(mode, duration):
    """Render the main content area for Incentives Computation."""
    st.markdown("### Calculate relationships between capacity, budget, and incentive rate")
    st.markdown(f"**Invariant:** `budget = capacity × rate × (duration / 365)` | Duration: **{duration} days**")

    if mode == "Single Variable":
        render_single_variable_mode(duration)
    else:
        render_two_variables_mode(duration)


def render_single_variable_mode(duration):
    """Render Single Variable mode."""
    fixed_var = st.session_state.comp_fixed_var

    if fixed_var == "Budget":
        fixed_value = st.session_state.comp_budget_fixed
    elif fixed_var == "Capacity":
        fixed_value = st.session_state.comp_capacity_fixed
    else:
        fixed_value = st.session_state.comp_rate_fixed / 100

    st.markdown(f"#### Showing combinations for fixed **{fixed_var}**")

    # Compact range inputs
    if fixed_var == "Budget":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            cap_min = st.number_input("Capacity Min", value=10_000_000.0, min_value=0.0, step=10_000_000.0, format="%.0f", key="comp_cap_min")
            st.caption(f"${cap_min/1_000_000:.0f}M")
        with col2:
            cap_max = st.number_input("Capacity Max", value=100_000_000.0, min_value=cap_min, step=10_000_000.0, format="%.0f", key="comp_cap_max")
            st.caption(f"${cap_max/1_000_000:.0f}M")
        with col3:
            rate_min = st.number_input("Rate Min (%)", value=1.0, min_value=0.01, step=0.5, key="comp_rate_min")
        with col4:
            rate_max = st.number_input("Rate Max (%)", value=20.0, min_value=rate_min, step=0.5, key="comp_rate_max")

        var1_name, var1_range = "capacity", (cap_min, cap_max)
        var2_name, var2_range = "rate", (rate_min / 100, rate_max / 100)

    elif fixed_var == "Capacity":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            budget_min = st.number_input("Budget Min", value=50_000.0, min_value=0.0, step=10_000.0, format="%.0f", key="comp_budget_min")
            st.caption(f"${budget_min/1_000:.0f}K")
        with col2:
            budget_max = st.number_input("Budget Max", value=500_000.0, min_value=budget_min, step=50_000.0, format="%.0f", key="comp_budget_max")
            st.caption(f"${budget_max/1_000:.0f}K")
        with col3:
            rate_min = st.number_input("Rate Min (%)", value=1.0, min_value=0.01, step=0.5, key="comp_rate_min2")
        with col4:
            rate_max = st.number_input("Rate Max (%)", value=20.0, min_value=rate_min, step=0.5, key="comp_rate_max2")

        var1_name, var1_range = "budget", (budget_min, budget_max)
        var2_name, var2_range = "rate", (rate_min / 100, rate_max / 100)

    else:  # Rate
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            budget_min = st.number_input("Budget Min", value=50_000.0, min_value=0.0, step=10_000.0, format="%.0f", key="comp_budget_min2")
            st.caption(f"${budget_min/1_000:.0f}K")
        with col2:
            budget_max = st.number_input("Budget Max", value=500_000.0, min_value=budget_min, step=50_000.0, format="%.0f", key="comp_budget_max2")
            st.caption(f"${budget_max/1_000:.0f}K")
        with col3:
            cap_min = st.number_input("Capacity Min", value=10_000_000.0, min_value=0.0, step=10_000_000.0, format="%.0f", key="comp_cap_min2")
            st.caption(f"${cap_min/1_000_000:.0f}M")
        with col4:
            cap_max = st.number_input("Capacity Max", value=100_000_000.0, min_value=cap_min, step=10_000_000.0, format="%.0f", key="comp_cap_max2")
            st.caption(f"${cap_max/1_000_000:.0f}M")

        var1_name, var1_range = "budget", (budget_min, budget_max)
        var2_name, var2_range = "capacity", (cap_min, cap_max)

    # Create and display chart
    fig = create_constraint_line_chart(
        fixed_var.lower(),
        fixed_value,
        var1_name,
        var1_range,
        var2_name,
        var2_range,
        duration
    )
    st.plotly_chart(fig, use_container_width=True)


def render_two_variables_mode(duration):
    """Render Two Variables mode."""
    calc_for = st.session_state.comp_calc_for

    st.markdown(f"#### Calculate **{calc_for}** from two inputs")

    if calc_for == "Budget":
        col1, col2 = st.columns(2)
        with col1:
            capacity_input = st.number_input("Capacity:", value=50_000_000.0, min_value=0.0, step=10_000_000.0, format="%.0f", key="comp_capacity_input")
            st.caption(f"${capacity_input/1_000_000:.1f}M")
        with col2:
            rate_input = st.number_input("Annual Rate (%):", value=5.0, min_value=0.01, max_value=1000.0, step=0.5, format="%.2f", key="comp_rate_input")

        rate_input = rate_input / 100
        result = calculate_budget(capacity_input, rate_input, duration)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric("Calculated Budget", f"${result:,.0f}")
            if result >= 1_000_000:
                st.caption(f"≈ ${result/1_000_000:.2f}M")
            elif result >= 1_000:
                st.caption(f"≈ ${result/1_000:.1f}K")

    elif calc_for == "Capacity":
        col1, col2 = st.columns(2)
        with col1:
            budget_input = st.number_input("Budget:", value=100_000.0, min_value=0.0, step=10_000.0, format="%.0f", key="comp_budget_input")
            st.caption(f"${budget_input/1_000:.0f}K")
        with col2:
            rate_input = st.number_input("Annual Rate (%):", value=5.0, min_value=0.01, max_value=1000.0, step=0.5, format="%.2f", key="comp_rate_input2")

        rate_input = rate_input / 100
        result = calculate_capacity(budget_input, rate_input, duration)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric("Calculated Capacity", f"{result:,.0f}")
            if result >= 1_000_000:
                st.caption(f"≈ ${result/1_000_000:.2f}M")

    else:  # Rate
        col1, col2 = st.columns(2)
        with col1:
            budget_input = st.number_input("Budget:", value=100_000.0, min_value=0.0, step=10_000.0, format="%.0f", key="comp_budget_input2")
            st.caption(f"${budget_input/1_000:.0f}K")
        with col2:
            capacity_input = st.number_input("Capacity:", value=50_000_000.0, min_value=0.0, step=10_000_000.0, format="%.0f", key="comp_capacity_input2")
            st.caption(f"${capacity_input/1_000_000:.1f}M")

        result = calculate_rate(budget_input, capacity_input, duration)
        result_pct = result * 100

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric("Calculated Annual Rate", f"{result_pct:.2f}%")
