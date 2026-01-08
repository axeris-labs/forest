import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from decimal import Decimal, getcontext

# Set high precision for decimal calculations
getcontext().prec = 50

# Kamino Constants
SLOTS_PER_SECOND = 2
SLOTS_PER_YEAR = SLOTS_PER_SECOND * 60 * 60 * 24 * 365.25

def calculate_supply_rate(utilization, borrow_rate, reserve_factor):
    """Calculate supply rate using the formula: supply_rate = utilization * borrow_rate * (1 - reserve_factor)"""
    return (utilization / 100) * borrow_rate * (1 - reserve_factor / 100)

def calculate_apy_from_apr(apr):
    """Convert APR to APY using compound interest formula"""
    base = Decimal('1') + (Decimal(str(apr)) / Decimal(str(SLOTS_PER_YEAR)))
    apy = float(base) ** SLOTS_PER_YEAR - 1
    return Decimal(str(apy))

def calculate_apr_from_apy(apy):
    """Convert APY to APR using reverse compound interest formula"""
    base = Decimal(str(apy)) + Decimal('1')
    exponent = 1 / SLOTS_PER_YEAR
    result = (float(base) ** exponent - 1) * SLOTS_PER_YEAR
    return Decimal(str(result))

def calculate_kamino_supply_rate(borrow_rate, utilization, fixed_host_rate, protocol_take_rate, slot_duration_ms):
    """
    Calculate supply rate using Kamino protocol's method
    
    Args:
        borrow_rate (float): Borrow rate as percentage
        utilization (float): Utilization rate as percentage
        fixed_host_rate (float): Fixed host interest rate as percentage
        protocol_take_rate (float): Protocol take rate as percentage
        slot_duration_ms (int): Recent slot duration in milliseconds
    
    Returns:
        float: Supply rate as percentage
    """
    # Calculate slot adjustment factor
    slot_adjustment_factor = 1000 / (SLOTS_PER_SECOND * slot_duration_ms)
    
    # Convert borrow rate percentage to decimal APY
    borrow_apy_decimal = Decimal(str(borrow_rate)) / Decimal('100')
    
    # Convert borrow APY to APR
    borrow_apr = calculate_apr_from_apy(borrow_apy_decimal)
    
    # Calculate borrow rate by subtracting fixed host interest rate
    fixed_host_decimal = Decimal(str(fixed_host_rate)) / Decimal('100')
    adjusted_borrow_rate = borrow_apr - (fixed_host_decimal * Decimal(str(slot_adjustment_factor)))
    
    # Convert utilization and protocol take rate to decimals
    utilization_decimal = Decimal(str(utilization)) / Decimal('100')
    protocol_take_decimal = Decimal(str(protocol_take_rate)) / Decimal('100')
    
    # Calculate supply APR
    supply_apr = utilization_decimal * adjusted_borrow_rate * (Decimal('1') - protocol_take_decimal)
    
    # Convert supply APR to supply APY
    supply_apy = calculate_apy_from_apr(supply_apr)
    
    # Convert back to percentage
    supply_apy_percent = supply_apy * Decimal('100')
    
    return float(supply_apy_percent)

def calculate_derivatives(utilization, rates):
    """Calculate the derivative (slope) of the rate curve using numpy gradient."""
    return np.gradient(rates, utilization)

def interpolate_curve(utilization_points, rate_points, common_util):
    """Interpolate curve to common utilization points."""
    return np.interp(common_util, utilization_points, rate_points)

def create_chart(curves_data, show_supply, show_derivatives, reserve_factor, util_range, use_kamino=False, fixed_host_rate=1.0, slot_duration_ms=500):
    """Create the main chart with borrow rates, optional supply rates, and optional derivatives."""
    
    # Create common utilization range
    common_util = np.linspace(util_range[0], util_range[1], int(util_range[1] - util_range[0]) + 1)
    
    # Determine subplot configuration
    if show_derivatives:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Interest Rate Models', 'Rate Derivatives'),
            vertical_spacing=0.15,  # Increased spacing between charts
            row_heights=[0.55, 0.45]  # More balanced sizing - derivative chart gets more space
        )
    else:
        fig = go.Figure()
    
    # Color palette for curves
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, curve in enumerate(curves_data):
        if not curve['utilization'] or not curve['borrow_rates']:
            continue
            
        color = colors[i % len(colors)]
        
        # Interpolate borrow rates
        interpolated_borrow = interpolate_curve(curve['utilization'], curve['borrow_rates'], common_util)
        
        # Add borrow rate curve
        if show_derivatives:
            fig.add_trace(
                go.Scatter(
                    x=common_util,
                    y=interpolated_borrow,
                    mode='lines',
                    name=f"{curve['name']} Borrow",
                    line=dict(color=color, width=3),
                    hovertemplate='<b>%{fullData.name}</b><br>Utilization: %{x:.1f}%<br>Rate: %{y:.2f}%<extra></extra>'
                ),
                row=1, col=1
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=common_util,
                    y=interpolated_borrow,
                    mode='lines',
                    name=f"{curve['name']} Borrow",
                    line=dict(color=color, width=3),
                    hovertemplate='<b>%{fullData.name}</b><br>Utilization: %{x:.1f}%<br>Rate: %{y:.2f}%<extra></extra>'
                )
            )
        
        # Add original data points
        if show_derivatives:
            fig.add_trace(
                go.Scatter(
                    x=curve['utilization'],
                    y=curve['borrow_rates'],
                    mode='markers',
                    name=f"{curve['name']} Data Points",
                    marker=dict(color=color, size=8, symbol='circle', line=dict(color='white', width=2)),
                    showlegend=False,
                    hovertemplate='<b>%{fullData.name}</b><br>Utilization: %{x:.1f}%<br>Rate: %{y:.2f}%<extra></extra>'
                ),
                row=1, col=1
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=curve['utilization'],
                    y=curve['borrow_rates'],
                    mode='markers',
                    name=f"{curve['name']} Data Points",
                    marker=dict(color=color, size=8, symbol='circle', line=dict(color='white', width=2)),
                    showlegend=False,
                    hovertemplate='<b>%{fullData.name}</b><br>Utilization: %{x:.1f}%<br>Rate: %{y:.2f}%<extra></extra>'
                )
            )
        
        # Add supply rate curve if enabled
        if show_supply:
            if use_kamino:
                # Calculate supply rates using Kamino method for each utilization point
                interpolated_supply = []
                for util, borrow_rate in zip(common_util, interpolated_borrow):
                    kamino_supply_rate = calculate_kamino_supply_rate(
                        borrow_rate, util, fixed_host_rate, reserve_factor, slot_duration_ms
                    )
                    interpolated_supply.append(kamino_supply_rate)
                interpolated_supply = np.array(interpolated_supply)
            else:
                interpolated_supply = calculate_supply_rate(common_util, interpolated_borrow, reserve_factor)
            
            if show_derivatives:
                fig.add_trace(
                    go.Scatter(
                        x=common_util,
                        y=interpolated_supply,
                        mode='lines',
                        name=f"{curve['name']} Supply",
                        line=dict(color=color, width=2, dash='dash'),
                        hovertemplate='<b>%{fullData.name}</b><br>Utilization: %{x:.1f}%<br>Rate: %{y:.2f}%<extra></extra>'
                    ),
                    row=1, col=1
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=common_util,
                        y=interpolated_supply,
                        mode='lines',
                        name=f"{curve['name']} Supply",
                        line=dict(color=color, width=2, dash='dash'),
                        hovertemplate='<b>%{fullData.name}</b><br>Utilization: %{x:.1f}%<br>Rate: %{y:.2f}%<extra></extra>'
                    )
                )
        
        # Add derivatives if enabled
        if show_derivatives:
            borrow_derivatives = calculate_derivatives(common_util, interpolated_borrow)
            
            fig.add_trace(
                go.Scatter(
                    x=common_util,
                    y=borrow_derivatives,
                    mode='lines',
                    name=f"{curve['name']} Borrow Derivative",
                    line=dict(color=color, width=2),
                    hovertemplate='<b>%{fullData.name}</b><br>Utilization: %{x:.1f}%<br>Slope: %{y:.3f}%<extra></extra>'
                ),
                row=2, col=1
            )
            
            if show_supply:
                supply_derivatives = calculate_derivatives(common_util, interpolated_supply)
                fig.add_trace(
                    go.Scatter(
                        x=common_util,
                        y=supply_derivatives,
                        mode='lines',
                        name=f"{curve['name']} Supply Derivative",
                        line=dict(color=color, width=2, dash='dash'),
                        hovertemplate='<b>%{fullData.name}</b><br>Utilization: %{x:.1f}%<br>Slope: %{y:.3f}%<extra></extra>'
                    ),
                    row=2, col=1
                )
    
    # Update layout
    fig.update_layout(
        height=900 if show_derivatives else 500,  # Increased height for better derivative chart visibility
        plot_bgcolor='#f0f0f0',
        paper_bgcolor='#f0f0f0',
        font=dict(color='black'),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        margin=dict(r=150)
    )
    
    # Update x-axes
    if show_derivatives:
        fig.update_xaxes(
            title_text="Utilization Rate (%)",
            gridcolor='rgba(128,128,128,0.3)',
            range=[util_range[0], util_range[1]]
        )
    else:
        fig.update_xaxes(
            title_text="Utilization Rate (%)",
            gridcolor='rgba(128,128,128,0.3)',
            range=[util_range[0], util_range[1]]
        )
    
    # Update y-axes
    if show_derivatives:
        fig.update_yaxes(
            title_text="Interest Rate (%)",
            gridcolor='rgba(128,128,128,0.3)',
            row=1, col=1
        )
        fig.update_yaxes(
            title_text="Rate Change (% per % utilization)",
            gridcolor='rgba(128,128,128,0.3)',
            row=2, col=1
        )
    else:
        fig.update_yaxes(
            title_text="Interest Rate (%)",
            gridcolor='rgba(128,128,128,0.3)'
        )
    
    return fig

def render():
    st.title("Interest Rate Model Analyzer")
    st.markdown("Comprehensive tool for analyzing and comparing interest rate models with borrow rates, supply rates, and derivatives.")
    
    # Initialize session state for curves
    if 'curves' not in st.session_state:
        st.session_state.irm_curves = [
            {'name': '', 'utilization': [], 'borrow_rates': []} for _ in range(5)
        ]
    
    # Sidebar controls
    with st.sidebar:
        st.header("Chart Controls")
        
        # Utilization range
        st.subheader("Utilization Range")
        util_range = st.slider(
            "Select utilization range to visualize",
            min_value=0,
            max_value=100,
            value=(0, 100),
            step=5,
            format="%d%%"
        )
        
        st.divider()
        
        # Toggle controls
        st.subheader("Display Options")
        show_derivatives = st.toggle("Show Derivative Chart", value=False)
        show_supply = st.toggle("Show Supply Rate Curves", value=False)
        
        # Kamino supply rate calculation toggle
        use_kamino = st.toggle("Kamino Supply Rate Calculation", value=False, help="Use Kamino protocol's supply rate calculation method")
        
        # Reserve factor (only show if supply curves are enabled)
        if show_supply:
            reserve_factor = st.number_input(
                "Reserve Factor (%)",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=0.5,
                help="Percentage of interest that goes to reserves (also used as PROTOCOL_TAKE_RATE for Kamino)"
            )
        else:
            reserve_factor = 10.0
        
        # Kamino-specific parameters (only show if Kamino toggle is enabled and supply curves are shown)
        if use_kamino and show_supply:
            st.subheader("Kamino Parameters")
            fixed_host_rate = st.number_input(
                "Fixed Host Interest Rate (%)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="Fixed host interest rate for Kamino calculation"
            )
            slot_duration_ms = st.number_input(
                "Recent Slot Duration (ms)",
                min_value=100,
                max_value=2000,
                value=500,
                step=50,
                help="Recent slot duration in milliseconds"
            )
        else:
            fixed_host_rate = 1.0
            slot_duration_ms = 500
        
        st.divider()
        
        # Clear all curves button
        if st.button("Clear All Curves", type="secondary"):
            st.session_state.irm_curves = [
                {'name': '', 'utilization': [], 'borrow_rates': []} for _ in range(5)
            ]
            st.rerun()
    
    # Main content area with 4-column layout (1/4 for controls, 3/4 for charts)
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.header("Curve Management")
        st.markdown("Add up to 5 interest rate curves by defining utilization and borrow rate points.")
        
        st.divider()  # Add space between curve management and controls
        
        # Curve input interface
        curves = st.session_state.get('irm_curves', [{'name': '', 'utilization': [], 'borrow_rates': []} for _ in range(5)])
        for i in range(5):
            curve_name_display = curves[i]['name'] if i < len(curves) and curves[i]['name'] else ''
            with st.expander(f"Curve {i+1}" + (f" - {curve_name_display}" if curve_name_display else ""), expanded=i==0):
                
                # Curve name
                curve_name = st.text_input(
                    f"Curve {i+1} Name",
                    value=st.session_state.irm_curves[i]['name'],
                    key=f"name_{i}",
                    placeholder=f"e.g., Model {i+1}"
                )
                st.session_state.irm_curves[i]['name'] = curve_name
                
                # Number of points
                num_points = st.number_input(
                    f"Number of Points",
                    min_value=2,
                    max_value=10,
                    value=max(2, len(st.session_state.irm_curves[i]['utilization'])),
                    key=f"points_{i}"
                )
                
                # Ensure we have the right number of points
                current_util = st.session_state.irm_curves[i]['utilization']
                current_rates = st.session_state.irm_curves[i]['borrow_rates']
                
                # Adjust arrays to match num_points
                if len(current_util) < num_points:
                    current_util.extend([0.0] * (num_points - len(current_util)))
                    current_rates.extend([0.0] * (num_points - len(current_rates)))
                elif len(current_util) > num_points:
                    current_util = current_util[:num_points]
                    current_rates = current_rates[:num_points]
                
                st.session_state.irm_curves[i]['utilization'] = current_util
                st.session_state.irm_curves[i]['borrow_rates'] = current_rates
                
                # Input fields for each point
                st.markdown("**Define Points:**")
                for j in range(num_points):
                    col_util, col_rate = st.columns(2)
                    
                    with col_util:
                        util_val = st.number_input(
                            f"Util {j+1} (%)",
                            min_value=0.0,
                            max_value=100.0,
                            value=float(st.session_state.irm_curves[i]['utilization'][j]),
                            step=1.0,
                            key=f"util_{i}_{j}"
                        )
                        st.session_state.irm_curves[i]['utilization'][j] = util_val
                    
                    with col_rate:
                        rate_val = st.number_input(
                            f"Rate {j+1} (%)",
                            min_value=0.0,
                            max_value=1000.0,
                            value=float(st.session_state.irm_curves[i]['borrow_rates'][j]),
                            step=0.1,
                            key=f"rate_{i}_{j}"
                        )
                        st.session_state.irm_curves[i]['borrow_rates'][j] = rate_val
                
                # Clear this curve button
                if st.button(f"Clear Curve {i+1}", key=f"clear_{i}"):
                    st.session_state.irm_curves[i] = {'name': '', 'utilization': [], 'borrow_rates': []}
                    st.rerun()
    
    with col2:
        st.header("Visualization")
        
        # Filter out empty curves
        active_curves = [
            curve for curve in st.session_state.irm_curves 
            if curve['name'] and curve['utilization'] and curve['borrow_rates']
        ]
        
        if active_curves:
            # Create and display chart
            fig = create_chart(active_curves, show_supply, show_derivatives, reserve_factor, util_range, use_kamino, fixed_host_rate, slot_duration_ms)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add spacing between chart and summary
            st.divider()
            
            # Summary statistics
            st.subheader("Curve Summary")
            
            summary_data = []
            for curve in active_curves:
                if curve['utilization'] and curve['borrow_rates']:
                    # Create common utilization range for interpolation
                    common_util = np.linspace(util_range[0], util_range[1], int(util_range[1] - util_range[0]) + 1)
                    interpolated_borrow = interpolate_curve(curve['utilization'], curve['borrow_rates'], common_util)
                    
                    max_rate = np.max(interpolated_borrow)
                    min_rate = np.min(interpolated_borrow)
                    avg_rate = np.mean(interpolated_borrow)
                    
                    # Rate at key points
                    rate_50 = interpolated_borrow[min(50 - util_range[0], len(interpolated_borrow) - 1)] if 50 >= util_range[0] and 50 <= util_range[1] else "N/A"
                    rate_90 = interpolated_borrow[min(90 - util_range[0], len(interpolated_borrow) - 1)] if 90 >= util_range[0] and 90 <= util_range[1] else "N/A"
                    
                    summary_data.append({
                        'Curve': curve['name'],
                        'Max Rate (%)': f"{max_rate:.2f}",
                        'Min Rate (%)': f"{min_rate:.2f}",
                        'Avg Rate (%)': f"{avg_rate:.2f}",
                        'Rate @ 50% (%)': f"{rate_50:.2f}" if rate_50 != "N/A" else "N/A",
                        'Rate @ 90% (%)': f"{rate_90:.2f}" if rate_90 != "N/A" else "N/A"
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
        else:
            st.info("Add at least one curve with a name and data points to see the visualization.")
            
            # Show example
            st.subheader("Example Usage")
            st.markdown("""
            **To get started:**
            1. Expand "Curve 1" in the left panel
            2. Give it a name (e.g., "USDC Model")
            3. Set number of points (e.g., 4)
            4. Define utilization and rate points:
               - Point 1: 0% utilization, 0% rate
               - Point 2: 80% utilization, 4% rate
               - Point 3: 90% utilization, 8% rate
               - Point 4: 100% utilization, 20% rate
            
            **Features:**
            - Toggle supply rate curves with customizable reserve factor
            - Toggle derivative charts to see rate of change
            - Adjust utilization range to focus on specific areas
            - Compare up to 5 different models simultaneously
            """)

def main():
    render()

if __name__ == "__main__":
    main()