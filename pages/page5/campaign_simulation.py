"""
Campaign Simulation Module
Simulate multiple incentive campaigns over time with different distribution strategies.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List

# Styling constants
chart_bg = "#f0f0f0"
grid_color = "#d9d9d9"


def initialize_campaigns():
    """Initialize campaign data in session state if not present."""
    if 'incentive_campaigns' not in st.session_state:
        st.session_state.incentive_campaigns = [
            {
                'name': '',
                'type': 'variable',
                'budget': 0.0,
                'target_rate': 0.0,
                'active': False,
                'visible': True
            }
            for _ in range(5)
        ]


def simulate_campaigns(
    campaigns: List[Dict],
    initial_capacity: float,
    final_capacity: float,
    duration: int,
    epoch_hours: float
) -> Dict:
    """
    Simulate multiple incentive campaigns over time.

    Campaign Types:
    1. Variable Rate: Distributes budget evenly across all epochs
       - Budget per epoch = total_budget / num_epochs
       - Rate per epoch = (budget_per_epoch / capacity) * (365 * 24 / epoch_hours) * 100
       - This ensures all budget is used by end of campaign

    2. Fixed Rate: Maintains constant annual rate regardless of capacity
       - Required per epoch = capacity * target_rate * (epoch_hours / 24 / 365)
       - If budget available: rate = target_rate, else rate = 0
       - Budget may be exhausted before campaign ends

    3. Capped Rate: Merkl approach - per-epoch comparison
       - For each epoch: needed = capacity * target_rate * (epoch_hours / 24 / 365)
       - Available per epoch = remaining_budget / remaining_epochs
       - If needed <= available: use needed (maintain target rate, save rest)
       - If needed > available: use available (dilute rate to stretch budget)
       - Ensures budget lasts entire campaign duration
    """
    # Calculate number of epochs
    num_epochs = int(duration * 24 / epoch_hours)

    # Initialize arrays
    time_points = np.linspace(0, duration, num_epochs)
    capacity_points = np.linspace(initial_capacity, final_capacity, num_epochs)

    results = {
        'time': time_points,
        'capacity': capacity_points,
        'campaigns': {}
    }

    for campaign in campaigns:
        if not campaign['active']:
            continue

        campaign_name = campaign['name']

        # Initialize tracking arrays
        rates = np.zeros(num_epochs)
        budget_remaining = np.zeros(num_epochs)
        current_budget = campaign['budget']

        if campaign['type'] == 'variable':
            # Variable Rate: Distribute budget evenly across all epochs
            budget_per_epoch = campaign['budget'] / num_epochs

            for i in range(num_epochs):
                if current_budget >= budget_per_epoch and capacity_points[i] > 0:
                    # Calculate annualized rate (as percentage)
                    rates[i] = (budget_per_epoch / capacity_points[i]) * (365 * 24 / epoch_hours) * 100
                    current_budget -= budget_per_epoch
                else:
                    rates[i] = 0

                budget_remaining[i] = current_budget

        elif campaign['type'] == 'fixed':
            # Fixed Rate: Maintain constant annual rate
            target_rate = campaign['target_rate'] / 100  # Convert % to decimal

            for i in range(num_epochs):
                if capacity_points[i] > 0:
                    # Calculate required budget for this epoch to maintain annual target rate
                    required = capacity_points[i] * target_rate * ((epoch_hours / 24) / 365)

                    if current_budget >= required:
                        rates[i] = target_rate * 100  # Maintain target rate
                        current_budget -= required
                    else:
                        # Budget exhausted - can't maintain rate
                        rates[i] = 0
                else:
                    rates[i] = 0

                budget_remaining[i] = current_budget

        elif campaign['type'] == 'capped':
            # Capped Rate: Merkl approach - compare needed vs available per epoch
            target_rate = campaign['target_rate'] / 100

            for i in range(num_epochs):
                if capacity_points[i] == 0:
                    rates[i] = 0
                    budget_remaining[i] = current_budget
                    continue

                # Calculate needed budget to maintain target rate for this epoch
                needed = capacity_points[i] * target_rate * ((epoch_hours / 24) / 365)

                # Calculate available budget per remaining epoch
                epochs_left = num_epochs - i
                available_per_epoch = current_budget / epochs_left

                if needed <= available_per_epoch:
                    # Use only what's needed (maintain target rate), save rest for future
                    rates[i] = target_rate * 100
                    current_budget -= needed
                else:
                    # Dilute: use all available for this epoch (ensures budget lasts)
                    rates[i] = (available_per_epoch / capacity_points[i]) * (365 * 24 / epoch_hours) * 100
                    current_budget -= available_per_epoch

                budget_remaining[i] = current_budget

        results['campaigns'][campaign_name] = {
            'rates': rates,
            'budget_remaining': budget_remaining
        }

    return results


def create_rate_chart(results: Dict, visible_campaigns: List[str]) -> go.Figure:
    """Create stacked area chart showing incentive rates vs capacity."""
    fig = go.Figure()

    colors = px.colors.qualitative.Set1

    for i, (name, data) in enumerate(results['campaigns'].items()):
        # Only show campaigns that are marked as visible
        if name not in visible_campaigns:
            continue

        fig.add_trace(go.Scatter(
            x=results['capacity'],
            y=data['rates'],
            mode='lines',
            name=name,
            stackgroup='one',
            fillcolor=colors[i % len(colors)],
            line=dict(width=0.5, color=colors[i % len(colors)]),
            customdata=np.column_stack((results['time'],)),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                          'Capacity: %{x:,.0f}<br>' +
                          'Time: %{customdata[0]:.2f} days<br>' +
                          'Rate: %{y:.2f}%<br>' +
                          '<extra></extra>'
        ))

    fig.update_layout(
        title={
            "text": "<span style=\"font-weight:normal\">Total Incentive Rate vs Capacity</span>",
            "x": 0.5,
            "xanchor": "center"
        },
        title_font=dict(size=18),
        xaxis_title="Capacity (TVL)",
        yaxis_title="Annual Incentive Rate (%)",
        plot_bgcolor=chart_bg,
        paper_bgcolor=chart_bg,
        height=450,
        margin=dict(l=80, r=80, t=60, b=50)
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=grid_color,
        gridwidth=1,
        showline=True,
        linecolor=grid_color,
        linewidth=1,
        tickformat=",.0f"
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=grid_color,
        gridwidth=1,
        showline=True,
        linecolor=grid_color,
        linewidth=1
    )

    return fig


def create_budget_chart(results: Dict, visible_campaigns: List[str]) -> go.Figure:
    """Create line chart showing budget consumption over time."""
    fig = go.Figure()

    colors = px.colors.qualitative.Set1

    for i, (name, data) in enumerate(results['campaigns'].items()):
        # Only show campaigns that are marked as visible
        if name not in visible_campaigns:
            continue

        fig.add_trace(go.Scatter(
            x=results['time'],
            y=data['budget_remaining'],
            mode='lines',
            name=name,
            line=dict(width=2, color=colors[i % len(colors)]),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                          'Time: %{x:.2f} days<br>' +
                          'Budget Remaining: %{y:,.2f}<br>' +
                          '<extra></extra>'
        ))

    fig.update_layout(
        title={
            "text": "<span style=\"font-weight:normal\">Budget Consumption Over Time</span>",
            "x": 0.5,
            "xanchor": "center"
        },
        title_font=dict(size=18),
        xaxis_title="Time (days)",
        yaxis_title="Remaining Budget",
        plot_bgcolor=chart_bg,
        paper_bgcolor=chart_bg,
        height=450,
        margin=dict(l=80, r=80, t=60, b=50)
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=grid_color,
        gridwidth=1,
        showline=True,
        linecolor=grid_color,
        linewidth=1
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=grid_color,
        gridwidth=1,
        showline=True,
        linecolor=grid_color,
        linewidth=1,
        tickformat=",.0f"
    )

    return fig


def get_campaign_type_key(display_name: str) -> str:
    """Convert display name to internal key."""
    type_map = {"Variable Rate": "variable", "Fixed Rate": "fixed", "Capped Rate": "capped"}
    return type_map.get(display_name, "variable")


def get_campaign_type_display(type_key: str) -> str:
    """Convert internal key to display name."""
    display_map = {"variable": "Variable Rate", "fixed": "Fixed Rate", "capped": "Capped Rate"}
    return display_map.get(type_key, "Variable Rate")


def render_sidebar():
    """Render sidebar controls for Campaign Simulation."""
    st.markdown("### Global Settings")

    initial_capacity = st.number_input(
        "Initial Capacity",
        value=100_000_000.0,
        min_value=0.0,
        step=10_000_000.0,
        format="%.0f",
        key="sim_initial_capacity",
        help="Enter capacity in full amount (e.g., 100000000 for 100M)"
    )
    st.caption(f"≈ ${initial_capacity/1_000_000:.1f}M")

    final_capacity = st.number_input(
        "Final Capacity",
        value=200_000_000.0,
        min_value=0.0,
        step=10_000_000.0,
        format="%.0f",
        key="sim_final_capacity",
        help="Enter capacity in full amount (e.g., 200000000 for 200M)"
    )
    st.caption(f"≈ ${final_capacity/1_000_000:.1f}M")

    campaign_duration = st.number_input(
        "Campaign Duration (days)",
        value=7,
        min_value=1,
        max_value=365,
        step=1,
        key="sim_duration"
    )

    epoch_hours = st.slider(
        "Epoch Duration (hours)",
        min_value=0.5,
        max_value=4.0,
        value=1.0,
        step=0.5,
        key="sim_epoch_hours"
    )

    return initial_capacity, final_capacity, campaign_duration, epoch_hours


def render_main_content(initial_capacity, final_capacity, campaign_duration, epoch_hours):
    """Render the main content area for Campaign Simulation."""
    initialize_campaigns()

    st.markdown("### Simulate multiple incentive campaigns")
    st.markdown(f"**Capacity:** \${initial_capacity/1_000_000:.1f}M → \${final_capacity/1_000_000:.1f}M | **Duration:** {campaign_duration} days | **Epoch:** {epoch_hours}h")

    # Check if global parameters changed (trigger re-simulation)
    if 'last_sim_params' not in st.session_state:
        st.session_state.last_sim_params = {
            'initial_capacity': initial_capacity,
            'final_capacity': final_capacity,
            'duration': campaign_duration,
            'epoch_hours': epoch_hours
        }
    else:
        # Check if any parameter changed
        params_changed = (
            st.session_state.last_sim_params['initial_capacity'] != initial_capacity or
            st.session_state.last_sim_params['final_capacity'] != final_capacity or
            st.session_state.last_sim_params['duration'] != campaign_duration or
            st.session_state.last_sim_params['epoch_hours'] != epoch_hours
        )

        if params_changed:
            # Update stored params
            st.session_state.last_sim_params = {
                'initial_capacity': initial_capacity,
                'final_capacity': final_capacity,
                'duration': campaign_duration,
                'epoch_hours': epoch_hours
            }
            # Trigger re-simulation
            st.session_state.needs_simulation_update = True

    # Check if we should show results
    show_results = 'simulation_results' in st.session_state and st.session_state.simulation_results is not None

    # 3-COLUMN LAYOUT
    col_campaigns, col_charts = st.columns([2, 5])

    # LEFT COLUMN: CAMPAIGNS
    with col_campaigns:
        st.markdown("#### Campaigns")

        # Display all campaigns
        for i in range(5):
            campaign = st.session_state.incentive_campaigns[i]
            campaign_display_name = campaign['name'] if campaign['name'] else f"Campaign {i+1}"
            
            # Layout: Checkbox (if active) | Expander
            col_check, col_exp = st.columns([0.05, 0.95])
            
            # Checkbox Column
            with col_check:
                if campaign['active'] and campaign['budget'] > 0:
                    visible = st.checkbox(
                        "toggle",
                        value=campaign.get('visible', True),
                        key=f"sim_visible_{i}",
                        label_visibility="collapsed"
                    )
                     # Update visibility if changed
                    if visible != campaign.get('visible', True):
                        st.session_state.incentive_campaigns[i]['visible'] = visible
                        st.rerun()
                else:
                    st.write("") # Placeholder for alignment

            # Expander Column
            with col_exp:
                expander_label = f"{'✓ ' if campaign['active'] else ''}{campaign_display_name}"
                if not campaign['active']:
                     # If not active, maybe giving it a muted look check, or just standard
                     pass
                
                with st.expander(
                    expander_label,
                    expanded=(i == 0 and not campaign['active'])
                ):
                    # Name input
                    campaign_name = st.text_input(
                        "Name",
                        value=campaign['name'],
                        placeholder=f"Campaign {i+1}",
                        key=f"sim_campaign_name_{i}"
                    )

                    # 2-Column Layout for inputs
                    col_left, col_right = st.columns(2)

                    with col_left:
                        budget = st.number_input(
                            "Budget",
                            value=float(campaign['budget']),
                            min_value=0.0,
                            step=10_000.0,
                            format="%.0f",
                            key=f"sim_campaign_budget_{i}"
                        )
                        if budget >= 1_000_000:
                            st.caption(f"≈ ${budget/1_000_000:.2f}M")
                        elif budget >= 1_000:
                            st.caption(f"≈ ${budget/1_000:.1f}K")

                    with col_right:
                        current_type = get_campaign_type_display(campaign['type'])
                        campaign_type = st.selectbox(
                            "Type",
                            ["Variable Rate", "Fixed Rate", "Capped Rate"],
                            index=["Variable Rate", "Fixed Rate", "Capped Rate"].index(current_type),
                            key=f"sim_campaign_type_{i}"
                        )

                    # Target rate only for Fixed/Capped
                    if campaign_type in ["Fixed Rate", "Capped Rate"]:
                        target_rate = st.number_input(
                            "Target Rate (%)",
                            value=float(campaign['target_rate']),
                            min_value=0.0,
                            max_value=10000.0,
                            step=1.0,
                            format="%.2f",
                            key=f"sim_campaign_rate_{i}"
                        )
                    else:
                        target_rate = 0.0

                    # Check if values changed from session state
                    final_name = campaign_name.strip() if campaign_name.strip() else f"Campaign {i+1}"
                    campaign_type_key = get_campaign_type_key(campaign_type)

                    values_changed = (
                        campaign['name'] != final_name or
                        campaign['type'] != campaign_type_key or
                        campaign['budget'] != budget or
                        campaign['target_rate'] != target_rate
                    )

                    # Update campaign in session state if changed
                    if values_changed:
                        st.session_state.incentive_campaigns[i] = {
                            'name': final_name,
                            'type': campaign_type_key,
                            'budget': budget,
                            'target_rate': target_rate,
                            'active': budget > 0,
                            'visible': campaign.get('visible', True)
                        }
                        # Mark that we need to re-run simulation
                        st.session_state.needs_simulation_update = True

                    # Clear button
                    if st.button("Clear", key=f"sim_clear_{i}", use_container_width=True):
                        st.session_state.incentive_campaigns[i] = {
                            'name': '', 'type': 'variable', 'budget': 0.0, 'target_rate': 0.0, 'active': False, 'visible': True
                        }
                        st.session_state.needs_simulation_update = True
                        st.rerun()

        # Simulation Controls
        st.markdown("---")

        # Auto-run simulation if needed
        active_campaigns = [c for c in st.session_state.incentive_campaigns if c['active']]
        needs_update = st.session_state.get('needs_simulation_update', False)

        if active_campaigns and (needs_update or 'simulation_results' not in st.session_state):
            with st.spinner("Updating..."):
                results = simulate_campaigns(
                    st.session_state.incentive_campaigns,
                    initial_capacity,
                    final_capacity,
                    campaign_duration,
                    epoch_hours
                )
                st.session_state.simulation_results = results
                st.session_state.needs_simulation_update = False
        elif not active_campaigns:
            st.session_state.simulation_results = None

        # Clear All button
        if st.button("Clear All", use_container_width=True):
            st.session_state.incentive_campaigns = [
                {'name': '', 'type': 'variable', 'budget': 0.0, 'target_rate': 0.0, 'active': False, 'visible': True}
                for _ in range(5)
            ]
            st.session_state.simulation_results = None
            st.session_state.needs_simulation_update = False
            st.rerun()

    # RIGHT COLUMN: CHARTS & RESULTS
    with col_charts:
        st.markdown("#### Analysis")
        if show_results:
            results = st.session_state.simulation_results
            active_campaigns = [c for c in st.session_state.incentive_campaigns if c['active']]

            # Get list of visible campaign names
            visible_campaigns = [c['name'] for c in active_campaigns if c.get('visible', True)]

            # Chart 1: Rate vs Capacity
            fig_rate = create_rate_chart(results, visible_campaigns)
            st.plotly_chart(fig_rate, use_container_width=True)

            # Chart 2: Budget Consumption
            fig_budget = create_budget_chart(results, visible_campaigns)
            st.plotly_chart(fig_budget, use_container_width=True)

            # Summary Table
            st.markdown("#### Campaign Summary")
            summary_data = []
            for name, data in results['campaigns'].items():
                campaign = next(c for c in active_campaigns if c['name'] == name)
                budget_val = campaign['budget']
                budget_used = budget_val - data['budget_remaining'][-1]
                is_visible = campaign.get('visible', True)

                summary_data.append({
                    'Campaign': name,
                    'Visible': 'Yes' if is_visible else 'No',
                    'Type': campaign['type'].title(),
                    'Budget': f"${budget_val:,.0f}" if budget_val < 1_000_000 else f"${budget_val/1_000_000:.2f}M",
                    'Used': f"${budget_used:,.0f}" if budget_used < 1_000_000 else f"${budget_used/1_000_000:.2f}M",
                    'Avg Rate': f"{np.mean(data['rates']):.2f}%",
                    'Max Rate': f"{np.max(data['rates']):.2f}%"
                })

            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

            # Export
            export_data = {'Time (days)': results['time'], 'Capacity': results['capacity']}
            for name, data in results['campaigns'].items():
                export_data[f'{name} - Rate (%)'] = data['rates']
                export_data[f'{name} - Budget Remaining'] = data['budget_remaining']

            csv = pd.DataFrame(export_data).to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"incentive_simulation_{campaign_duration}days.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("Configure campaigns on the left to see results")
