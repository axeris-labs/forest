"""
Incentive Modeling Page
Provides tools for DeFi incentive campaign analysis and modeling.
"""

import streamlit as st

# Import section modules
from pages.page5 import incentives_computation
from pages.page5 import campaign_simulation


def render():
    """Render the Incentive Modeling page."""
    st.title("Incentive Modeling")
    st.markdown("**Model and analyze DeFi incentive campaigns**")

    # Initialize session state for section selection
    if 'current_section' not in st.session_state:
        st.session_state.current_section = 'Incentives Computation'

    # SIDEBAR: SECTION SELECTION + SECTION-SPECIFIC CONTROLS
    with st.sidebar:
        st.markdown("### Section")
        section = st.radio(
            "Select Section:",
            ["Incentives Computation", "Campaign Simulation"],
            key="section_selector",
            label_visibility="collapsed"
        )
        st.session_state.current_section = section

        st.markdown("---")

        # Section-specific sidebar controls
        if section == "Incentives Computation":
            duration, mode = incentives_computation.render_sidebar()
        else:  # Campaign Simulation
            initial_capacity, final_capacity, campaign_duration, epoch_hours = campaign_simulation.render_sidebar()

    # MAIN CONTENT: RENDER SELECTED SECTION
    if section == "Incentives Computation":
        incentives_computation.render_main_content(mode, duration)
    else:  # Campaign Simulation
        campaign_simulation.render_main_content(initial_capacity, final_capacity, campaign_duration, epoch_hours)
