import streamlit as st
import importlib
import os
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Multi-App Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for current page
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'dashboard'

def get_available_pages():
    """Dynamically discover available pages from the pages directory"""
    pages_dir = Path("pages")
    if not pages_dir.exists():
        return {}
    
    # Custom display names for specific pages
    custom_names = {
        'page1': 'Euler Liquidation Factor',
        'page2': 'Loan Liquidation Risk',
        'page4': 'Interest Rate Model Analyzer'
    }
    
    pages = {}
    for page_dir in pages_dir.iterdir():
        if page_dir.is_dir() and not page_dir.name.startswith('_'):
            # Look for main.py in each page directory
            main_file = page_dir / "main.py"
            if main_file.exists():
                # Use custom name if available, otherwise convert directory name
                if page_dir.name in custom_names:
                    display_name = custom_names[page_dir.name]
                else:
                    display_name = page_dir.name.replace('_', ' ').title()
                
                pages[page_dir.name] = {
                    'display_name': display_name,
                    'module_path': f"pages.{page_dir.name}.main"
                }
    
    return pages

def load_page_module(module_path):
    """Dynamically import and return page module"""
    try:
        return importlib.import_module(module_path)
    except ImportError as e:
        st.error(f"Error loading page module: {e}")
        return None

def render_navigation():
    """Render the navigation dropdown in sidebar"""
    st.sidebar.markdown("### Navigate")
    
    # Get available pages
    pages = get_available_pages()
    
    if not pages:
        st.sidebar.warning("No pages found. Please create pages in the 'pages' directory.")
        return None
    
    # Create dropdown options
    page_options = ["Dashboard"] + [pages[key]['display_name'] for key in pages.keys()]
    page_keys = ["dashboard"] + list(pages.keys())
    
    # Navigation dropdown
    selected_display = st.sidebar.selectbox(
        "Choose an app:",
        page_options,
        index=page_keys.index(st.session_state.current_page) if st.session_state.current_page in page_keys else 0,
        key="navigation_select"
    )
    
    # Update current page based on selection
    selected_key = page_keys[page_options.index(selected_display)]
    if selected_key != st.session_state.current_page:
        st.session_state.current_page = selected_key
        st.rerun()
    
    st.sidebar.markdown("---")
    return selected_key

def render_dashboard():
    """Render the main dashboard page"""
    st.title("Multi-App Dashboard")
    st.markdown("Welcome to your modular Streamlit application!")
    
    # Dashboard content
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Overview")
        st.info("This is a minimal, modular Streamlit application with:")
        st.markdown("""
        - **Collapsible sidebar navigation**
        - **Modular page structure**
        - **Independent app components**
        - **Easy to extend and maintain**
        """)
    
    with col2:
        st.markdown("### Available Apps")
        pages = get_available_pages()
        if pages:
            for key, info in pages.items():
                st.markdown(f"- **{info['display_name']}**")
        else:
            st.warning("No additional apps found. Create new apps in the 'pages' directory.")

def main():
    """Main application logic"""
    # Render navigation
    current_page = render_navigation()
    
    if current_page == "dashboard" or current_page is None:
        render_dashboard()
    else:
        # Load and render the selected page
        pages = get_available_pages()
        if current_page in pages:
            module_path = pages[current_page]['module_path']
            page_module = load_page_module(module_path)
            
            if page_module and hasattr(page_module, 'render'):
                try:
                    page_module.render()
                except Exception as e:
                    st.error(f"Error rendering page: {e}")
                    st.exception(e)
            else:
                st.error(f"Page module '{module_path}' does not have a 'render' function.")
        else:
            st.error(f"Page '{current_page}' not found.")

if __name__ == "__main__":
    main()