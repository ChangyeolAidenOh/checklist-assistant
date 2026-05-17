"""
Coverage Checklist Assistant - Streamlit Dashboard.

5-tab interface:
  1. My Coverage    - product summary input + parsed coverage display
  2. My Profile     - demographic profile + interest radar chart
  3. Check Items    - gap analysis with priority ranking
  4. Consultation Checklist - final checklist for consultation
  5. Pipeline Trace - node-by-node execution trace + cost breakdown
"""

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from config import STREAMLIT_PAGE_TITLE, STREAMLIT_PAGE_ICON, STREAMLIT_LAYOUT

st.set_page_config(
    page_title=STREAMLIT_PAGE_TITLE,
    page_icon=STREAMLIT_PAGE_ICON,
    layout=STREAMLIT_LAYOUT,
)

st.title(f"{STREAMLIT_PAGE_ICON} {STREAMLIT_PAGE_TITLE}")

# Import tabs
from dashboard.tabs.tab1_my_coverage import render as render_tab1
from dashboard.tabs.tab2_my_profile import render as render_tab2
from dashboard.tabs.tab3_check_items import render as render_tab3
from dashboard.tabs.tab4_checklist import render as render_tab4
from dashboard.tabs.tab5_trace import render as render_tab5

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "My Coverage",
    "My Profile",
    "Check Items",
    "Consultation Checklist",
    "Pipeline Trace",
])

with tab1:
    render_tab1()
with tab2:
    render_tab2()
with tab3:
    render_tab3()
with tab4:
    render_tab4()
with tab5:
    render_tab5()
