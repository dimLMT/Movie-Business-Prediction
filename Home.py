import streamlit as st
# Create multiple pages from apps in pages folder
st.set_page_config()
st.write('# Analyzing movie reviews')
st.subheader(':movie_camera: :cinema: :clapper:')
st.title(':point_left: *Choose page on sidebar:*')
st.sidebar.success('Select a page above')