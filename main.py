import streamlit as st
from app.Home import Home
from app.Info import Info

if '__main__' in __name__:
    pg = st.navigation([Home,Info], position = 'top')
    pg.run()