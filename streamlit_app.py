import streamlit as st


st.set_page_config(page_title = 'IST 488 HW',
               initial_sidebar_state = 'expanded')
st.title ("IST 488 HW")
HW1 = st.Page('HW/HW1.py', title = 'HW 1')
HW2 = st.Page('HW/HW2.py', title = 'HW 2')
HW3 = st.Page('HW/HW3.py', title = 'HW 3', default = True)

pg = st.navigation([HW3, HW2, HW1])
pg.run()