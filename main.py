import pandas as pd 
import numpy as np

import streamlit as st
import altair as alt

st.write('Hello!!!')

data = pd.read_excel('BRA2022.xlsx')
st.dataframe(data)