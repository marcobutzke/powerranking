import pandas as pd 
import numpy as np

import streamlit as st
import altair as alt

st.write('Hello!!!')

data = pd.read_feather('bra2022.feather')
st.dataframe(data)