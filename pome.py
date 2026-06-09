import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

st.set_page_config(page_title="boAt Price Analytics", layout="wide")


@st.cache_data
def load_data():
    relative_path = "boat_products_price_history_india_format.csv"
    absolute_path = r"C:\Users\sharo\Downloads\boat_products_price_history_india_format.csv"
    
    if os.path.exists(relative_path):
        df = pd.read_csv(relative_path)
    elif os.path.exists(absolute_path):
        df = pd.read_csv(absolute_path)
    else:
        raise FileNotFoundError("Could not locate the price history CSV file locally or in the repository.")
        
    df["Date_obj"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    return df.sort_values("Date_obj")

try:
    df = load_data()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.info("Please make sure the CSV exists in the same folder or at: C:\\Users\\sharo\\Downloads\\boat_products_price_history_india_format.csv")
    st.stop()

st.title("⚓ boAt Product Price Analytics")


product_list = sorted(df["Product Name"].unique())
selected_product = st.sidebar.selectbox("Choose a Product Line:", product_list)


df_p = df[df["Product Name"] == selected_product]

fig, ax = plt.subplots(figsize=(11, 4.5), facecolor='#def7ec') 
ax.set_facecolor('#ffffff')


ax.plot(df_p["Date_obj"], df_p["Price (INR)"], color="#e07a5f", linewidth=1.8, drawstyle="steps-post", zorder=3)

ax.fill_between(df_p["Date_obj"], df_p["Price (INR)"], step="post", color="#fceade", alpha=0.7, zorder=2)
ax.fill_between(df_p["Date_obj"], df_p["Price (INR)"], step="post", color="#e8f5e9", alpha=0.4, zorder=1)


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#dddddd')
ax.spines['bottom'].set_color('#dddddd')


vals = ax.get_yticks()
ax.set_yticks(vals)
ax.set_yticklabels([f"₹{int(v)}" if v >= 1000 else f"₹{int(v)}" for v in vals], color='#555555')


ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.tick_params(axis='x', colors='#555555')

# Append matching UI header items
fig.text(0.04, 0.88, "Price History", fontsize=16, fontweight='bold', color='#111111')
fig.text(0.80, 0.88, "1 Month   3 Month   [ Max ]", fontsize=10, color='#333333', 
         bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#dddddd", lw=1))

plt.tight_layout()
fig.subplots_adjust(top=0.82)

st.pyplot(fig)

p_max = df_p["Price (INR)"].max()
p_min = df_p["Price (INR)"].min()
p_avg = int(df_p["Price (INR)"].mean())

col1, col2, col3 = st.columns(3)
col1.metric("Peak Maximum Price", f"₹{p_max}")
col2.metric("Floor Minimum Price", f"₹{p_min}")
col3.metric("Calculated Baseline Average", f"₹{p_avg}")
