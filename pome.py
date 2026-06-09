import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import datetime
import os
st.set_page_config(page_title="boAt Price Analytics", layout="wide")
# Set page layout configuration
st.set_page_config(
    page_title="boAt Price Tracker & ML Predictor",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Apply custom styling for a premium dark-mode feel
st.markdown("""
<style>
    /* Main container styling */
    .reportview-container {
        background: #0f172a;
    }
    
    /* Rounded metric cards */
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #e2e8f0;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #94a3b8;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-sub {
        font-size: 0.75rem;
        margin-top: 0.5rem;
    }
    
    /* Buy Confidence Advice Box */
    .advice-card {
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-left: 5px solid;
    }
    
    /* Recommendation badges */
    .badge {
        padding: 0.25rem 0.6rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: bold;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)
# Load data helper function
@st.cache_data
def load_data():
    relative_path = "boat_products_price_history_india_format.csv"
        raise FileNotFoundError("Could not locate the price history CSV file locally or in the repository.")
        
    df["Date_obj"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    return df.sort_values("Date_obj")
    df = df.sort_values("Date_obj")
    return df
try:
    df = load_data()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.info("Please make sure the CSV exists in the same folder or at: C:\\Users\\sharo\\Downloads\\boat_products_price_history_india_format.csv")
    st.stop()
st.title("⚓ boAt Product Price Analytics")
# Helper function to categorize products based on name
def get_category(name):
    name_lower = name.lower()
    if "bar" in name_lower or "aavante" in name_lower:
        return "Soundbars"
    elif "airdopes" in name_lower:
        return "Wireless Earbuds"
    elif "lunar" in name_lower or "wave" in name_lower or "ultima" in name_lower or "ember" in name_lower:
        return "Smartwatches"
    elif "stone" in name_lower:
        return "Bluetooth Speakers"
    elif "rockerz" in name_lower or "nirvana" in name_lower:
        if any(h in name_lower for h in ["751", "eutopia", "411", "412", "413"]):
            return "Headphones"
        else:
            return "Neckbands / Earphones"
    return "Other Accessories"
# Add categories to dataframe
df["Category"] = df["Product Name"].apply(get_category)
product_list = sorted(df["Product Name"].unique())
selected_product = st.sidebar.selectbox("Choose a Product Line:", product_list)
# Sidebar UI
st.sidebar.markdown("<h2 style='text-align: center; color: #38bdf8;'>⚓ boAt Engine</h2>", unsafe_allow_html=True)
st.sidebar.write("Analyze price variations and predict buying opportunities using Machine Learning.")
# Sidebar Filters
category_list = ["All Categories"] + sorted(df["Category"].unique().tolist())
selected_category = st.sidebar.selectbox("Filter Category:", category_list)
df_p = df[df["Product Name"] == selected_product]
if selected_category != "All Categories":
    filtered_df = df[df["Category"] == selected_category]
else:
    filtered_df = df
fig, ax = plt.subplots(figsize=(11, 4.5), facecolor='#def7ec') 
ax.set_facecolor('#ffffff')
product_list = sorted(filtered_df["Product Name"].unique().tolist())
selected_product = st.sidebar.selectbox("Select Product Line:", product_list)
# Filter product dataset
df_p = df[df["Product Name"] == selected_product].sort_values("Date_obj")
ax.plot(df_p["Date_obj"], df_p["Price (INR)"], color="#e07a5f", linewidth=1.8, drawstyle="steps-post", zorder=3)
# Main Page Header
st.markdown(f"<h1 style='margin-bottom:0;'>🏷️ {selected_product}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color: #64748b; font-size:1.1rem; margin-top:0;'>Category: <b>{get_category(selected_product)}</b> | Tracking <b>{len(df_p)} days</b> of historical prices</p>", unsafe_allow_html=True)
ax.fill_between(df_p["Date_obj"], df_p["Price (INR)"], step="post", color="#fceade", alpha=0.7, zorder=2)
ax.fill_between(df_p["Date_obj"], df_p["Price (INR)"], step="post", color="#e8f5e9", alpha=0.4, zorder=1)
# Machine Learning & Feature Engineering
@st.cache_data(ttl=600)  # cache training for 10 mins per product
def train_and_forecast(df_product):
    df_sorted = df_product.sort_values("Date_obj").copy()
    
    # Feature engineering
    df_sorted['Price_Lag_1'] = df_sorted['Price (INR)'].shift(1)
    df_sorted['Price_Lag_2'] = df_sorted['Price (INR)'].shift(2)
    df_sorted['Price_Lag_7'] = df_sorted['Price (INR)'].shift(7)
    df_sorted['Rolling_Mean_7'] = df_sorted['Price (INR)'].shift(1).rolling(7).mean()
    df_sorted['Rolling_Std_7'] = df_sorted['Price (INR)'].shift(1).rolling(7).std()
    df_sorted['Rolling_Mean_30'] = df_sorted['Price (INR)'].shift(1).rolling(30).mean()
    df_sorted['Rolling_Min_30'] = df_sorted['Price (INR)'].shift(1).rolling(30).min()
    df_sorted['Rolling_Max_30'] = df_sorted['Price (INR)'].shift(1).rolling(30).max()
    
    df_sorted['Price_Percentile_30'] = (df_sorted['Price_Lag_1'] - df_sorted['Rolling_Min_30']) / (df_sorted['Rolling_Max_30'] - df_sorted['Rolling_Min_30'] + 1e-5)
    
    df_sorted['DayOfWeek'] = df_sorted['Date_obj'].dt.dayofweek
    df_sorted['DayOfMonth'] = df_sorted['Date_obj'].dt.day
    df_sorted['Month'] = df_sorted['Date_obj'].dt.month
    
    feature_cols = [
        'Price_Lag_1', 'Price_Lag_2', 'Price_Lag_7',
        'Rolling_Mean_7', 'Rolling_Std_7',
        'Rolling_Mean_30', 'Rolling_Min_30', 'Rolling_Max_30',
        'Price_Percentile_30', 'DayOfWeek', 'DayOfMonth', 'Month'
    ]
    
    latest_row = df_sorted.iloc[-1:]
    latest_features = latest_row[feature_cols]
    
    # Check if we have enough non-null features in latest row
    if latest_features.isnull().values.any():
        # Fallback if history is too short to construct features
        last_price = df_sorted['Price (INR)'].values[-1]
        return [last_price]*7, [0.0]*7, {}, feature_cols, 0.0, 0.0
        
    forecast_prices = []
    maes = []
    r2_scores_list = []
    feature_importances_list = []
    
    for lead in range(1, 8):
        df_sorted['Target'] = df_sorted['Price (INR)'].shift(-lead)
        train_data = df_sorted.dropna(subset=['Target'] + feature_cols)
        
        if len(train_data) < 15:
            # Fallback if too few rows
            forecast_prices.append(df_sorted['Price (INR)'].values[-1])
            maes.append(0.0)
            r2_scores_list.append(0.0)
            continue
            
        X = train_data[feature_cols]
        y = train_data['Target']
        
        # Split train/validation
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Fit evaluation model
        eval_model = RandomForestRegressor(n_estimators=50, random_state=42)
        eval_model.fit(X_train, y_train)
        preds = eval_model.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        maes.append(mae)
        r2_scores_list.append(max(0.0, r2)) # Clip r2 to >= 0
        
        # Fit full model
        full_model = RandomForestRegressor(n_estimators=50, random_state=42)
        full_model.fit(X, y)
        
        pred = full_model.predict(latest_features)[0]
        forecast_prices.append(pred)
        feature_importances_list.append(full_model.feature_importances_)
        
    avg_mae = np.mean(maes) if maes else 0.0
    avg_r2 = np.mean(r2_scores_list) if r2_scores_list else 0.0
    avg_importances = np.mean(feature_importances_list, axis=0) if feature_importances_list else [0.0]*len(feature_cols)
    importances_dict = dict(zip(feature_cols, avg_importances))
    
    return forecast_prices, maes, importances_dict, feature_cols, avg_mae, avg_r2
with st.spinner("🧠 Re-training Machine Learning models..."):
    forecast_prices, maes, importances, features, avg_mae, avg_r2 = train_and_forecast(df_p)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#dddddd')
ax.spines['bottom'].set_color('#dddddd')
# Basic Stats Calculations
latest_price = int(df_p["Price (INR)"].values[-1])
latest_date = df_p["Date_obj"].values[-1]
avg_price = int(df_p["Price (INR)"].mean())
min_price = int(df_p["Price (INR)"].min())
max_price = int(df_p["Price (INR)"].max())
# Dates for min/max prices
min_price_date = df_p[df_p["Price (INR)"] == min_price]["Date"].values[0]
max_price_date = df_p[df_p["Price (INR)"] == max_price]["Date"].values[0]
vals = ax.get_yticks()
ax.set_yticks(vals)
ax.set_yticklabels([f"₹{int(v)}" if v >= 1000 else f"₹{int(v)}" for v in vals], color='#555555')
# Previous price to see change
prev_price = int(df_p["Price (INR)"].values[-2]) if len(df_p) > 1 else latest_price
price_change = latest_price - prev_price
price_change_pct = (price_change / prev_price) * 100 if prev_price > 0 else 0.0
# -----------------
# CONFIDENCE SCORE CALCULATION
# -----------------
# 1. Percentile Rank score (relative to recent 30-day window and overall history)
recent_30 = df_p.tail(30)
recent_min = recent_30["Price (INR)"].min()
recent_max = recent_30["Price (INR)"].max()
range_recent = recent_max - recent_min
recent_percentile = (latest_price - recent_min) / range_recent if range_recent > 0 else 0.5
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.tick_params(axis='x', colors='#555555')
overall_range = max_price - min_price
overall_percentile = (latest_price - min_price) / overall_range if overall_range > 0 else 0.5
# Append matching UI header items
fig.text(0.04, 0.88, "Price History", fontsize=16, fontweight='bold', color='#111111')
fig.text(0.80, 0.88, "1 Month   3 Month   [ Max ]", fontsize=10, color='#333333', 
         bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#dddddd", lw=1))
percentile_score = (1.0 - recent_percentile) * 0.7 + (1.0 - overall_percentile) * 0.3
plt.tight_layout()
fig.subplots_adjust(top=0.82)
# 2. ML Forecast Trend Score
forecast_5d = forecast_prices[4] # 5-day prediction
predicted_change_5d = (forecast_5d - latest_price) / latest_price
st.pyplot(fig)
# If price is predicted to drop, buying now is sub-optimal unless it's already near min
# If price is predicted to rise, buying now is highly recommended
if predicted_change_5d >= 0.05:
    trend_score = 1.0
elif predicted_change_5d <= -0.05:
    trend_score = 0.0
else:
    # Scale from 0.0 to 1.0 between -5% and +5%
    trend_score = (predicted_change_5d + 0.05) / 0.10
p_max = df_p["Price (INR)"].max()
p_min = df_p["Price (INR)"].min()
p_avg = int(df_p["Price (INR)"].mean())
# Combine scores (60% based on historical placement, 40% based on ML forecast trend)
confidence_score = int(np.clip((0.6 * percentile_score + 0.4 * trend_score) * 100, 5, 98))
col1, col2, col3 = st.columns(3)
col1.metric("Peak Maximum Price", f"₹{p_max}")
col2.metric("Floor Minimum Price", f"₹{p_min}")
# Define Recommendation status
if confidence_score >= 85:
    recommendation = "STRONG BUY"
    rec_color = "#10b981" # Mint/Green
    rec_text = "Price is at or near local bottoms, and model forecasts an upward movement. Excellent buying opportunity!"
    advice_bg = "#ecfdf5"
    advice_border = "#10b981"
    advice_font = "#065f46"
elif confidence_score >= 65:
    recommendation = "BUY"
    rec_color = "#34d399"
    rec_text = "Good entry point. Price is below average, with predicted steady prices or potential increments soon."
    advice_bg = "#f0fdf4"
    advice_border = "#34d399"
    advice_font = "#166534"
elif confidence_score >= 45:
    recommendation = "HOLD / NEUTRAL"
    rec_color = "#fbbf24" # Orange/Yellow
    rec_text = "Price is stable and close to average. Standard purchasing value. Buy if required immediately."
    advice_bg = "#fffbeb"
    advice_border = "#fbbf24"
    advice_font = "#92400e"
elif confidence_score >= 25:
    recommendation = "WAIT / POSTPONE"
    rec_color = "#f97316" # Dark Orange
    rec_text = "Price is currently inflated compared to recent averages. Model expects a downward correction soon."
    advice_bg = "#fff7ed"
    advice_border = "#f97316"
    advice_font = "#9a3412"
else:
    recommendation = "AVOID NOW"
    rec_color = "#ef4444" # Red
    rec_text = "Price is at or near all-time peak levels. Highly inflated. Postpone purchase until price drops."
    advice_bg = "#fef2f2"
    advice_border = "#ef4444"
    advice_font = "#991b1b"
# --- KPI METRICS DISPLAY ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    change_color = "#ef4444" if price_change > 0 else "#10b981" if price_change < 0 else "#64748b"
    change_symbol = "+" if price_change > 0 else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">CURRENT PRICE</div>
        <div class="metric-value">₹{latest_price:,}</div>
        <div class="metric-sub" style="color: {change_color}; font-weight: bold;">
            {change_symbol}₹{price_change:,} ({change_symbol}{price_change_pct:.1f}%) vs Yesterday
        </div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">HISTORICAL MINIMUM</div>
        <div class="metric-value" style="color: #10b981;">₹{min_price:,}</div>
        <div class="metric-sub" style="color: #94a3b8;">Recorded on {min_price_date}</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">HISTORICAL MAXIMUM</div>
        <div class="metric-value" style="color: #ef4444;">₹{max_price:,}</div>
        <div class="metric-sub" style="color: #94a3b8;">Recorded on {max_price_date}</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    savings_pct = ((avg_price - latest_price) / avg_price) * 100 if avg_price > 0 else 0.0
    savings_text = f"Save ₹{avg_price - latest_price:,} ({savings_pct:.1f}%)" if latest_price < avg_price else f"Inflated by ₹{latest_price - avg_price:,} ({abs(savings_pct):.1f}%)"
    savings_color = "#10b981" if latest_price < avg_price else "#f97316"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">AVERAGE HISTORICAL PRICE</div>
        <div class="metric-value">₹{avg_price:,}</div>
        <div class="metric-sub" style="color: {savings_color}; font-weight: bold;">{savings_text}</div>
    </div>
    """, unsafe_allow_html=True)
st.write("---")
# Layout: Main Chart (Left 70%) & Recommendation card (Right 30%)
main_col, side_col = st.columns([7, 3])
with main_col:
    st.subheader("📈 Price History & ML 7-Day Forecast")
    
    # Generate dates for forecast
    last_date = df_p["Date_obj"].max()
    forecast_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 8)]
    
    # Construct historical series for Plotly
    fig = go.Figure()
    
    # 1. Historical Prices
    fig.add_trace(go.Scatter(
        x=df_p["Date_obj"],
        y=df_p["Price (INR)"],
        mode='lines',
        name='Historical Price',
        line=dict(color='#e07a5f', width=2),
        fill='tozeroy',
        fillcolor='rgba(252, 234, 222, 0.4)',
        hovertemplate='Date: %{x|%d %b %Y}<br>Price: ₹%{y:,}<extra></extra>'
    ))
    
    # 2. Predicted Prices
    # Add a transition point connecting the last historical price to the first forecast price
    full_forecast_dates = [last_date] + forecast_dates
    full_forecast_prices = [latest_price] + forecast_prices
    
    fig.add_trace(go.Scatter(
        x=full_forecast_dates,
        y=full_forecast_prices,
        mode='lines+markers',
        name='ML Forecast (7 Days)',
        line=dict(color='#38bdf8', width=2, dash='dash'),
        marker=dict(size=6, symbol='circle'),
        hovertemplate='Forecast Date: %{x|%d %b %Y}<br>Expected Price: ₹%{y:,.0f}<extra></extra>'
    ))
    
    # 3. Horizontal Average Line
    fig.add_trace(go.Scatter(
        x=[df_p["Date_obj"].min(), forecast_dates[-1]],
        y=[avg_price, avg_price],
        mode='lines',
        name='Avg Price Baseline',
        line=dict(color='#64748b', width=1, dash='dot'),
        hovertemplate='Average Price: ₹%{y:,}<extra></extra>'
    ))
    
    # Markers for Min and Max points
    fig.add_trace(go.Scatter(
        x=[pd.to_datetime(min_price_date, format="%d-%m-%Y")],
        y=[min_price],
        mode='markers',
        name='All-Time Low',
        marker=dict(color='#10b981', size=12, symbol='triangle-up'),
        hovertemplate='All-Time Low: ₹%{y:,}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=[pd.to_datetime(max_price_date, format="%d-%m-%Y")],
        y=[max_price],
        mode='markers',
        name='All-Time High',
        marker=dict(color='#ef4444', size=12, symbol='triangle-down'),
        hovertemplate='All-Time High: ₹%{y:,}<extra></extra>'
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(
            showgrid=True,
            gridcolor='#1e293b',
            title="Timeline",
            tickformat='%d %b'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#1e293b',
            title="Price (INR)",
            tickprefix="₹"
        ),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
with side_col:
    st.subheader("💡 Buy Decision Engine")
    
    # Progress ring / styled gauge for Confidence Score
    confidence_color = rec_color
    st.markdown(f"""
    <div style="background-color: #1e293b; border-radius: 12px; border: 1px solid #334155; padding: 1.5rem; text-align: center;">
        <span class="badge" style="background-color: {confidence_color}; color: #0f172a; font-size: 1.1rem; padding: 0.4rem 1rem;">
            {recommendation}
        </span>
        <div style="margin: 1.5rem 0;">
            <span style="font-size: 4rem; font-weight: 800; color: {confidence_color};">{confidence_score}%</span>
            <br>
            <span style="color: #94a3b8; font-weight: bold; font-size: 0.9rem; letter-spacing: 0.1em;">BUY CONFIDENCE SCORE</span>
        </div>
        <div class="advice-card" style="background-color: {advice_bg}; border-color: {advice_border}; color: {advice_font}; font-size:0.95rem; text-align: left; margin:0;">
            {rec_text}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Price Trend description
    trend_symbol = "📈" if predicted_change_5d > 0.01 else "📉" if predicted_change_5d < -0.01 else "➡️"
    trend_desc = f"increase by {predicted_change_5d*100:.1f}%" if predicted_change_5d > 0 else f"decrease by {abs(predicted_change_5d)*100:.1f}%" if predicted_change_5d < 0 else "remain stable"
    st.markdown(f"""
    <div style="margin-top: 1rem; padding: 0.8rem; background-color: #0f172a; border-radius: 8px; border: 1px dashed #334155; font-size:0.875rem;">
        <b>Forecast Insight:</b> Model predicts price will {trend_desc} in the next 5 days ({trend_symbol} expected price: <b>₹{int(forecast_5d):,}</b>).
    </div>
    """, unsafe_allow_html=True)
st.write("---")
# Visual Grid: Simulator, Recommendations, and ML Insights
tab1, tab2, tab3 = st.tabs(["💰 Smart Savings Simulator", "📦 Smart Recommendations", "🔬 ML Algorithm Insights"])
with tab1:
    st.subheader("Smart Savings & Target Alert Simulator")
    st.write("Determine potential savings and simulate target price alerts based on historical volatility.")
    
    sim_col1, sim_col2 = st.columns(2)
    
    with sim_col1:
        target_price = st.slider("Set your Target Price (INR):", min_value=int(min_price * 0.9), max_value=int(max_price * 1.1), value=int(latest_price), step=10)
        
        # Calculate percentage of days where price was below target
        days_below_target = len(df_p[df_p["Price (INR)"] <= target_price])
        pct_days_below = (days_below_target / len(df_p)) * 100
        
        # Check target vs current price
        if latest_price <= target_price:
            st.success(f"🎯 **Target Reached!** Current price (₹{latest_price:,}) is below your target price. Buy now and save!")
        else:
            diff = latest_price - target_price
            st.warning(f"⏳ **Price is above target by ₹{diff:,}.** Historically, the price drops to or below ₹{target_price:,} about **{pct_days_below:.1f}%** of the time.")
            
    with sim_col2:
        # Potential savings card
        pot_sav_max = max_price - latest_price
        pot_sav_avg = avg_price - latest_price
        
        st.markdown(f"""
        <div style="background-color: #1e293b; border-radius: 10px; border: 1px solid #334155; padding: 1.2rem;">
            <h4 style="margin: 0 0 0.8rem 0; color: #38bdf8;">📊 Potential Savings Summary</h4>
            <ul style="margin: 0; padding-left: 1.2rem; color: #e2e8f0; font-size:0.95rem;">
                <li style="margin-bottom:0.4rem;">Savings compared to <b>All-Time Peak (₹{max_price:,})</b>: <b style='color:#10b981;'>₹{pot_sav_max:,}</b></li>
                <li style="margin-bottom:0.4rem;">Savings compared to <b>Average baseline (₹{avg_price:,})</b>: <b style='color:{ "#10b981" if pot_sav_avg > 0 else "#ef4444" };'>{ "₹"+str(pot_sav_avg) if pot_sav_avg >= 0 else "-₹"+str(abs(pot_sav_avg)) }</b></li>
                <li>Daily Price Volatility (Std Dev): <b>₹{int(df_p["Price (INR)"].std()):,}</b></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
with tab2:
    st.subheader("🛍️ Alternative Deals in Category: " + get_category(selected_product))
    st.write("We calculated the Buy Confidence Score for other products in this category. Here are the best deals currently available:")
    
    # Filter products in the same category
    category_df = df[df["Category"] == get_category(selected_product)].copy()
    other_products = [p for p in category_df["Product Name"].unique() if p != selected_product]
    
    if not other_products:
        st.info("No other products in this category currently tracked.")
    else:
        rec_list = []
        for other_p in other_products:
            other_p_df = category_df[category_df["Product Name"] == other_p].sort_values("Date_obj")
            if len(other_p_df) < 30:
                continue
            
            p_latest = int(other_p_df["Price (INR)"].values[-1])
            p_avg = int(other_p_df["Price (INR)"].mean())
            p_min = int(other_p_df["Price (INR)"].min())
            p_max = int(other_p_df["Price (INR)"].max())
            
            # Simple static estimate of confidence score for performance (avoids retraining 30 models on tab load)
            p_range = p_max - p_min
            p_percentile = (p_latest - p_min) / p_range if p_range > 0 else 0.5
            p_confidence = int(np.clip((1.0 - p_percentile) * 100, 10, 95))
            
            rec_list.append({
                "Product": other_p,
                "Current Price": f"₹{p_latest:,}",
                "Avg Price": f"₹{p_avg:,}",
                "Discount vs Avg": f"{(p_avg - p_latest) / p_avg * 100:.1f}%" if p_latest < p_avg else f"-{(p_latest - p_avg) / p_avg * 100:.1f}%",
                "Buy Confidence": p_confidence
            })
            
        if rec_list:
            rec_df = pd.DataFrame(rec_list).sort_values("Buy Confidence", ascending=False)
            
            # Format display styling
            def color_score(val):
                if val >= 80: return 'background-color: #ecfdf5; color: #065f46; font-weight:bold;'
                elif val >= 60: return 'background-color: #f0fdf4; color: #166534;'
                elif val >= 40: return 'background-color: #fffbeb; color: #92400e;'
                else: return 'background-color: #fef2f2; color: #991b1b;'
            
            styled_df = rec_df.style.map(color_score, subset=['Buy Confidence'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            st.info("Insufficient historical data to analyze alternative recommendations.")
with tab3:
    st.subheader("🔬 Random Forest Regressor - Details")
    
    col_ml1, col_ml2 = st.columns(2)
    
    with col_ml1:
        st.markdown("""
        #### How is the Confidence Score computed?
        The **Buy Confidence Score** is generated through a two-part mathematical pipeline:
        1. **Historical Distribution Rank (60% weight):** Determines where the current price sits relative to the recent 30-day min/max and all-time bounds. Values near local minimums score up to 100.
        2. **Random Forest Price Trend (40% weight):** A Random Forest Regressor forecasts the expected price in 5 days. If the forecast indicates a price rise, buying now saves money, raising the score. A forecasted decline lowers the score.
        """)
        
        # Display model accuracy
        st.metric(label="Model Out-Of-Sample MAE (Prediction Accuracy)", value=f"₹{int(avg_mae):,}", help="Mean Absolute Error of the test validation folds")
        st.metric(label="Model R² (Variance Explained)", value=f"{avg_r2:.2f}", help="Indicates the goodness of fit of the predictions (closer to 1.0 is better)")
        
    with col_ml2:
        if importances:
            st.markdown("#### Feature Importances")
            st.write("Which features did the Random Forest model rely on most to make forecasts?")
            
            imp_df = pd.DataFrame({
                "Feature": list(importances.keys()),
                "Importance (%)": [v * 100 for v in importances.values()]
            }).sort_values("Importance (%)", ascending=True)
            
            # Map clean names to features
            feature_labels = {
                'Price_Lag_1': 'Price Yesterday',
                'Price_Lag_2': 'Price 2 Days Ago',
                'Price_Lag_7': 'Price 7 Days Ago',
                'Rolling_Mean_7': '7-Day Avg Price',
                'Rolling_Std_7': '7-Day Volatility',
                'Rolling_Mean_30': '30-Day Avg Price',
                'Rolling_Min_30': '30-Day Lowest Price',
                'Rolling_Max_30': '30-Day Highest Price',
                'Price_Percentile_30': 'Current Price Position (%)',
                'DayOfWeek': 'Day of the Week',
                'DayOfMonth': 'Day of the Month',
                'Month': 'Month of Year'
            }
            imp_df["Feature Description"] = imp_df["Feature"].map(feature_labels)
            
            fig_imp = px.bar(
                imp_df,
                y="Feature Description",
                x="Importance (%)",
                orientation='h',
                color="Importance (%)",
                color_continuous_scale="Viridis",
                height=300
            )
            fig_imp.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_imp, use_container_width=True)
# Expandable Data Explorer
with st.expander("📥 Explore Raw Historical Price Data"):
    csv_download = df_p.drop(columns=["Category"]).to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Product CSV",
        data=csv_download,
        file_name=f"{selected_product.replace(' ', '_')}_price_history.csv",
        mime="text/csv"
    )
    st.dataframe(df_p[["Date", "Price (INR)"]].sort_values("Date", ascending=False), use_container_width=True, hide_index=True)
col3.metric("Calculated Baseline Average", f"₹{p_avg}")
