import streamlit as st
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime
import os
import re
import json
import urllib.request

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
    
    /* Live Sync Tracker Styling */
    .live-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    .live-badge {
        background-color: #10b981;
        color: #ffffff;
        font-weight: 700;
        font-size: 0.7rem;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
    }
    .pulsing-dot {
        width: 8px;
        height: 8px;
        background-color: #ffffff;
        border-radius: 50%;
        box-shadow: 0 0 0 0 rgba(255, 255, 255, 0.7);
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(255, 255, 255, 0.7);
        }
        70% {
            transform: scale(1);
            box-shadow: 0 0 0 6px rgba(255, 255, 255, 0);
        }
        100% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(255, 255, 255, 0);
        }
    }
</style>
""", unsafe_allow_html=True)

# Load data helper function
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
    df = df.sort_values("Date_obj")
    return df

def clean_slug(name):
    # Mapping of custom spelling fixes in dataset to fit real URLs
    corrections = {
        "diascovery": "discovery",
        "boAt": "boat",
    }
    name_lower = name.lower()
    for k, v in corrections.items():
        name_lower = name_lower.replace(k, v)
    
    # Remove special characters except alphanumeric, spaces, and dashes
    name_lower = name_lower.replace("+", "")
    slug = re.sub(r'[^a-z0-9\s\-]', '', name_lower)
    slug = re.sub(r'[\s\-]+', '-', slug)
    return slug.strip('-')

def fetch_from_boat_url(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8', errors='ignore')
            
        # 1. Look for type application/ld+json offers
        json_ld_blocks = re.findall(r'<script type="application/ld\+json">(.*?)</script>', html, re.DOTALL)
        for block in json_ld_blocks:
            try:
                data = json.loads(block.strip())
                graph = data.get('@graph', [data]) if isinstance(data, dict) else data
                for item in graph:
                    if isinstance(item, dict) and item.get('@type') == 'Product':
                        offers = item.get('offers', [])
                        if isinstance(offers, dict):
                            offers = [offers]
                        
                        prices = []
                        for offer in offers:
                            if isinstance(offer, dict) and 'price' in offer:
                                try:
                                    p = float(offer['price'])
                                    prices.append(p)
                                except:
                                    pass
                        if prices:
                            return min(prices)
            except:
                pass
                
        # 2. Look for og metadata tags
        meta_price = re.search(r'<meta property="product:price:amount" content="([^"]+)"', html)
        if meta_price:
            try:
                return float(meta_price.group(1).replace(",", ""))
            except:
                pass
    except:
        pass
    return None

def search_price_via_ddg(query):
    url = "https://lite.duckduckgo.com/lite/"
    data = urllib.parse.urlencode({'q': query}).encode('utf-8')
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    try:
        req = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8', errors='ignore')
            
        patterns = [
            r'Rs\.\s*(\d+(?:,\d+)*(?:\.\d+)?)',
            r'INR\s*(\d+(?:,\d+)*(?:\.\d+)?)',
            r'\u20b9\s*(\d+(?:,\d+)*(?:\.\d+)?)'
        ]
        
        found_prices = []
        for pat in patterns:
            matches = re.findall(pat, html)
            for m in matches:
                try:
                    price_val = float(m.replace(",", ""))
                    if 100 < price_val < 50000:
                        found_prices.append(price_val)
                except:
                    pass
        if found_prices:
            from collections import Counter
            counts = Counter(found_prices)
            most_common = counts.most_common(1)[0][0]
            return most_common
    except:
        pass
    return None

def fetch_live_price(product_name, links_df=None):
    boat_url = None
    fk_url = None
    amz_url = None
    
    if links_df is not None:
        # Match case-insensitive
        match_rows = links_df[links_df["Product Name"].str.lower() == product_name.lower()]
        if match_rows.empty:
            # Fuzzy match
            match_rows = links_df[links_df["Product Name"].apply(lambda x: str(x).lower() in product_name.lower() or product_name.lower() in str(x).lower())]
            
        if not match_rows.empty:
            row = match_rows.iloc[0]
            boat_url = row.get("boAt Lifestyle Link")
            fk_url = row.get("Flipkart Link")
            amz_url = row.get("Amazon Link")
            
    # Normalize URLs
    def clean_url(u):
        if pd.isna(u) or not isinstance(u, str) or not u.strip():
            return None
        return u.strip()
        
    boat_url = clean_url(boat_url)
    fk_url = clean_url(fk_url)
    amz_url = clean_url(amz_url)
    
    # 1. Try boAt Lifestyle direct link
    if boat_url:
        price = fetch_from_boat_url(boat_url)
        if price:
            return price, boat_url, "boAt Store Link"
            
    # 2. Try Flipkart search fallback
    if fk_url:
        price = search_price_via_ddg(f"{product_name} price Flipkart")
        if price:
            return price, fk_url, "Flipkart Link (Search)"
            
    # 3. Try Amazon search fallback
    if amz_url:
        price = search_price_via_ddg(f"{product_name} price Amazon")
        if price:
            return price, amz_url, "Amazon Link (Search)"
            
    # 4. Try standard slug construction on boAt Lifestyle
    slug = clean_slug(product_name)
    default_url = f"https://www.boat-lifestyle.com/products/{slug}"
    price = fetch_from_boat_url(default_url)
    if price:
        return price, default_url, "boAt Store (Default Slug)"
        
    # 5. Final fallback search
    price = search_price_via_ddg(f"{product_name} price in India")
    if price:
        return price, boat_url or fk_url or amz_url or default_url, "Web Search Index"
        
    return None, boat_url or fk_url or amz_url or default_url, "Unknown"

def save_live_to_csv(product_name, price, date_str):
    relative_path = "boat_products_price_history_india_format.csv"
    absolute_path = r"C:\Users\sharo\Downloads\boat_products_price_history_india_format.csv"
    
    target_paths = []
    if os.path.exists(relative_path):
        target_paths.append(relative_path)
    if os.path.exists(absolute_path):
        target_paths.append(absolute_path)
        
    if not target_paths:
        target_paths = [absolute_path]
        
    for path in target_paths:
        if os.path.exists(path):
            try:
                df_temp = pd.read_csv(path)
            except:
                continue
        else:
            df_temp = pd.DataFrame(columns=["Date", "Product Name", "Price (INR)"])
            
        match_mask = (df_temp["Product Name"] == product_name) & (df_temp["Date"] == date_str)
        if match_mask.any():
            df_temp.loc[match_mask, "Price (INR)"] = price
        else:
            new_row = pd.DataFrame([{"Date": date_str, "Product Name": product_name, "Price (INR)": price}])
            df_temp = pd.concat([df_temp, new_row], ignore_index=True)
            
        df_temp.to_csv(path, index=False)

try:
    df = load_data()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.info("Please make sure the CSV exists in the same folder or at: C:\\Users\\sharo\\Downloads\\boat_products_price_history_india_format.csv")
    st.stop()

# Load product links Excel file if available
def load_product_links():
    relative_path = "boAt Products Links.xlsx"
    absolute_path = r"C:\Users\sharo\Downloads\boAt Products Links.xlsx"
    
    if os.path.exists(relative_path):
        try:
            return pd.read_excel(relative_path)
        except:
            pass
    if os.path.exists(absolute_path):
        try:
            return pd.read_excel(absolute_path)
        except:
            pass
    return None

links_df = load_product_links()

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

# Sidebar UI
st.sidebar.markdown("<h2 style='text-align: center; color: #38bdf8;'>⚓ boAt Engine</h2>", unsafe_allow_html=True)
st.sidebar.write("Analyze price variations and predict buying opportunities using Machine Learning.")

# Sidebar Filters
category_list = ["All Categories"] + sorted(df["Category"].unique().tolist())
selected_category = st.sidebar.selectbox("Filter Category:", category_list)

if selected_category != "All Categories":
    filtered_df = df[df["Category"] == selected_category]
else:
    filtered_df = df

product_list = sorted(filtered_df["Product Name"].unique().tolist())
selected_product = st.sidebar.selectbox("Select Product Line:", product_list)

# Sidebar Settings for Live Tracker
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Live Tracker Settings")
sim_mode = st.sidebar.checkbox("Enable Live Simulator Mode", value=False, help="Simulate price fluctuations instead of querying the website")
manual_input_allowed = st.sidebar.checkbox("Allow Manual Price Correction", value=False, help="Add manual price input field if scraping fails or matches are incorrect")

# Filter product dataset
df_p = df[df["Product Name"] == selected_product].sort_values("Date_obj")

# Main Page Header
st.markdown(f"<h1 style='margin-bottom:0;'>🏷️ {selected_product}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color: #64748b; font-size:1.1rem; margin-top:0;'>Category: <b>{get_category(selected_product)}</b> | Tracking <b>{len(df_p)} days</b> of historical prices</p>", unsafe_allow_html=True)

# Self-Attentive Jump-Diffusion (SAJD) Simulation Model
@st.cache_data(ttl=600)
def run_sajd_forecast(df_product, N_paths=10000, T_days=30, beta=0.05):
    prod_data = df_product.sort_values('Date_obj').copy().reset_index(drop=True)
    last_price = float(prod_data['Price (INR)'].iloc[-1])
    last_date = prod_data['Date_obj'].iloc[-1]
    
    # Check for sufficient data
    if len(prod_data) < 5:
        # Return fallback flat-line arrays
        future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, T_days + 1)]
        expected = [last_price] * T_days
        low = [last_price] * T_days
        high = [last_price] * T_days
        calibration = {
            'alpha': 0.0, 'sigma': 0.001, 'p_jump': 0.0,
            'jump_mean': 0.0, 'jump_std': 0.05, 'beta': beta,
            'N_paths': N_paths, 'T_days': T_days
        }
        return future_dates, expected, low, high, calibration

    # Calculate daily log returns
    returns = np.log(prod_data['Price (INR)'] / prod_data['Price (INR)'].shift(1)).dropna()
    
    # --- DYNAMIC CALIBRATION ---
    mean_all = returns.mean()
    std_all = returns.std()
    
    # Handle flat-line products (zero volatility) to prevent crashing
    if std_all == 0 or pd.isna(std_all):
        alpha, sigma, p_jump = 0.0, 0.001, 0.0
        jump_mean, jump_std = 0.0, 0.05
    else:
        # Separate normal market noise from sudden price shocks (> 2 std devs)
        jumps = returns[abs(returns - mean_all) > 2 * std_all]
        normal_returns = returns[abs(returns - mean_all) <= 2 * std_all]
        
        alpha = normal_returns.mean() if len(normal_returns) > 0 else 0.0
        sigma = normal_returns.std() if len(normal_returns) > 0 else 0.001
        
        p_jump = len(jumps) / len(returns)
        if len(jumps) > 0:
            jump_mean = jumps.mean()
            jump_std = jumps.std() if (len(jumps) > 1 and jumps.std() > 0) else 0.05
        else:
            jump_mean, jump_std = -0.15, 0.05 # Default 15% drop if no history of jumps
            
    # Set seed for reproducible results
    np.random.seed(42)
    
    # Generate random matrices for this specific product's volatility profile
    Z = np.random.normal(0, 1, (N_paths, T_days))
    J = np.random.binomial(1, p_jump, (N_paths, T_days))
    Jump_sizes = np.random.normal(jump_mean, jump_std, (N_paths, T_days))
    
    simulated_paths = np.zeros((N_paths, T_days + 1))
    simulated_paths[:, 0] = last_price
    
    for t in range(1, T_days + 1):
        P_prev = simulated_paths[:, t-1]
        
        # Calculate the 4 forces
        memory_effect = beta * (last_price - P_prev)
        drift = alpha * P_prev + memory_effect
        vol = sigma * P_prev * Z[:, t-1]
        
        jump_mult = np.exp(Jump_sizes[:, t-1])
        jump_comp = P_prev * J[:, t-1] * (jump_mult - 1)
        
        # Advance the timeline
        P_next = P_prev + drift + vol + jump_comp
        simulated_paths[:, t] = np.maximum(P_next, last_price * 0.2) # Hard floor at 80% price drop
        
    # --- DATA EXTRACTION ---
    # Extract the 5th (Pessimistic), 50th (Expected), and 95th (Optimistic) percentiles
    percentiles = np.percentile(simulated_paths, [5, 50, 95], axis=0)
    
    future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, T_days + 1)]
    expected_path = [round(x, 2) for x in percentiles[1][1:]]
    low_path = [round(x, 2) for x in percentiles[0][1:]]
    high_path = [round(x, 2) for x in percentiles[2][1:]]
    
    calibration = {
        'alpha': alpha,
        'sigma': sigma,
        'p_jump': p_jump,
        'jump_mean': jump_mean,
        'jump_std': jump_std,
        'beta': beta,
        'N_paths': N_paths,
        'T_days': T_days
    }
    
    return future_dates, expected_path, low_path, high_path, calibration

with st.spinner("🔮 Running SAJD Monte Carlo Diffusion models (10,000 paths)..."):
    forecast_dates, expected_forecast, pessimistic_forecast, optimistic_forecast, sajd_calibration = run_sajd_forecast(df_p)

# Basic Stats Calculations
latest_price = int(df_p["Price (INR)"].values[-1])
latest_date = df_p["Date_obj"].values[-1]
avg_price = int(df_p["Price (INR)"].mean())
min_price = int(df_p["Price (INR)"].min())
max_price = int(df_p["Price (INR)"].max())

# Dates for min/max prices
min_price_date = df_p[df_p["Price (INR)"] == min_price]["Date"].values[0]
max_price_date = df_p[df_p["Price (INR)"] == max_price]["Date"].values[0]

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

overall_range = max_price - min_price
overall_percentile = (latest_price - min_price) / overall_range if overall_range > 0 else 0.5

percentile_score = (1.0 - recent_percentile) * 0.7 + (1.0 - overall_percentile) * 0.3

# 2. ML Forecast Trend Score
forecast_5d = expected_forecast[4] # 5-day prediction
predicted_change_5d = (forecast_5d - latest_price) / latest_price

# If price is predicted to drop, buying now is sub-optimal unless it's already near min
# If price is predicted to rise, buying now is highly recommended
if predicted_change_5d >= 0.05:
    trend_score = 1.0
elif predicted_change_5d <= -0.05:
    trend_score = 0.0
else:
    # Scale from 0.0 to 1.0 between -5% and +5%
    trend_score = (predicted_change_5d + 0.05) / 0.10

# Combine scores (60% based on historical placement, 40% based on ML forecast trend)
confidence_score = int(np.clip((0.6 * percentile_score + 0.4 * trend_score) * 100, 5, 98))

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

# -----------------
# REAL-TIME PRICE TRACKER PANEL
# -----------------
# Session state initialization
if "live_price" not in st.session_state:
    st.session_state.live_price = None
if "live_url" not in st.session_state:
    st.session_state.live_url = None
if "live_source" not in st.session_state:
    st.session_state.live_source = None
if "last_scanned_product" not in st.session_state:
    st.session_state.last_scanned_product = None
if "scan_error" not in st.session_state:
    st.session_state.scan_error = None

# Clear state on product change
if st.session_state.last_scanned_product != selected_product:
    st.session_state.live_price = None
    st.session_state.live_url = None
    st.session_state.live_source = None
    st.session_state.scan_error = None
    st.session_state.last_scanned_product = selected_product

# Render Excel details or alert status in sidebar
if links_df is not None:
    # See if the product has links mapped in Excel
    match_rows = links_df[links_df["Product Name"].str.lower() == selected_product.lower()]
    if match_rows.empty:
        match_rows = links_df[links_df["Product Name"].apply(lambda x: str(x).lower() in selected_product.lower() or selected_product.lower() in str(x).lower())]
    
    if not match_rows.empty:
        row = match_rows.iloc[0]
        mapped_sources = []
        if not pd.isna(row.get("boAt Lifestyle Link")) and str(row.get("boAt Lifestyle Link")).strip():
            mapped_sources.append("boAt Store")
        if not pd.isna(row.get("Flipkart Link")) and str(row.get("Flipkart Link")).strip():
            mapped_sources.append("Flipkart")
        if not pd.isna(row.get("Amazon Link")) and str(row.get("Amazon Link")).strip():
            mapped_sources.append("Amazon")
            
        if mapped_sources:
            st.sidebar.success(f"🔗 Mapped: {', '.join(mapped_sources)}")
        else:
            st.sidebar.info("🔗 Product has no active links in Excel sheet.")
    else:
        st.sidebar.info("ℹ️ Product not matched in Excel links sheet.")
else:
    st.sidebar.warning("⚠️ Excel mapping sheet not loaded.")

st.markdown("### ⏱️ Real-time Price Tracker")

col_live_status, col_live_action = st.columns([7, 3])

with col_live_status:
    if st.session_state.live_price is not None:
        price_diff = st.session_state.live_price - latest_price
        diff_pct = (price_diff / latest_price) * 100 if latest_price > 0 else 0.0
        
        # Color based on price direction
        if price_diff < 0:
            diff_color = "#10b981" # green
            diff_desc = f"Price is down by ₹{abs(price_diff):,} ({abs(diff_pct):.1f}%) compared to yesterday's recorded price! 🔥 Excellent deal."
        elif price_diff > 0:
            diff_color = "#ef4444" # red
            diff_desc = f"Price is up by ₹{price_diff:,} ({diff_pct:.1f}%) compared to yesterday's recorded price. ⚠️ You might want to hold off."
        else:
            diff_color = "#94a3b8" # grey
            diff_desc = "Price is identical to yesterday's recorded price. No change detected."
            
        source_label = st.session_state.live_source if st.session_state.live_source else "boAt Store"
        st.markdown(f"""
        <div class="live-container">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span class="live-badge">
                    <span class="pulsing-dot"></span> Live Scan Active
                </span>
                <span style="font-size: 0.85rem; color: #94a3b8;"><span style="color: #38bdf8; font-weight: bold;">{source_label}</span> | Link: <a href="{st.session_state.live_url}" target="_blank" style="color: #38bdf8; text-decoration: none;">Product Page</a></span>
            </div>
            <div style="display: flex; align-items: baseline; gap: 1rem;">
                <span style="font-size: 2.25rem; font-weight: 800; color: #f8fafc;">₹{st.session_state.live_price:,}</span>
                <span style="font-size: 1rem; color: {diff_color}; font-weight: bold;">{"+" if price_diff > 0 else ""}₹{price_diff:,} ({"+" if price_diff > 0 else ""}{diff_pct:.1f}%)</span>
            </div>
            <p style="color: #e2e8f0; font-size: 0.95rem; margin-top: 0.5rem; margin-bottom: 0;">{diff_desc}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No active scan. Click 'Scan Live Price' to query the current live store price.")

with col_live_action:
    # Scan trigger button
    if st.button("🔍 Scan Live Price", use_container_width=True):
        if sim_mode:
            # Simulate a fluctuation +/- 3% around the latest price
            fluctuation = np.random.uniform(-0.03, 0.03)
            simulated_price = round(latest_price * (1 + fluctuation), -1) # round to nearest 10
            st.session_state.live_price = int(simulated_price)
            st.session_state.live_url = f"https://www.boat-lifestyle.com/products/{clean_slug(selected_product)}"
            st.session_state.live_source = "Simulation Ticker"
            st.session_state.scan_error = None
            st.rerun()
        else:
            with st.spinner("Scanning real-time price using Excel references..."):
                price, url, src = fetch_live_price(selected_product, links_df=links_df)
                if price is not None:
                    st.session_state.live_price = int(price)
                    st.session_state.live_url = url
                    st.session_state.live_source = src
                    st.session_state.scan_error = None
                else:
                    st.session_state.scan_error = "Could not automatically retrieve price from Excel target links."
                    st.session_state.live_url = url
                    st.session_state.live_source = src
                st.rerun()
                
    if st.session_state.scan_error:
        st.error(st.session_state.scan_error)
        
    # Manual entry override if error exists or manually allowed
    if (st.session_state.scan_error or manual_input_allowed) and st.session_state.live_price is None:
        manual_price = st.number_input("Enter current price manually (INR):", min_value=100, max_value=100000, value=int(latest_price), step=50)
        if st.button("Set Manual Price", use_container_width=True):
            st.session_state.live_price = int(manual_price)
            st.session_state.live_source = "Manual Override"
            st.session_state.scan_error = None
            st.rerun()

    # Save to database button
    if st.session_state.live_price is not None:
        today_str = datetime.date.today().strftime("%d-%m-%Y")
        already_has_today = not df_p[df_p["Date"] == today_str].empty
        
        if already_has_today:
            st.info("Today's price is already in the database. Updating this entry will replace it.")
            btn_label = "🔄 Update Today's Price"
        else:
            btn_label = "💾 Save Price to Database"
            
        if st.button(btn_label, use_container_width=True):
            save_live_to_csv(selected_product, st.session_state.live_price, today_str)
            st.success("Successfully saved to database!")
            st.cache_data.clear() # clear all caches so dataset reloads and models retrain
            st.rerun()

st.markdown("---")

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
    st.subheader("📈 Price History & ML 30-Day Forecast")
    
    last_date = df_p["Date_obj"].max()
    
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
    
    # 2. Predicted Prices (SAJD 30-Day Forecast)
    full_forecast_dates = [last_date] + list(forecast_dates)
    full_expected_prices = [latest_price] + list(expected_forecast)
    full_pessimistic_prices = [latest_price] + list(pessimistic_forecast)
    full_optimistic_prices = [latest_price] + list(optimistic_forecast)
    
    # Optimistic Forecast Trace (upper bound)
    fig.add_trace(go.Scatter(
        x=full_forecast_dates,
        y=full_optimistic_prices,
        mode='lines',
        name='Optimistic Bound (95%)',
        line=dict(color='rgba(56, 189, 248, 0.3)', width=1, dash='dash'),
        hovertemplate='Optimistic: ₹%{y:,.0f}<extra></extra>'
    ))
    
    # Pessimistic Forecast Trace (lower bound) with fill to upper bound
    fig.add_trace(go.Scatter(
        x=full_forecast_dates,
        y=full_pessimistic_prices,
        mode='lines',
        name='Pessimistic Bound (5%)',
        line=dict(color='rgba(239, 68, 68, 0.3)', width=1, dash='dash'),
        fill='tonexty',
        fillcolor='rgba(56, 189, 248, 0.12)',
        hovertemplate='Pessimistic: ₹%{y:,.0f}<extra></extra>'
    ))
    
    # Expected Forecast Trace (50th percentile)
    fig.add_trace(go.Scatter(
        x=full_forecast_dates,
        y=full_expected_prices,
        mode='lines+markers',
        name='Expected Forecast (50%)',
        line=dict(color='#38bdf8', width=2),
        marker=dict(size=4, symbol='circle'),
        hovertemplate='Expected: ₹%{y:,.0f}<extra></extra>'
    ))
    
    # 2.5 Scanned Live Price (Unsaved)
    today_str = datetime.date.today().strftime("%d-%m-%Y")
    has_today_in_db = not df_p[df_p["Date"] == today_str].empty
    
    if st.session_state.live_price is not None and not has_today_in_db:
        connector_dates = [last_date, pd.to_datetime(today_str, format="%d-%m-%Y")]
        connector_prices = [latest_price, st.session_state.live_price]
        
        fig.add_trace(go.Scatter(
            x=connector_dates,
            y=connector_prices,
            mode='lines+markers',
            name='Scanned Live Price (Unsaved)',
            line=dict(color='#10b981', width=2, dash='dot'),
            marker=dict(size=8, symbol='star', color='#10b981'),
            hovertemplate='Live Date: %{x|%d %b %Y}<br>Scanned Price: ₹%{y:,}<extra></extra>'
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
    st.subheader("🔬 Self-Attentive Jump-Diffusion (SAJD) Model")
    
    col_ml1, col_ml2 = st.columns(2)
    
    with col_ml1:
        st.markdown("""
        #### Mathematical Model Formulation
        The future price path of the product is modeled as a discrete-time **Self-Attentive Jump-Diffusion** process. 
        Each daily price step is driven by four key forces:
        """)
        
        st.latex(r"P_t = P_{t-1} + \left(\alpha P_{t-1} + \beta (P_{\text{baseline}} - P_{t-1})\right)\Delta t + \sigma P_{t-1} Z_t \sqrt{\Delta t} + P_{t-1} J_t \left(e^{Y_t} - 1\right)")
        
        st.markdown(r"""
        **Where:**
        *   **$\Delta t = 1$ Day**: The time step of the simulation.
        *   **Drift ($\alpha$):** Represents the baseline daily price direction under normal market conditions.
        *   **Attentional Memory ($\beta = 0.05$):** A mean-reversion force that pulls the price back toward the baseline (the latest actual price $P_{\text{baseline}}$) after a jump shock.
        *   **Market Volatility ($\sigma$):** Normal day-to-day price noise. $Z_t \sim \mathcal{N}(0, 1)$ represents standard white noise.
        *   **Price Shocks / Jumps ($J_t$):** Sudden price changes. $J_t \sim \text{Bernoulli}(p_{\text{jump}})$ controls shock arrivals, and $Y_t \sim \mathcal{N}(\mu_{\text{jump}}, \sigma_{\text{jump}})$ represents the magnitude of the log-return shock.
        """)
        
    with col_ml2:
        st.markdown("#### Calibrated Model Parameters")
        st.write(f"The parameters below are dynamically calibrated based on the historical daily log returns of **{selected_product}**:")
        
        # Display parameters in a nice 2x3 grid of metrics
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.metric(
                label="Baseline Drift (α)", 
                value=f"{sajd_calibration['alpha']:.5f}",
                help="The average daily return of the product excluding jump days."
            )
            st.metric(
                label="Daily Volatility (σ)", 
                value=f"{sajd_calibration['sigma']:.4%}",
                help="Standard deviation of daily price returns under normal market conditions."
            )
            st.metric(
                label="Jump Shock Probability (p)", 
                value=f"{sajd_calibration['p_jump']:.2%}",
                help="The historical frequency of daily price movements exceeding 2 standard deviations."
            )
        with p_col2:
            st.metric(
                label="Mean Jump Magnitude (μ_jump)", 
                value=f"{sajd_calibration['jump_mean']:.2%}",
                help="The average log-return impact when a price shock occurs."
            )
            st.metric(
                label="Jump Volatility (σ_jump)", 
                value=f"{sajd_calibration['jump_std']:.2%}",
                help="The variability of shock magnitudes."
            )
            st.metric(
                label="Memory Pull Weight (β)", 
                value=f"{sajd_calibration['beta']:.3f}",
                help="The strength of mean-reversion pulling price paths back to the baseline."
            )
            
        st.markdown(f"""
        <div style="background-color: #1e293b; border-radius: 8px; border: 1px solid #334155; padding: 1rem; margin-top: 1rem; font-size: 0.85rem;">
            <b>Monte Carlo Settings:</b> Simulated <b>{sajd_calibration['N_paths']:,} paths</b> over a <b>{sajd_calibration['T_days']}-day horizon</b>.
            The shaded confidence band in the main chart highlights the 5th and 95th percentiles of these simulated paths.
        </div>
        """, unsafe_allow_html=True)

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
