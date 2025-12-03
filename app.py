"""
Streamlit App: Bakaro Inventory & Demand Predictor
Filename: streamlit_bakaro_inventory_app.py

Purpose:
A full Streamlit app that helps small vendors in Bakaro Market / Mogadishu forecast short-term sales and get inventory reorder suggestions.

What's included in this single-file app:
- Problem definition and app description
- Example synthetic dataset generator (so you can try the app without real data)
- CSV / Excel upload
- Preprocessing and feature engineering (lags, rolling means)
- Visualization (sales over time, distribution)
- A RandomForest regression model to predict next-period demand
- Reorder suggestion logic (recommended reorder quantity, alerts)
- Downloadable prediction results

How to run:
1. Create a virtualenv (optional): python -m venv venv && source venv/bin/activate
2. Install dependencies:
   pip install streamlit pandas numpy scikit-learn matplotlib plotly
3. Run:
   streamlit run streamlit_bakaro_inventory_app.py

Notes:
- This app uses a straightforward RandomForest approach for short-term demand predictions. Replace or extend with time-series models (Prophet, ARIMA) as needed.
- The app is intentionally simple and well-commented to help you adapt it to real Bakaro market data.

"""

import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import timedelta

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Bakaro Inventory & Demand Predictor", layout="wide")

# ------------------------- Helper functions -------------------------
@st.cache_data
def generate_synthetic_data(days=180, n_items=8, seed=42):
    """Generate example sales history for small vendors in Bakaro market.
    Columns: date, item_id, item_name, store, price, quantity_sold
    """
    rng = np.random.RandomState(seed)
    start = pd.to_datetime("2025-01-01")
    dates = pd.date_range(start, periods=days, freq='D')

    item_names = [
        "Mobile Charger", "SIM Card", "Phone Screen", "Earphones",
        "Flashlight", "Power Bank", "Memory Card", "Phone Case"
    ][:n_items]

    rows = []
    for item in item_names:
        base = rng.randint(5, 40)  # base daily demand
        seasonality = rng.uniform(0.8, 1.3, size=len(dates))
        noise = rng.normal(scale=base * 0.25, size=len(dates))
        for i, d in enumerate(dates):
            qty = max(0, int(base * seasonality[i] + noise[i]))
            price = round(float(rng.uniform(3.0, 40.0) * (1 + rng.uniform(-0.1, 0.1))), 2)
            rows.append({
                'date': d.date(),
                'item_name': item,
                'store': 'Bakaro Main St',
                'price': price,
                'quantity_sold': qty
            })
    df = pd.DataFrame(rows)
    return df


def preprocess(df):
    """Group sales to daily item-level, create lag features and rolling averages."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Aggregate to daily item-level
    daily = df.groupby(['date', 'item_name']).agg({
        'quantity_sold': 'sum',
        'price': 'mean'
    }).reset_index().sort_values(['item_name', 'date'])

    # Create features per item
    feature_frames = []
    for item, g in daily.groupby('item_name'):
        g = g.set_index('date').asfreq('D', fill_value=0).reset_index()
        g['item_name'] = item
        g['qty_lag_1'] = g['quantity_sold'].shift(1).fillna(0)
        g['qty_lag_7'] = g['quantity_sold'].shift(7).fillna(0)
        g['rolling_7'] = g['quantity_sold'].rolling(7, min_periods=1).mean().fillna(0)
        g['day_of_week'] = g['date'].dt.dayofweek
        g['month'] = g['date'].dt.month
        feature_frames.append(g)
    full = pd.concat(feature_frames, ignore_index=True)
    return full


def train_model(train_X, train_y, n_estimators=200, random_state=42):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(train_X, train_y)
    return model


def prepare_prediction_frame(last_rows, horizon=7):
    """Given the last rows per item, create frame to predict next `horizon` days.
    We'll roll-forward lag features in a simple way (recursive forecasting).
    This is a pragmatic approach for short horizons.
    """
    preds = []
    for item, g in last_rows.groupby('item_name'):
        row = g.sort_values('date').iloc[-1:].copy()
        cur_row = row.copy()
        for day in range(1, horizon+1):
            target_date = cur_row['date'].iloc[0] + pd.Timedelta(days=day)
            new = {}
            new['date'] = target_date
            new['item_name'] = item
            new['day_of_week'] = target_date.dayofweek
            new['month'] = target_date.month
            # For recursive features, use previous predictions (approx)
            # We'll set qty_lag_1 as last observed rolling_7 and qty_lag_7 as last observed qty_lag_7
            new['qty_lag_1'] = cur_row['rolling_7'].iloc[0]
            new['qty_lag_7'] = cur_row['qty_lag_7'].iloc[0]
            new['rolling_7'] = cur_row['rolling_7'].iloc[0]
            new['price'] = cur_row['price'].iloc[0]
            preds.append(new)
            # update cur_row rolling_7 roughly by keeping same (simple)
            # For more accuracy, use model predictions to update rolling window
    return pd.DataFrame(preds)


# ------------------------- App layout -------------------------
st.title("Bakaro Inventory & Demand Predictor")
st.markdown(
    """
    **Purpose:** Help small vendors in Bakaro (Mogadishu) forecast short-term demand and get reorder recommendations.

    **How it works:** Upload your sales history (daily transactions). The app trains a simple RandomForest model to predict short-term demand per item, shows visualizations, and suggests reorder quantities using basic safety-stock formulas.
    """
)

# Sidebar controls
st.sidebar.header("Options")
mode = st.sidebar.radio("Use sample data or upload your own?", ['Sample data', 'Upload CSV/Excel'])
horizon = st.sidebar.number_input('Prediction horizon (days)', min_value=1, max_value=30, value=7)
reorder_lead_time = st.sidebar.number_input('Lead time (days) — how long it takes to restock', min_value=1, max_value=30, value=7)
safety_z = st.sidebar.slider('Safety-stock Z-score (higher = more safety stock)', min_value=0.0, max_value=3.0, value=1.65)
train_test_split_pct = st.sidebar.slider('Train/Test split (fraction for training)', 0.5, 0.95, 0.8)

# Data input
if mode == 'Sample data':
    st.info('Using synthetic example data (useful for testing).')
    df = generate_synthetic_data(days=240, n_items=8)
    st.markdown('**Sample data preview**')
    st.dataframe(df.head(10))
else:
    uploaded = st.file_uploader('Upload sales CSV or Excel (columns: date, item_name, price, quantity_sold, store optional)', type=['csv', 'xlsx'])
    if uploaded is not None:
        try:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.success('File loaded successfully')
            st.dataframe(df.head(10))
        except Exception as e:
            st.error('Could not read uploaded file: ' + str(e))
            st.stop()
    else:
        st.warning('Upload a file to continue, or switch to Sample data.')
        st.stop()

# Minimal validation
required_cols = {'date', 'item_name', 'quantity_sold'}
if not required_cols.issubset(set(df.columns)):
    st.error(f'Data must contain these columns: {required_cols}')
    st.stop()

# Preprocess
with st.spinner('Preprocessing data...'):
    processed = preprocess(df)

st.markdown('## Historical Sales Overview')
col1, col2 = st.columns([2,1])
with col1:
    sample_item = st.selectbox('Choose item to inspect', processed['item_name'].unique())
    item_df = processed[processed['item_name']==sample_item].sort_values('date')
    fig = px.line(item_df, x='date', y='quantity_sold', title=f'Daily sales — {sample_item}')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown('**Summary stats**')
    stats = item_df['quantity_sold'].describe().to_frame().T
    st.dataframe(stats)

# Show distribution across items
st.markdown('### Recent average daily sales by item')
recent = processed.groupby('item_name').tail(30).groupby('item_name')['quantity_sold'].mean().reset_index().sort_values('quantity_sold', ascending=False)
fig2 = px.bar(recent, x='item_name', y='quantity_sold', title='Average daily sales (recent 30 days)')
st.plotly_chart(fig2, use_container_width=True)

# Modeling
st.markdown('## Train model and predict')
model_train_button = st.button('Train model & predict')

if model_train_button:
    st.info('Training model — this may take a few seconds depending on data size.')
    # Simple model: predict quantity_sold for each date+item using lag features
    features = ['qty_lag_1', 'qty_lag_7', 'rolling_7', 'day_of_week', 'month', 'price']
    df_model = processed.dropna(subset=features + ['quantity_sold']).copy()
    X = df_model[features]
    y = df_model['quantity_sold']

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_test_split_pct, random_state=42)

    model = train_model(X_train, y_train)

    # evaluate
    preds_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds_test)
    rmse = mean_squared_error(y_test, preds_test, squared=False)
    st.success(f'Trained RandomForest — MAE: {mae:.2f}, RMSE: {rmse:.2f}')

    # Prepare prediction frame (simple recursive) using last rows per item
    last_rows = processed.groupby('item_name').tail(1)
    pred_frame = prepare_prediction_frame(last_rows, horizon=horizon)

    if pred_frame.empty:
        st.error('Not enough data for prediction.')
    else:
        X_pred = pred_frame[['qty_lag_1','qty_lag_7','rolling_7','day_of_week','month','price']]
        pred_frame['pred_qty'] = model.predict(X_pred).round().astype(int).clip(lower=0)

        st.markdown('### Predicted daily demand (next %d days)' % horizon)
        st.dataframe(pred_frame[['date','item_name','pred_qty']].sort_values(['item_name','date']).head(50))

        # Aggregate predictions to get total horizon demand per item
        agg = pred_frame.groupby('item_name')['pred_qty'].sum().reset_index().rename(columns={'pred_qty':'pred_total_next_%d_days' % horizon})

        # Compute simple safety stock and reorder quantity
        # Estimate demand std from recent 30 days per item
        recent_stats = processed.groupby('item_name').tail(30).groupby('item_name')['quantity_sold'].agg(['mean','std']).reset_index()
        agg = agg.merge(recent_stats, on='item_name', how='left')
        agg['lead_time'] = reorder_lead_time
        agg['safety_stock'] = (agg['std'].fillna(0) * np.sqrt(agg['lead_time']) * safety_z).round().astype(int)
        agg['reorder_point'] = (agg['mean'].fillna(0) * agg['lead_time']).round().astype(int) + agg['safety_stock']
        # recommended order = predicted demand over horizon + safety_stock - current_on_hand (unknown). We'll assume on_hand = 0 (user can edit later)
        agg['recommended_order_qty'] = (agg['pred_total_next_%d_days' % horizon] + agg['safety_stock']).astype(int)

        st.markdown('### Reorder suggestions')
        st.dataframe(agg[['item_name','pred_total_next_%d_days' % horizon,'mean','std','safety_stock','reorder_point','recommended_order_qty']])

        # Allow user to edit current on-hand and compute final qty
        st.markdown('---')
        st.markdown('### Adjust for current on-hand stock (optional)')
        current_on_hand = {}
        with st.form('onhand_form'):
            for it in agg['item_name']:
                current_on_hand[it] = st.number_input(f'Current on-hand — {it}', min_value=0, value=0, step=1, key=f'onhand_{it}')
            submitted = st.form_submit_button('Compute final orders')
        if submitted:
            agg['on_hand'] = agg['item_name'].map(current_on_hand).astype(int)
            agg['final_order_qty'] = (agg['recommended_order_qty'] - agg['on_hand']).clip(lower=0).astype(int)
            st.dataframe(agg[['item_name','pred_total_next_%d_days' % horizon,'on_hand','recommended_order_qty','final_order_qty']])

            # Download button
            csv = agg.to_csv(index=False).encode('utf-8')
            st.download_button('Download reorder plan CSV', data=csv, file_name='bakaro_reorder_plan.csv', mime='text/csv')

        # Plots: heatmap of predicted totals
        fig3 = px.bar(agg, x='item_name', y='pred_total_next_%d_days' % horizon, title='Predicted total demand per item (horizon)')
        st.plotly_chart(fig3, use_container_width=True)

# If not pressed
else:
    st.info('Press the **Train model & predict** button to train the model on your data and generate predictions.')

# ------------------------- Footer / Tips -------------------------
st.markdown('---')
st.markdown('**Tips to improve predictions:**')
st.markdown('- Provide at least 3 months of daily sales per item.\n- Include `on_hand` column if you want the app to compute exact reorder quantities.\n- For long-term forecasting, consider time-series models (Prophet / ARIMA) or adding promotion/price features.')

st.markdown('**Need help?** Reply here and tell me: Do you want this app extended to support multi-store, automatic on_hand ingestion, or export to a printable purchase order?')

# End of app
