import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objs as go
from datetime import datetime

# ---------- UI ----------
st.set_page_config(page_title="XAL-FORECAST", layout="centered")
st.image("https://i.imgur.com/9XLt8OX.png", width=120)
st.title("XAL-FORECAST 🟢")
st.markdown("**Soomaali:** Foomkan waxa uu kuu diyaarinayaa liiska dib-u-iibsid 7-ka maalmood soo socda.")

# ---------- SOMALI HOLIDAYS (Hardcoded for 2025) ----------
@st.cache_data
def get_holidays():
    ramadan = pd.DataFrame({
        "holiday": ["Ramadan"] * 30,
        "ds": pd.date_range("2025-03-01", periods=30),
        "lower_window": 0, "upper_window": 1,
    })
    eid = pd.DataFrame({
        "holiday": ["Eid"] * 3,
        "ds": pd.date_range("2025-03-30", periods=3),
        "lower_window": 0, "upper_window": 1,
    })
    return pd.concat([ramadan, eid])

# ---------- SAMPLE DATA ----------
sample = pd.DataFrame({
    "date": pd.date_range("2025-01-01", periods=30).strftime("%Y-%m-%d"),
    "item": ["milk"] * 30,
    "qty": np.random.randint(5, 20, 30),
})
st.download_button("📥 Soo dejiye tusaale CSV", sample.to_csv(index=False), "sample.csv")

# ---------- UPLOAD ----------
uploaded = st.file_uploader("Faylkaaga CSV (date, item, qty)", type=["csv"])
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        df.columns = [c.strip().lower() for c in df.columns]
        
        # Validate columns
        if not {"date", "qty"}.issubset(df.columns):
            st.error("❌ CSV-ga waa lahan 'date' iyo 'qty' tiirar.")
            st.stop()
        
        # Clean data
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "qty"])
        df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0)
        
        # Aggregate daily
        daily = df.groupby("date")["qty"].sum().reset_index()
        daily.columns = ["ds", "y"]
        
        if len(daily) < 14:
            st.warning(f"⚠️ U baahanahay 14 maalmood, hadda waa {len(daily)}.")
            st.stop()
        
        # Forecast
        with st.spinner("AI wuxuu xisaabinayaa..."):
            holidays = get_holidays()
            model = Prophet(holidays=holidays, yearly_seasonality=True, weekly_seasonality=True)
            model.fit(daily)
            future = model.make_future_dataframe(periods=7)
            forecast = model.predict(future).tail(7)
            forecast["reorder"] = forecast["yhat"].clip(lower=0).round(0).astype(int)
        
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast.ds, y=forecast.reorder, mode="lines+markers+text",
                                 text=forecast.reorder, textposition="top center"))
        fig.update_layout(title="7 Maalmood soo socda", xaxis_title="Taariikh", yaxis_title="Tirada")
        st.plotly_chart(fig, use_container_width=True)
        
        # Download CSV
        output = forecast[["ds", "reorder"]].rename(columns={"ds": "date"})
        csv = output.to_csv(index=False)
        st.download_button("📄 Soo dejiye CSV", csv, "reorder_list.csv")
        
        # WhatsApp
        phone = st.text_input("WhatsApp (+25261...)", placeholder="+25261")
        if st.button("📲 Soo dir"):
            if phone.startswith("+252"):
                msg = f"XAL-FORECAST: {output.reorder.sum()} qayb 7 maalmood. Link: https://xal-forecast.streamlit.app"
                url = f"https://api.callmebot.com/whatsapp.php?phone={phone[1:]}&text={msg}"
                try:
                    requests.get(url, timeout=5)
                    st.success("✅ Message waa la diray!")
                except:
                    st.error("❌ Hubi internetka ama CallMeBot setup.")
            else:
                st.error("Nambarka waa qalad.")
                
    except Exception as e:
        st.error(f"❌ Khalad: {str(e)[:80]}")
