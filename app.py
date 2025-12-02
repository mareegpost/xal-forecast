import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objs as go
import base64
from datetime import datetime

# ---------- CONFIG ----------
st.set_page_config(page_title="XAL-FORECAST", layout="centered")
st.markdown(
    """
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button { background-color: #00a651; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- SOMALI HOLIDAYS ----------
@st.cache_data
def get_somali_holidays():
    ramadan_start = pd.to_datetime("2025-03-01")  # Update yearly
    ramadan = pd.DataFrame({
        "holiday": ["Ramadan"] * 30,
        "ds": pd.date_range(ramadan_start, periods=30),
        "lower_window": 0, "upper_window": 1,
    })
    eid_fitr = pd.DataFrame({
        "holiday": ["Eid_Fitr"] * 3,
        "ds": pd.date_range("2025-03-30", periods=3),
        "lower_window": 0, "upper_window": 1,
    })
    return pd.concat([ramadan, eid_fitr])

# ---------- UI ----------
st.image("https://i.imgur.com/9XLt8OX.png", width=120)
st.title("XAL-FORECAST 🟢")
st.markdown("**Soomaali:** Foomkan waxa uu kuu diyaarinayaa liiska dib-u-iibsid 7-ka maalmood soo socda.")

# ---------- SAMPLE DATA ----------
sample = pd.DataFrame({
    "date": pd.date_range("2025-01-01", periods=30).strftime("%Y-%m-%d"),
    "item": ["milk"] * 30,
    "qty": np.random.randint(5, 20, 30),
})
st.download_button("📥 Soo dejiye tusaale CSV", sample.to_csv(index=False), "sample.csv", mime="text/csv")

# ---------- FILE UPLOAD ----------
uploaded = st.file_uploader("CSV faylkaaga (date, item, qty)", type=["csv"])
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        df.columns = [c.strip().lower() for c in df.columns]
        
        if not {"date", "qty"}.issubset(df.columns):
            st.error("❌ CSV-gaagu ma lahan 'date' iyo 'qty' tiirar. Fadlan eeg tusaale.")
            st.stop()
        
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "qty"])
        df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0)
        
        daily = df.groupby("date")["qty"].sum().reset_index()
        daily.columns = ["ds", "y"]
        
        if len(daily) < 14:
            st.warning("⚠️ U baahanahay ugu yaraan 14 maalmood. Hadda waa {len(daily)}.")
            st.stop()
        
        # ---------- FORECAST ----------
        with st.spinner("AI wuxuu xisaabinayaa..."):
            holidays = get_somali_holidays()
            model = Prophet(holidays=holidays, yearly_seasonality=True, weekly_seasonality=True)
            model.fit(daily)
            future = model.make_future_dataframe(periods=7)
            forecast = model.predict(future).tail(7)
            forecast["reorder_qty"] = forecast["yhat"].clip(lower=0).round(0).astype(int)
        
        # ---------- VISUAL ----------
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast.ds, y=forecast.reorder_qty, mode="lines+markers+text",
                                 text=forecast.reorder_qty, textposition="top center"))
        fig.update_layout(title="7 Maalmood soo socda", xaxis_title="Taariikh", yaxis_title="Tirada")
        st.plotly_chart(fig, use_container_width=True)
        
        # ---------- DOWNLOAD ----------
        output = forecast[["ds", "reorder_qty"]].rename(columns={"ds": "date"})
        csv = output.to_csv(index=False)
        st.download_button("📄 Soo dejiye CSV", csv, "reorder_list.csv", mime="text/csv")
        
        # ---------- HTML REPORT (replaces PDF) ----------
        html = f"""
        <h1>XAL-FORECAST Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        <table border="1" style="width:100%; border-collapse:collapse;">
          <tr style="background:#00a651; color:white;"><th>Date</th><th>Reorder Qty</th></tr>
          {''.join([f"<tr><td>{row.date.strftime('%Y-%m-%d')}</td><td>{row.reorder_qty}</td></tr>" for _, row in output.iterrows()])}
        </table>
        """
        st.download_button("📄 Soo dejiye HTML Report", html, "report.html", mime="text/html")
        
        # ---------- WHATSAPP (Simple) ----------
        phone = st.text_input("WhatsApp (+25261...)", placeholder="+25261")
        if st.button("📲 Soo dir qoraal gaaban"):
            if phone.startswith("+252"):
                msg = f"XAL-FORECAST: {output.reorder_qty.sum()} urur dib-u-iibsan ah 7 maalmood. Link: https://xal-forecast.streamlit.app"
                url = f"https://api.callmebot.com/whatsapp.php?phone={phone[1:]}&text={msg}&apikey="
                try:
                    r = requests.get(url, timeout=10)
                    if r.status_code == 200:
                        st.success("✅ Message waa la diray!")
                    else:
                        st.error("❌ CallMeBot waa loo baahan yahay inaad hore u furato.")
                except:
                    st.error("❌ Internet ma jiro ama nambarka waa qalad.")
            else:
                st.error("Fadlan gali +25261...")
                
    except Exception as e:
        st.error(f"❌ Khalad: {str(e)[:100]}")
        st.info("Fadlan eeg tusaale CSV-ga sare.")
