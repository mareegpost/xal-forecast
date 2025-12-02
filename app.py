import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objs as go
import base64
from datetime import datetime, timedelta
import requests
import io
from weasyprint import HTML

# ---------- CONFIG ----------
st.set_page_config(page_title="XAL-FORECAST", layout="centered", initial_sidebar_state="collapsed")
st.image("https://i.imgur.com/9XLt8OX.png", width=120)  # Somali flag logo

# ---------- SOMALI HOLIDAYS ----------
@st.cache_data
def get_somali_holidays(year=2025):
    ramadan_start = datetime(year, 2, 28)
    eid_fitr = datetime(year, 3, 30)
    eid_adha = datetime(year, 6, 6)
    dates = []
    for i in range(30):
        dates.append(ramadan_start + timedelta(days=i))
    holidays = pd.DataFrame({
        "holiday": ["Ramadan"]*30 + ["Eid_Fitr"]*3 + ["Eid_Adha"]*3,
        "ds": pd.to_datetime(dates + [eid_fitr, eid_fitr+timedelta(1), eid_fitr+timedelta(2)] +
                                       [eid_adha, eid_adha+timedelta(1), eid_adha+timedelta(2)]),
        "lower_window": 0, "upper_window": 0,
    })
    return holidays

# ---------- UI ----------
st.title("XAL-FORECAST 🟢")
st.markdown("**Soomaali:** Foomkan waxa uu kuu diyaarinayaa liiska dib-u-iibsid 7-ka maalmood soo socda.")
st.info("📤 Soo deji CSV fayl: date, item, qty (tirada la iibiyay)")

# ---------- DATA UPLOAD ----------
uploaded = st.file_uploader("Faylka CSV", type=["csv"], label_visibility="collapsed")
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        df.columns = [c.strip().lower() for c in df.columns]
        required = {'date', 'item', 'qty'}
        if not required.issubset(set(df.columns)):
            st.error(f"Khalad: Weydiiso tiirarka {required}")
            st.stop()
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date', 'qty'])
        df['qty'] = pd.to_numeric(df['qty'], errors='coerce').fillna(0)
        
        # Aggregate daily
        daily = df.groupby('date')['qty'].sum().reset_index()
        daily.columns = ['ds', 'y']
        daily = daily.sort_values('ds')
        
        if len(daily) < 14:
            st.warning("⚠️ Xogta aad soo dirtay way yartahay. U baahanahay ugu yaraan 14 maalmood.")
            st.stop()
        
        # ---------- PROPHET ----------
        with st.spinner("AI wuxuu xisaabinayaa 7-ka maalmood soo socda..."):
            holidays = get_somali_holidays()
            m = Prophet(holidays=holidays, daily_seasonality=False, weekly_seasonality=True)
            m.fit(daily)
            future = m.make_future_dataframe(periods=7)
            fcst = m.predict(future)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)
            fcst['yhat'] = fcst['yhat'].clip(lower=0).round(0)
            fcst['need_to_reorder'] = fcst['yhat'].astype(int)
        
        # ---------- VISUALIZE ----------
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fcst.ds, y=fcst.need_to_reorder, mode='lines+markers+text',
                                 text=fcst.need_to_reorder, textposition="top center", name='Dib-u-iibso'))
        fig.update_layout(title="7 Maalmood soo socda", xaxis_title="Taariikh", yaxis_title="Tirada")
        st.plotly_chart(fig, use_container_width=True)
        
        # ---------- DOWNLOAD CSV ----------
        csv = fcst[['ds', 'need_to_reorder']].to_csv(index=False, encoding='utf-8')
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="reorder_list.csv">📥 Soo dejiye liiska CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        # ---------- PDF REPORT ----------
        html_report = f"""
        <h1>XAL-FORECAST Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        <table border="1" style="border-collapse: collapse; width: 100%;">
          <tr><th>Maalinta</th><th>Dib u iibso</th></tr>
          {''.join([f"<tr><td>{row.ds.strftime('%Y-%m-%d')}</td><td>{int(row.need_to_reorder)}</td></tr>" for _,row in fcst.iterrows()])}
        </table>
        <p>📊 <em>AI waxa uu tixgeliyaa saacadaha shaqada, fasaxa, & Ramadan.</em></p>
        """
        
        pdf = HTML(string=html_report).write_pdf()
        b64_pdf = base64.b64encode(pdf).decode()
        href_pdf = f'<a href="data:application/pdf;base64,{b64_pdf}" download="xal_forecast_report.pdf">📄 Soo dejiye PDF</a>'
        st.markdown(href_pdf, unsafe_allow_html=True)
        
        # ---------- WHATSAPP ----------
        phone = st.text_input("WhatsApp number (+25261xxxxxx)", max_chars=13, placeholder="+25261...")
        if st.button("📲 Soo dir PDF WhatsApp"):
            if phone and phone.startswith("+252"):
                msg = f"XAL-FORECAST Report: {datetime.now().strftime('%Y-%m-%d')}. Link: https://share.streamlit.io/mareegpost/xal-forecast"
                url = f"https://api.callmebot.com/whatsapp.php?phone={phone[1:]}&text={msg}"
                try:
                    requests.get(url, timeout=10)
                    st.success("✅ Report waa la diray WhatsApp!")
                except:
                    st.error("❌ WhatsApp ma aaday. Kontarol nambarka.")
            else:
                st.error("Fadlan gali nambarka saxda ah (+25261...)")
        
    except Exception as e:
        st.error(f"Khalad: {str(e)}")
        st.stop()

else:
    st.markdown("### 📋 Tusaale CSV format:")
    sample = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "item": ["milk", "bread", "milk"],
        "qty": [12, 8, 15]
    })
    st.dataframe(sample, use_container_width=True)
    st.download_button("📥 Soo dejiye sample CSV fayl", sample.to_csv(index=False), "sample.csv")
