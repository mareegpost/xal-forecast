"""
Ari Qiime - AI Goat Price Predictor
Optimized for Streamlit Cloud (Python 3.14)
Bilingual: Somali + English
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

# ============== PAGE SETUP ==============

st.set_page_config(
    page_title="Ari Qiime 🐐",
    page_icon="🐐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== CUSTOM STYLES ==============

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2e8b57;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #8b4513;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2e8b57;
        margin: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .info-banner {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
    }
    .success-banner {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# ============== DATA GENERATION ==============

@st.cache_data
def load_goat_data():
    """Generate realistic Somali goat market data"""
    np.random.seed(42)
    n = 150
    
    data = {
        'id': range(1, n + 1),
        'daDa_bilaha': np.random.randint(5, 25, n),  # age_months
        'miisaanka_kg': np.random.randint(22, 52, n),  # weight_kg
        'caafimaadka': np.random.randint(4, 11, n),  # health_score
        'gobolka': np.random.choice(['Waqooyi', 'Dhexe', 'Koonfur'], n),
        'xilliga': np.random.choice(['Roob', 'Abaal'], n),
    }
    
    df = pd.DataFrame(data)
    
    # Realistic price calculation
    base = 35
    df['qiimaha_usd'] = (
        base +
        df['miisaanka_kg'] * 0.85 +
        df['caafimaadka'] * 2.5 +
        (24 - df['daDa_bilaha']) * 0.6 +
        np.where(df['gobolka'] == 'Waqooyi', 8, 0) +
        np.where(df['xilliga'] == 'Roob', 6, 0) +
        np.random.normal(0, 4, n)
    ).round(2)
    
    return df

df = load_goat_data()

# ============== SIDEBAR NAVIGATION ==============

st.sidebar.title("🧭 Meesha aad rabto")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Dooro bogga / Choose page:",
    [
        "🏠 Hoyga / Home",
        "📊 Xogta / Data", 
        "🤖 Barashada / Training",
        "💰 Qiyaas / Predict",
        "📈 Sawirro / Charts"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
🎓 **Baro AI bilaash ah**
- Learn AI for free
- No coding needed
- Works on any phone
""")

# ============== PAGE 1: HOME ==============

if page == "🏠 Hoyga / Home":
    
    st.markdown('<p class="main-header">🐐 Ari Qiime</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI Goat Price Predictor for Somalia</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>🇺🇸 English</h3>
        <p>This app uses <b>Artificial Intelligence</b> to predict goat prices 
        based on age, weight, and health - just like experienced Somali elders!</p>
        <ul>
        <li>✅ 150 sample goats from Somali markets</li>
        <li>✅ Real-time price predictions</li>
        <li>✅ Works on any phone browser</li>
        <li>✅ 100% free, no installation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>🇸🇴 Af-Soomaali</h3>
        <p>App-kaan wuxuu isticmaalaa <b>Artificial Intelligence</b> si uu 
        u qiyaaso qiimaha geela - sida adeerada Soomaaliyeed oo waayo-aragnimo leh!</p>
        <ul>
        <li>✅ 150 geel oo ka yimid suuqyada Soomaaliyeed</li>
        <li>✅ Qiyaasta qiimaha waqti-dhab ah</li>
        <li>✅ Wuxuu shaqayaa browser-ka taleefanka</li>
        <li>✅ 100% bilaash, ma u baahan rakib</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Key concept
    st.markdown("""
    <div class="info-banner">
    <h4>💡 Xasuus / Remember:</h4>
    <p><b>"AI waa muraayad"</b> - waxa muujiya waxa aad u dhex dhigto, ma aha maskax shaqaysa.</p>
    <p><i>"AI is a mirror" - it shows what you put in, not a magic brain.</i></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick stats
    st.subheader("📊 Tusaale ahaan / Quick Preview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Goats", "150")
    c2.metric("Avg Price", f"${df['qiimaha_usd'].mean():.0f}")
    c3.metric("Regions", "3")
    c4.metric("Features", "5")

# ============== PAGE 2: DATA ==============

elif page == "📊 Xogta / Data":
    
    st.header("📋 Xogta Aria / Goat Data")
    
    tab1, tab2 = st.tabs(["📊 Table / Jedwal", "📈 Stats / Tirakoobyada"])
    
    with tab1:
        st.markdown("Halkan waxaa ah 150 geel oo fiktiis ah / Here are 150 sample goats:")
        
        # Display with formatting
        display_df = df.copy()
        display_df.columns = ['ID', 'Age (mo)', 'Weight (kg)', 'Health', 'Region', 'Season', 'Price ($)']
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Download
        csv = df.to_csv(index=False)
        st.download_button(
            "⬇️ Soo deg CSV / Download CSV",
            csv,
            "goat_prices.csv",
            "text/csv"
        )
    
    with tab2:
        st.subheader("Tirakoobyada Guud / Summary Statistics")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Qiimaha Ugu Sarreeya / Max Price", f"${df['qiimaha_usd'].max():.2f}")
        c2.metric("Qiimaha Dhexdhexaad / Mean Price", f"${df['qiimaha_usd'].mean():.2f}")
        c3.metric("Qiimaha Ugu Hooseeya / Min Price", f"${df['qiimaha_usd'].min():.2f}")
        c4.metric("Kala Duwanaanshaha / Std Dev", f"${df['qiimaha_usd'].std():.2f}")
        
        # Region breakdown
        st.subheader("Qiimaha Gobollada / Prices by Region")
        region_stats = df.groupby('gobolka')['qiimaha_usd'].agg(['mean', 'count']).round(2)
        region_stats.columns = ['Average Price ($)', 'Count']
        st.dataframe(region_stats)

# ============== PAGE 3: TRAINING ==============

elif page == "🤖 Barashada / Training":
    
    st.header("🤖 Barashada AI-ga / Training the AI")
    
    # Prepare features
    X = df[['daDa_bilaha', 'miisaanka_kg', 'caafimaadka']]
    y = df['qiimaha_usd']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    # Results display
    st.markdown("""
    <div class="success-banner">
    <h3>✅ AI-ga waa la baray! / AI Trained!</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy (R²)", f"{r2:.1%}", "Waa Fiican! / Good!" if r2 > 0.8 else "OK")
    col2.metric("Average Error", f"${rmse:.2f}", "±$4-5 typical")
    col3.metric("Test Goats", f"{len(y_test)}", "20% of data")
    
    # Feature importance
    st.subheader("Muhiimadda Xogta / Feature Importance")
    
    importance = pd.DataFrame({
        'Waxyaabaha / Feature': ['Age (months)', 'Weight (kg)', 'Health (1-10)'],
        'Qiimaha / Value': model.coef_,
        'Saamaynta / Impact': ['Lower is better' if c < 0 else 'Higher is better' for c in model.coef_]
    })
    
    # Add absolute importance for sorting
    importance['|Muhiimadda|'] = np.abs(importance['Qiimaha / Value'])
    importance = importance.sort_values('|Muhiimadda|', ascending=False)
    
    st.dataframe(importance[['Waxyaabaha / Feature', 'Qiimaha / Value', 'Saamaynta / Impact']], 
                use_container_width=True)
    
    # Visualization
    st.subheader("Dhabta vs Qiyaasta / Actual vs Predicted")
    
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=y_test, y=predictions,
        mode='markers',
        name='Ari / Goats',
        marker=dict(size=10, color='green', opacity=0.6)
    ))
    
    # Perfect prediction line
    fig.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode='lines',
        name='Qiyaas Sax / Perfect',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        xaxis_title='Qiimaha Dhabta / Actual Price ($)',
        yaxis_title='Qiimaha Qiyaasta / Predicted Price ($)',
        title='Sidee uu u fiican yahay AI-ga? / How good is the AI?',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============== PAGE 4: PREDICTION ==============

elif page == "💰 Qiyaas / Predict":
    
    st.header("🔮 Qiyaas Ari Cusub / Predict New Goat")
    
    st.markdown("""
    <div class="info-banner">
    Geli xogta geelkaaga / Enter your goat's details:
    </div>
    """, unsafe_allow_html=True)
    
    # Input columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider(
            "🎂 Da'da / Age",
            min_value=5, max_value=24, value=12,
            help="Bilaha / Months"
        )
        st.caption("5-24 bilood / months")
    
    with col2:
        weight = st.slider(
            "⚖️ Miisaanka / Weight",
            min_value=20, max_value=55, value=35,
            help="Kiloogaraam / Kilograms"
        )
        st.caption("20-55 kg")
    
    with col3:
        health = st.slider(
            "❤️ Caafimaadka / Health",
            min_value=1, max_value=10, value=8,
            help="10 = ugu wanaagsan / best"
        )
        st.caption("1-10 score")
    
    # Train model for prediction
    X = df[['daDa_bilaha', 'miisaanka_kg', 'caafimaadka']]
    y = df['qiimaha_usd']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict button
    if st.button("🔮 QIYAAS / PREDICT", use_container_width=True, type="primary"):
        
        # Make prediction
        new_goat = [[age, weight, health]]
        predicted = model.predict(new_goat)[0]
        
        # Display result
        st.markdown(f"""
        <div class="prediction-box">
            💰 ${predicted:.2f} USD
        </div>
        """, unsafe_allow_html=True)
        
        # Feedback
        if predicted > 85:
            st.success("🌟 Qiimo sare! / High price! - Ari wanaagsan / Good goat")
        elif predicted > 70:
            st.info("👍 Qiimo dhexdhexaad / Fair price")
        else:
            st.warning("⚠️ Qiimo hoose / Low price - Hubi caafimaadka / Check health")
        
        # Similar goats
        st.subheader("Ari La Mid Ah / Similar Goats")
        
        similar = df[
            (df['daDa_bilaha'].between(age-2, age+2)) &
            (df['miisaanka_kg'].between(weight-3, weight+3))
        ].head(5)
        
        if len(similar) > 0:
            sim_display = similar[['daDa_bilaha', 'miisaanka_kg', 'caafimaadka', 'qiimaha_usd']].copy()
            sim_display.columns = ['Age', 'Weight', 'Health', 'Price ($)']
            st.dataframe(sim_display, use_container_width=True)
            
            avg_similar = similar['qiimaha_usd'].mean()
            st.caption(f"Qiimaha dhexdhexaad ee la midka ah / Average similar price: ${avg_similar:.2f}")

# ============== PAGE 5: CHARTS ==============

elif page == "📈 Sawirro / Charts":
    
    st.header("📊 Sawirro Kala Duwan / Different Charts")
    
    chart_type = st.selectbox(
        "Dooro nooca sawirka / Choose chart type:",
        [
            "Scatter: Miisaanka vs Qiimaha / Weight vs Price",
            "Bar: Gobollada / By Region", 
            "Box: Xilliyada / By Season",
            "Histogram: Qoondeeyska Qiimaha / Price Distribution"
        ]
    )
    
    if chart_type == "Scatter: Miisaanka vs Qiimaha / Weight vs Price":
        
        fig = px.scatter(
            df, x='miisaanka_kg', y='qiimaha_usd',
            color='caafimaadka',
            size='caafimaadka',
            hover_data=['daDa_bilaha', 'gobolka'],
            labels={
                'miisaanka_kg': 'Miisaanka (kg) / Weight',
                'qiimaha_usd': 'Qiimaha ($) / Price',
                'caafimaadka': 'Caafimaadka / Health'
            },
            title='Ari: Miisaanka, Qiimaha, iyo Caafimaadka / Goats: Weight, Price & Health',
            color_continuous_scale='RdYlGn'
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💚 Cagaaran = Caafimaad wanaagsan / Green = Good health | ❤️ Gaduudan = Caafimaad xun / Red = Poor health")
    
    elif chart_type == "Bar: Gobollada / By Region":
        
        region_avg = df.groupby('gobolka')['qiimaha_usd'].mean().reset_index()
        region_avg = region_avg.sort_values('qiimaha_usd', ascending=False)
        
        fig = px.bar(
            region_avg, x='gobolka', y='qiimaha_usd',
            labels={
                'gobolka': 'Gobolka / Region',
                'qiimaha_usd': 'Qiimaha Dhexdhexaad ($) / Average Price'
            },
            title='Qiimaha Aria ee Gobollada / Goat Prices by Region',
            color='qiimaha_usd',
            color_continuous_scale='Greens'
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        winner = region_avg.iloc[0]['gobolka']
        st.success(f"🏆 Gobolka ugu qiimaha badan / Highest price region: {winner}")
    
    elif chart_type == "Box: Xilliyada / By Season":
        
        fig = px.box(
            df, x='xilliga', y='qiimaha_usd',
            labels={
                'xilliga': 'Xilliga / Season',
                'qiimaha_usd': 'Qiimaha ($) / Price'
            },
            title='Kala Duwanaanshaha Xilliyada / Seasonal Price Variation',
            color='xilliga',
            color_discrete_map={'Roob': '#2e8b57', 'Abaal': '#cd853f'}
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        rainy_avg = df[df['xilliga']=='Roob']['qiimaha_usd'].mean()
        dry_avg = df[df['xilliga']=='Abaal']['qiimaha_usd'].mean()
        
        col1, col2 = st.columns(2)
        col1.metric("🌧️ Roob / Rainy", f"${rainy_avg:.2f}")
        col2.metric("🌵 Abaal / Dry", f"${dry_avg:.2f}")
    
    else:  # Histogram
        
        fig = px.histogram(
            df, x='qiimaha_usd',
            nbins=20,
            labels={'qiimaha_usd': 'Qiimaha ($) / Price'},
            title='Qoondeeyska Qiimaha / Price Distribution',
            color_discrete_sequence=['#2e8b57']
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption(f"Qiimaha ugu badan ee joogtada ah / Most common price range: ${df['qiimaha_usd'].mode().iloc[0]:.0f}")

# ============== FOOTER ==============

st.sidebar.markdown("---")
st.sidebar.markdown("""
🐐 **Ari Qiime v1.0**

Made with ❤️ for Somali students learning AI

🔗 Links:
- [Streamlit](https://streamlit.io)
- [Scikit-learn](https://scikit-learn.org)
""")
