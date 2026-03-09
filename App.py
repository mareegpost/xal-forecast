import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Geel Qiime - AI Goat Price Predictor",
    page_icon="🐐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Somali theme
st.markdown("""
<style>
    .main {
        background-color: #f5f5dc;
    }
    .stButton>button {
        background-color: #2e8b57;
        color: white;
        font-size: 18px;
        border-radius: 10px;
    }
    .stHeader {
        color: #8b4513;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
        color: #2e8b57;
    }
    .info-box {
        background-color: #e6f3ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Title and Introduction
st.title("🐐 Geel Qiime - AI Goat Price Predictor")
st.subheader("From Camels to Code: AI/ML for Somali Development")

st.markdown("""
<div class="info-box">
    <h3>🎯 Waa maxay tan? (What is this?)</h3>
    <p>Web appkan wuxuu isticmaalaa Artificial Intelligence si uu u qiyaaso qiimaha geela Soomaaliyeed.</p>
    <p><i>This web app uses AI to predict Somali goat prices based on age, weight, and health.</i></p>
</div>
""", unsafe_allow_html=True)

# Create sample dataset (since we can't upload files easily in web app)
@st.cache_data
def create_sample_data():
    np.random.seed(42)
    n_samples = 150
    
    data = {
        'goat_id': range(1, n_samples + 1),
        'age_months': np.random.randint(5, 25, n_samples),
        'weight_kg': np.random.randint(20, 55, n_samples),
        'health_score': np.random.randint(4, 11, n_samples),
        'region': np.random.choice(['North', 'Central', 'South'], n_samples),
        'season': np.random.choice(['Rainy', 'Dry'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic price based on features
    base_price = 40
    df['price_usd'] = (
        base_price + 
        df['weight_kg'] * 0.8 +           # Weight adds value
        df['health_score'] * 3 +           # Health adds value
        (25 - df['age_months']) * 0.5 +    # Younger is better
        np.where(df['region'] == 'North', 10, 0) +  # North premium
        np.where(df['season'] == 'Rainy', 5, 0) +   # Rainy season premium
        np.random.normal(0, 5, n_samples)  # Random variation
    ).round(2)
    
    return df

# Load data
df = create_sample_data()

# Sidebar navigation
st.sidebar.title("🧭 Hagaha (Navigation)")
page = st.sidebar.radio(
    "Dooro boga (Choose page):",
    ["🏠 Hoyga (Home)", "📊 Xogta (Data)", "🤖 AI Model", "🔮 Qiyaas (Predict)", "📈 Sawirro (Charts)"]
)

# HOME PAGE
if page == "🏠 Hoyga (Home)":
    st.header("🌍 Ku soo dhawoow! (Welcome!)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🇸🇴 Af-Soomaali
        Web appkan wuxuu ka caawiyaa:
        - ✅ Qiyaasta qiimaha geela
        - ✅ Fahamka xiriirka xogta
        - ✅ Barashada AI bilaash ah
        
        **Ma u baahan tahay coding?** Maya! Kaliya isticmaal button-yada.
        """)
    
    with col2:
        st.markdown("""
        ### 🇬🇧 English
        This app helps you:
        - ✅ Predict goat prices
        - ✅ Understand data relationships
        - ✅ Learn AI for free
        
        **Need coding skills?** No! Just use the buttons.
        """)
    
    st.image("https://images.unsplash.com/photo-1569527071833-8f3c3a5b9c1c?w=600", 
             caption="Somali Goats - Our Heritage, Our Data")
    
    st.markdown("""
    <div class="success-box">
        <h3>🚀 Soo koobid (Summary)</h3>
        <p>AI waa muraayad - waxa muujiya waxa aad u dhex dhigto. 
        Halkan waxaan u dhex dhignaa xogta geela Soomaaliyeed!</p>
    </div>
    """, unsafe_allow_html=True)

# DATA PAGE
elif page == "📊 Xogta (Data)":
    st.header("📋 Xogta Geela (Goat Data)")
    
    st.markdown("Halkan waxaa ah 150 geel oo fiktiis ah (sample data):")
    
    # Show data
    st.dataframe(df.style.highlight_max(subset=['price_usd'], color='green'), 
                 use_container_width=True)
    
    # Statistics
    st.subheader("📊 Tirakoobyada (Statistics)")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Wadar (Total Goats)", len(df))
    with col2:
        st.metric("Qiimaha Dhexdhexaad", f"${df['price_usd'].mean():.2f}")
    with col3:
        st.metric("Qiimaha Ugu Sarreeya", f"${df['price_usd'].max():.2f}")
    with col4:
        st.metric("Qiimaha Ugu Hooseeya", f"${df['price_usd'].min():.2f}")
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="⬇️ Soo deg xogta (Download CSV)",
        data=csv,
        file_name='goat_prices.csv',
        mime='text/csv'
    )

# AI MODEL PAGE
elif page == "🤖 AI Model":
    st.header("🤖 Barashada AI-ga (Training the AI)")
    
    st.markdown("""
    ### Tallaabooyinka (Steps):
    1. **Xogta Barashada (Training Data)** - 80% xogta
    2. **Xogta Tijaabada (Test Data)** - 20% xogta  
    3. **Bar (Train)** - AI-ga waxaan baraynaa patterns-ka
    4. **Qiimee (Evaluate)** - Arag sida uu u fiican yahay
    """)
    
    # Prepare data
    X = df[['age_months', 'weight_kg', 'health_score']]
    y = df['price_usd']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    # Show results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="success-box">
            <h3>✅ Natiijooyinka (Results)</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("Accuracy (R²)", f"{r2:.1%}")
        st.metric("Khaladka Dhexdhexaad (RMSE)", f"${rmse:.2f}")
        st.metric("Go'aan (Verdict)", "Waa Fiican!" if r2 > 0.8 else "Waa La Wadaag Karaa")
    
    with col2:
        st.markdown("### Muhiimadda Xogta (Feature Importance)")
        
        importance_df = pd.DataFrame({
            'Waxyaabaha (Feature)': ['Da\'da (Age)', 'Miisaanka (Weight)', 'Caafimaadka (Health)'],
            'Qiimaha (Value)': model.coef_,
            'Muhiimadda (Importance)': np.abs(model.coef_)
        }).sort_values('Muhiimadda (Importance)', ascending=False)
        
        st.dataframe(importance_df, use_container_width=True)
        
        st.info("""
        💡 **Macluumaad:** Miisaanku waa ugu muhiimsan! 
        Kilo kasta oo dheeraad ah waxay ku dartaa qiimo.
        """)
    
    # Show actual vs predicted
    st.subheader("📊 Dhabta vs Qiyaasta (Actual vs Predicted)")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, predictions, alpha=0.6, color='green', s=100)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
            'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Qiimaha Dhabta (Actual Price $)')
    ax.set_ylabel('Qiimaha Qiyaasta (Predicted Price $)')
    ax.set_title('Sidee uu u fiican yahay model-ka?')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

# PREDICTION PAGE
elif page == "🔮 Qiyaas (Predict)":
    st.header("🔮 Qiyaas Geel Cusub (Predict New Goat)")
    
    st.markdown("""
    <div class="info-box">
        <p>Geli xogta geel cusub, AI-ga wuu kuu qiyaasi doonaa qiimaha!</p>
        <p><i>Enter your goat's details, AI will predict the price!</i></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("🎂 Da'da (Age in months)", 
                       min_value=5, max_value=24, value=12)
        st.caption("5-24 bilood (months)")
    
    with col2:
        weight = st.slider("⚖️ Miisaanka (Weight in kg)", 
                          min_value=20, max_value=55, value=35)
        st.caption("20-55 kg")
    
    with col3:
        health = st.slider("❤️ Caafimaadka (Health score)", 
                          min_value=1, max_value=10, value=8)
        st.caption("1-10 (10 = ugu wanaagsan)")
    
    # Prepare model
    X = df[['age_months', 'weight_kg', 'health_score']]
    y = df['price_usd']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict button
    if st.button("🔮 Qiyaas Qiimaha (Predict Price)", use_container_width=True):
        new_goat = [[age, weight, health]]
        predicted_price = model.predict(new_goat)[0]
        
        # Display result with big numbers
        st.markdown(f"""
        <div style="text-align: center; padding: 30px; background-color: #e8f5e9; 
                    border-radius: 15px; margin: 20px 0;">
            <h2>💰 Qiimaha La Qiyaasay (Predicted Price)</h2>
            <p class="big-font">${predicted_price:.2f} USD</p>
            <p>Geel: {age} bilood, {weight} kg, Caafimaad: {health}/10</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Advice
        if predicted_price > 85:
            st.success("🌟 Qiimo sare! Geel wanaagsan. (High price! Good goat.)")
        elif predicted_price > 70:
            st.info("👍 Qiimaha waa dhexdhexaad. (Fair price.)")
        else:
            st.warning("⚠️ Qiimo hoose. Hubi caafimaadka. (Low price. Check health.)")
        
        # Comparison with similar goats
        st.subheader("🐐 Geel La Mid Ah (Similar Goats)")
        similar = df[
            (df['age_months'].between(age-2, age+2)) &
            (df['weight_kg'].between(weight-3, weight+3))
        ].head(5)
        
        if not similar.empty:
            st.dataframe(similar[['age_months', 'weight_kg', 'health_score', 'price_usd']], 
                        use_container_width=True)
            st.caption(f"Qiimaha dhexdhexaad ee la midka ah: ${similar['price_usd'].mean():.2f}")

# CHARTS PAGE
elif page == "📈 Sawirro (Charts)":
    st.header("📈 Sawirro iyo Xog (Charts and Data)")
    
    chart_type = st.selectbox(
        "Dooro nooca sawirka (Choose chart type):",
        ["Scatter Plot", "Bar Chart", "Box Plot", "Heatmap"]
    )
    
    if chart_type == "Scatter Plot":
        st.subheader("Miisaanka vs Qiimaha (Weight vs Price)")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df['weight_kg'], df['price_usd'], 
                           c=df['health_score'], cmap='RdYlGn', 
                           s=100, alpha=0.6, edgecolors='black')
        ax.set_xlabel('Miisaanka (kg)')
        ax.set_ylabel('Qiimaha ($)')
        ax.set_title('Geel: Miisaanka, Qiimaha, iyo Caafimaadka')
        plt.colorbar(scatter, label='Caafimaadka (1-10)')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        st.info("💚 Midabka cagaaran = Caafimaad wanaagsan. ❤️ Midabka gaduudan = Caafimaad xun.")
    
    elif chart_type == "Bar Chart":
        st.subheader("Qiimaha Gobollada (Price by Region)")
        
        region_prices = df.groupby('region')['price_usd'].mean().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(region_prices.index, region_prices.values, 
                     color=['#2e8b57', '#4682b4', '#cd853f'])
        ax.set_ylabel('Qiimaha Dhexdhexaad ($)')
        ax.set_xlabel('Gobolka (Region)')
        ax.set_title('Qiimaha Geela ee Gobollada Kala Duwan')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${height:.2f}', ha='center', va='bottom')
        
        st.pyplot(fig)
        
        st.success(f"🏆 Gobolka ugu qiimaha badan: {region_prices.index[0]}")
    
    elif chart_type == "Box Plot":
        st.subheader("Kala Duwanaanshaha Qiimaha (Price Distribution)")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bp = ax.boxplot([df[df['season']=='Rainy']['price_usd'],
                        df[df['season']=='Dry']['price_usd']],
                       labels=['Roob (Rainy)', 'Abaal (Dry)'],
                       patch_artist=True)
        
        bp['boxes'][0].set_facecolor('#2e8b57')
        bp['boxes'][1].set_facecolor('#cd853f')
        
        ax.set_ylabel('Qiimaha ($)')
        ax.set_title('Qiimaha Xilliyada Kala Duwan')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        rainy_mean = df[df['season']=='Rainy']['price_usd'].mean()
        dry_mean = df[df['season']=='Dry']['price_usd'].mean()
        st.info(f"🌧️ Roob: ${rainy_mean:.2f} vs 🌵 Abaal: ${dry_mean:.2f}")
    
    elif chart_type == "Heatmap":
        st.subheader("Xiriirka Xogta (Data Correlations)")
        
        # Select numeric columns
        numeric_df = df[['age_months', 'weight_kg', 'health_score', 'price_usd']]
        corr = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='RdYlGn', center=0,
                   square=True, ax=ax, cbar_kws={'label': 'Xiriirka (Correlation)'})
        ax.set_title('Xiriirka Waxyaabaha Kala Duwan')
        
        st.pyplot(fig)
        
        st.markdown("""
        **Macnaha Midabka:**
        - 🟢 Cagaaran = Xiriir togan (wax kasta oo kordha, kan kale wuu kordhaa)
        - 🔴 Gaduudan = Xiriir taban (wax kasta oo kordha, kan kale wuu yaraadaa)
        - ⬜ Caddaan = Xiriir la'aan
        """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
🎓 **Baro AI bilaash ah**
- No coding required
- Works on any phone
- Free forever

**Created for:** Somali University Students
""")

st.sidebar.markdown("""
---
🔗 **Xiriirka (Links):**
- [GitHub](https://github.com/yourusername)
- [Email](mailto:your.email@university.edu.so)
""")
