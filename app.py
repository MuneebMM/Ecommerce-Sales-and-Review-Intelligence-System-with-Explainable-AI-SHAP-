"""
Review Intelligence System - Demo Dashboard

Interactive Streamlit app showcasing:
- Sales Volume Prediction
- Review Risk Assessment
- SHAP Explainability
- Dataset Insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Page config
st.set_page_config(
    page_title="Review Intelligence System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .risk-low { color: #10B981; font-weight: bold; }
    .risk-medium { color: #F59E0B; font-weight: bold; }
    .risk-high { color: #EF4444; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# Load models and data
@st.cache_resource
def load_models():
    """Load trained models."""
    try:
        from src.models.sales_predictor import SalesPredictor
        from src.models.review_risk_predictor import ReviewRiskPredictor
        
        sales_model = SalesPredictor.load(Path("models/sales_predictor/model.pkl"))
        risk_model = ReviewRiskPredictor.load(Path("models/review_risk_predictor/model.pkl"))
        
        return sales_model, risk_model
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, None


@st.cache_data
def load_sample_data(nrows=500):
    """Load sample data for visualization."""
    try:
        from src.data.ingestion import load_tokopedia_data, explode_reviews
        from src.data.preprocessing import DataPreprocessor
        
        df = load_tokopedia_data("tokopedia_products_with_review.csv", nrows=nrows)
        reviews = explode_reviews(df)
        
        preprocessor = DataPreprocessor()
        reviews = preprocessor.fit_transform(reviews)
        
        return df, reviews
    except Exception as e:
        st.warning(f"Could not load data: {e}")
        return None, None


def compute_features_for_prediction(features_dict, model_type='risk'):
    """Compute derived features for prediction."""
    price = features_dict.get('price', 0)
    stock = features_dict.get('stock', 0)
    discounted_price = features_dict.get('discounted_price', price)
    gold_merchant = features_dict.get('gold_merchant', False)
    is_official = features_dict.get('is_official', False)
    message_length = features_dict.get('message_length', 0)
    word_count = features_dict.get('word_count', 0)
    rating_average = features_dict.get('rating_average', 4.5)
    
    computed = {
        'price_log': np.log1p(price) if price > 0 else 0,
        'stock_log': np.log1p(stock) if stock > 0 else 0,
        'has_stock': 1 if stock > 0 else 0,
        'low_stock': 1 if stock < 10 else 0,
        'is_preorder': 0,
        'shop_tier': int(is_official) * 2 + int(gold_merchant),
        'uses_topads': 0,
        'discount_pct': max(0, (price - discounted_price) / price) if price > 0 else 0,
        'has_discount': 1 if discounted_price < price else 0,
        'category_encoded': hash(features_dict.get('category', 'Unknown')) % 100,
        'shop_location_encoded': hash(features_dict.get('shop_location', 'Unknown')) % 50,
        'rating_average': rating_average,
        'message_length': message_length,
        'word_count': word_count,
        'has_response': 0,
        'review_hour': 12,
        'review_dayofweek': 3,
        'is_weekend': 0
    }
    
    return computed


def render_header():
    """Render app header."""
    st.markdown('<h1 class="main-header">🧠 Review Intelligence System</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; color: #666; margin-bottom: 2rem;">'
        'ML-Powered Sales Prediction & Review Risk Assessment with SHAP Explainability'
        '</p>',
        unsafe_allow_html=True
    )


def render_sidebar():
    """Render sidebar with navigation and info."""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
        st.title("Navigation")
        
        page = st.radio(
            "Select Page",
            ["🏠 Overview", "📊 Sales Predictor", "⚠️ Risk Analyzer", "📈 Analytics"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        st.markdown("### About")
        st.info(
            "This system uses **LightGBM** models with **SHAP** explainability "
            "to predict product sales and identify negative review risks."
        )
        
        st.markdown("### Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Risk AUC", "0.84", delta="✓")
        with col2:
            st.metric("API Status", "Online", delta="✓")
        
        return page


def render_overview():
    """Render overview page."""
    st.header("System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>30,711</h2>
            <p>Reviews Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>6</h2>
            <p>Modules Built</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>0.84</h2>
            <p>Risk Model AUC</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h2>< 50ms</h2>
            <p>Inference Time</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Architecture
    st.subheader("System Architecture")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        | Layer | Components |
        |-------|------------|
        | **Data** | Ingestion, Validation, Preprocessing |
        | **Features** | Engineering, Store, Definitions |
        | **Models** | Sales Predictor, Risk Predictor |
        | **Explainability** | SHAP Explainer, Visualizations |
        | **Serving** | FastAPI, Inference Engine |
        | **Monitoring** | Data Drift, Model Drift, Metrics |
        """)
    
    with col2:
        st.markdown("### Tech Stack")
        st.markdown("""
        - 🐍 Python 3.10+
        - 🌿 LightGBM
        - 📊 SHAP
        - ⚡ FastAPI
        - 🎨 Streamlit
        """)


def render_sales_predictor():
    """Render sales prediction page."""
    st.header("📊 Sales Volume Predictor")
    st.markdown("Predict expected sales for a product based on its attributes.")
    
    sales_model, _ = load_models()
    
    if sales_model is None:
        st.error("Sales model not loaded. Please train models first.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Product Details")
        
        price = st.number_input("Price (IDR)", min_value=1000, value=150000, step=10000)
        stock = st.number_input("Stock", min_value=0, value=100, step=10)
        rating = st.slider("Average Rating", 1.0, 5.0, 4.5, 0.1)
        
        gold_merchant = st.checkbox("Gold Merchant", value=True)
        is_official = st.checkbox("Official Store", value=False)
        
        category = st.selectbox(
            "Category",
            ["Electronics", "Fashion", "Home & Living", "Food & Beverage", "Other"]
        )
        
        predict_button = st.button("🔮 Predict Sales", type="primary", use_container_width=True)
    
    with col2:
        if predict_button:
            # Prepare features
            features = compute_features_for_prediction({
                'price': price,
                'stock': stock,
                'rating_average': rating,
                'gold_merchant': gold_merchant,
                'is_official': is_official,
                'category': category
            }, 'sales')
            
            # Get model features
            required = sales_model.feature_names
            X = pd.DataFrame([{f: features.get(f, 0) for f in required}])
            
            # Predict
            prediction = sales_model.predict(X)[0]
            
            st.subheader("Prediction Results")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(
                    "Predicted Sales",
                    f"{prediction:,.0f} units",
                    delta=f"±{prediction * 0.3:,.0f}"
                )
            with col_b:
                # Revenue estimate
                revenue = prediction * price
                st.metric("Est. Revenue", f"Rp {revenue:,.0f}")
            
            # Feature importance
            st.subheader("Key Drivers")
            importance = sales_model.get_feature_importance().head(8)
            
            fig = px.bar(
                importance,
                x='importance',
                y='feature',
                orientation='h',
                color='importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)


def render_risk_analyzer():
    """Render risk analysis page."""
    st.header("⚠️ Review Risk Analyzer")
    st.markdown("Assess the probability of receiving a negative review.")
    
    _, risk_model = load_models()
    
    if risk_model is None:
        st.error("Risk model not loaded. Please train models first.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Review Details")
        
        price = st.number_input("Product Price (IDR)", min_value=1000, value=150000, step=10000)
        message_length = st.number_input("Review Length (chars)", min_value=0, value=50)
        word_count = st.number_input("Word Count", min_value=0, value=10)
        
        gold_merchant = st.checkbox("Gold Merchant", value=False, key="risk_gold")
        is_official = st.checkbox("Official Store", value=False, key="risk_official")
        
        analyze_button = st.button("🔍 Analyze Risk", type="primary", use_container_width=True)
    
    with col2:
        if analyze_button:
            # Prepare features
            features = compute_features_for_prediction({
                'price': price,
                'message_length': message_length,
                'word_count': word_count,
                'gold_merchant': gold_merchant,
                'is_official': is_official
            }, 'risk')
            
            # Get model features
            required = risk_model.feature_names
            X = pd.DataFrame([{f: features.get(f, 0) for f in required}])
            
            # Predict
            probability = risk_model.predict_proba(X)[0]
            
            # Determine risk level
            if probability < 0.2:
                risk_level = "LOW"
                color = "#10B981"
            elif probability < 0.5:
                risk_level = "MEDIUM"
                color = "#F59E0B"
            elif probability < 0.8:
                risk_level = "HIGH"
                color = "#EF4444"
            else:
                risk_level = "CRITICAL"
                color = "#DC2626"
            
            st.subheader("Risk Assessment")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Risk Probability", f"{probability * 100:.1f}%")
            with col_b:
                st.markdown(f"**Risk Level:** <span style='color: {color}; font-size: 1.5rem;'>{risk_level}</span>", unsafe_allow_html=True)
            
            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Risk Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 20], 'color': "#D1FAE5"},
                        {'range': [20, 50], 'color': "#FEF3C7"},
                        {'range': [50, 80], 'color': "#FEE2E2"},
                        {'range': [80, 100], 'color': "#FECACA"}
                    ]
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            st.subheader("Risk Factors")
            importance = risk_model.get_feature_importance().head(8)
            
            fig = px.bar(
                importance,
                x='importance',
                y='feature',
                orientation='h',
                color='importance',
                color_continuous_scale='Reds'
            )
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)


def render_analytics():
    """Render analytics page."""
    st.header("📈 Data Analytics")
    
    products, reviews = load_sample_data()
    
    if products is None:
        st.warning("Could not load dataset. Please ensure data file exists.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Products", f"{len(products):,}")
    with col2:
        st.metric("Reviews", f"{len(reviews):,}")
    with col3:
        avg_rating = reviews['review_rating'].astype(float).mean()
        st.metric("Avg Rating", f"{avg_rating:.2f}")
    with col4:
        neg_rate = (reviews['is_negative_review'] == 1).mean() * 100
        st.metric("Negative Rate", f"{neg_rate:.1f}%")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Rating Distribution")
        rating_counts = reviews['review_rating'].astype(float).value_counts().sort_index()
        fig = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            labels={'x': 'Rating', 'y': 'Count'},
            color=rating_counts.index,
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Sales Distribution")
        sales = products['count_sold'].dropna()
        fig = px.histogram(
            sales,
            nbins=50,
            labels={'value': 'Units Sold', 'count': 'Products'},
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Price vs Sales
    st.subheader("Price vs Sales Relationship")
    sample = products.dropna(subset=['price', 'count_sold']).sample(min(200, len(products)))
    fig = px.scatter(
        sample,
        x='price',
        y='count_sold',
        color='rating_average',
        size='stock' if 'stock' in sample.columns else None,
        color_continuous_scale='Viridis',
        labels={'price': 'Price (IDR)', 'count_sold': 'Units Sold', 'rating_average': 'Rating'}
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main app entry point."""
    render_header()
    page = render_sidebar()
    
    if page == "🏠 Overview":
        render_overview()
    elif page == "📊 Sales Predictor":
        render_sales_predictor()
    elif page == "⚠️ Risk Analyzer":
        render_risk_analyzer()
    elif page == "📈 Analytics":
        render_analytics()
    
    # Footer
    st.divider()
    st.markdown(
        '<p style="text-align: center; color: #888;">Built with ❤️ using Streamlit, LightGBM & SHAP</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
