import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import os

# Page Configuration
st.set_page_config(
    layout="wide",
    page_title="Mobile Market Intelligence",
    page_icon="📱"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4;}
    .sub-header {font-size: 1.5rem; color: #ff7f0e;}
    .metric-container {background-color: #f0f2f6; padding: 20px; border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

# Load artifacts with error handling
@st.cache_resource
def load_artifacts():
    required_files = ['price_model.pkl', 'scaler.pkl', 'data.pkl']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        st.error(f"❌ Missing required files: {', '.join(missing_files)}")
        st.info("Please ensure the following files are in your repository:")
        for file in required_files:
            st.write(f"- {file}")
        st.stop()
    
    try:
        with open('price_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('data.pkl', 'rb') as f:
            data = pickle.load(f)
        return model, scaler, data
    except Exception as e:
        st.error(f"❌ Error loading files: {str(e)}")
        st.stop()

model, scaler, data = load_artifacts()

# Price range mapping
PRICE_LABELS = {
    0: "Low Cost",
    1: "Medium",
    2: "High",
    3: "Very High"
}

# Sidebar Navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["📊 Market Analysis Dashboard", "🔬 Price Estimator & Competitor Analysis"]
)

# ==================== PAGE 1: MARKET ANALYSIS DASHBOARD ====================
if page == "📊 Market Analysis Dashboard":
    st.markdown('<p class="main-header">📊 Market Analysis Dashboard</p>', unsafe_allow_html=True)
    st.markdown("**Comprehensive insights into mobile phone market dynamics**")
    st.divider()
    
    # KPIs Section
    st.markdown('<p class="sub-header">📈 Key Market Metrics</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        median_ram = data['ram'].median()
        st.metric("Median RAM (MB)", f"{int(median_ram):,}")
    
    with col2:
        avg_battery = data['battery_power'].mean()
        st.metric("Avg Battery Power (mAh)", f"{int(avg_battery):,}")
    
    with col3:
        avg_px_density = data['px_density'].mean()
        st.metric("Avg Pixel Density", f"{avg_px_density:.2f}")
    
    with col4:
        avg_screen_area = data['screen_area'].mean()
        st.metric("Avg Screen Area (cm²)", f"{avg_screen_area:.1f}")
    
    st.divider()
    
    # Competitive Landscape Scatter Plot
    st.markdown('<p class="sub-header">🎯 Competitive Landscape</p>', unsafe_allow_html=True)
    st.markdown("**Market positioning: RAM vs Battery Power by Price Range**")
    
    fig_scatter = px.scatter(
        data,
        x='ram',
        y='battery_power',
        color='price_range',
        color_continuous_scale='Viridis',
        labels={
            'ram': 'RAM (MB)',
            'battery_power': 'Battery Power (mAh)',
            'price_range': 'Price Range'
        },
        title='Market Positioning Analysis',
        hover_data=['px_density', 'int_memory', 'total_cameras'],
        opacity=0.6
    )
    
    fig_scatter.update_layout(
        height=500,
        hovermode='closest',
        coloraxis_colorbar=dict(
            title="Price Range",
            tickvals=[0, 1, 2, 3],
            ticktext=["Low", "Medium", "High", "Very High"]
        )
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.divider()
    
    # Interactive Feature Explorer
    st.markdown('<p class="sub-header">🔍 Interactive Feature Explorer</p>', unsafe_allow_html=True)
    st.markdown("**Analyze how different features vary across price ranges**")
    
    # Feature selection
    available_features = [col for col in data.columns if col != 'price_range']
    selected_feature = st.selectbox(
        "Select a feature to explore:",
        available_features,
        index=available_features.index('px_density') if 'px_density' in available_features else 0
    )
    
    # Create box plot
    fig_box = px.box(
        data,
        x='price_range',
        y=selected_feature,
        color='price_range',
        labels={
            'price_range': 'Price Range',
            selected_feature: selected_feature.replace('_', ' ').title()
        },
        title=f'Distribution of {selected_feature.replace("_", " ").title()} Across Price Ranges',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig_box.update_xaxes(
        tickmode='array',
        tickvals=[0, 1, 2, 3],
        ticktext=["Low Cost", "Medium", "High", "Very High"]
    )
    
    fig_box.update_layout(height=500, showlegend=False)
    
    st.plotly_chart(fig_box, use_container_width=True)
    
    # Statistical summary
    with st.expander("📊 Statistical Summary"):
        summary = data.groupby('price_range')[selected_feature].describe()
        summary.index = ["Low Cost", "Medium", "High", "Very High"]
        st.dataframe(summary, use_container_width=True)

# ==================== PAGE 2: PRICE ESTIMATOR & COMPETITOR ANALYSIS ====================
else:
    st.markdown('<p class="main-header">🔬 Price Estimator & Competitor Analysis</p>', unsafe_allow_html=True)
    st.markdown("**Predict price range and discover similar competitors**")
    st.divider()
    
    # Sidebar Inputs
    st.sidebar.markdown("### 📱 Phone Specifications")
    
    # Get feature names from data
    feature_cols = [col for col in data.columns if col != 'price_range']
    
    # Create input widgets
    user_inputs = {}
    
    st.sidebar.markdown("**Battery & Performance**")
    user_inputs['battery_power'] = st.sidebar.slider("Battery Power (mAh)", 500, 2000, 1000)
    user_inputs['ram'] = st.sidebar.slider("RAM (MB)", 256, 4000, 2000)
    user_inputs['int_memory'] = st.sidebar.slider("Internal Memory (GB)", 2, 64, 32)
    user_inputs['clock_speed'] = st.sidebar.slider("Clock Speed (GHz)", 0.5, 3.0, 1.5, 0.1)
    
    st.sidebar.markdown("**Display**")
    user_inputs['sc_h'] = st.sidebar.slider("Screen Height (cm)", 5, 20, 12)
    user_inputs['sc_w'] = st.sidebar.slider("Screen Width (cm)", 0, 18, 6)
    user_inputs['px_height'] = st.sidebar.slider("Pixel Height", 0, 2000, 1000)
    user_inputs['px_width'] = st.sidebar.slider("Pixel Width", 500, 2000, 1200)
    
    st.sidebar.markdown("**Cameras**")
    user_inputs['fc'] = st.sidebar.slider("Front Camera (MP)", 0, 20, 5)
    user_inputs['pc'] = st.sidebar.slider("Primary Camera (MP)", 0, 20, 12)
    
    st.sidebar.markdown("**Features**")
    user_inputs['blue'] = st.sidebar.selectbox("Bluetooth", [0, 1], index=1)
    user_inputs['dual_sim'] = st.sidebar.selectbox("Dual SIM", [0, 1], index=1)
    user_inputs['four_g'] = st.sidebar.selectbox("4G", [0, 1], index=1)
    user_inputs['three_g'] = st.sidebar.selectbox("3G", [0, 1], index=1)
    user_inputs['touch_screen'] = st.sidebar.selectbox("Touch Screen", [0, 1], index=1)
    user_inputs['wifi'] = st.sidebar.selectbox("WiFi", [0, 1], index=1)
    
    st.sidebar.markdown("**Other**")
    user_inputs['m_dep'] = st.sidebar.slider("Mobile Depth (cm)", 0.1, 1.0, 0.5, 0.1)
    user_inputs['mobile_wt'] = st.sidebar.slider("Mobile Weight (g)", 80, 200, 140)
    user_inputs['n_cores'] = st.sidebar.slider("Number of Cores", 1, 8, 4)
    user_inputs['talk_time'] = st.sidebar.slider("Talk Time (hrs)", 2, 20, 10)
    
    # Engineer features for user input
    user_inputs['screen_area'] = user_inputs['sc_h'] * user_inputs['sc_w']
    denominator = user_inputs['sc_h'] + user_inputs['sc_w']
    user_inputs['px_density'] = (user_inputs['px_height'] + user_inputs['px_width']) / denominator if denominator > 0 else 0
    user_inputs['total_cameras'] = user_inputs['fc'] + user_inputs['pc']
    
    # Predict Button
    predict_button = st.sidebar.button("🎯 Predict Price Range", type="primary", use_container_width=True)
    
    # Main Content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📋 Input Summary")
        summary_data = {
            "Specification": [
                "Battery Power", "RAM", "Internal Memory", "Screen Area",
                "Pixel Density", "Total Cameras", "4G Enabled"
            ],
            "Value": [
                f"{user_inputs['battery_power']} mAh",
                f"{user_inputs['ram']} MB",
                f"{user_inputs['int_memory']} GB",
                f"{user_inputs['screen_area']:.1f} cm²",
                f"{user_inputs['px_density']:.2f}",
                f"{user_inputs['total_cameras']} MP",
                "Yes" if user_inputs['four_g'] == 1 else "No"
            ]
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
    
    if predict_button:
        try:
            # Prepare input data
            input_df = pd.DataFrame([user_inputs])
            input_df = input_df[feature_cols]  # Ensure correct order
            
            # Scale input
            input_scaled = scaler.transform(input_df)
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]
            
            with col2:
                st.markdown("### 🎯 Prediction Result")
                st.markdown(f"### Predicted Price Range: **{PRICE_LABELS[prediction]}**")
                
                # Probability distribution
                prob_df = pd.DataFrame({
                    'Price Range': [PRICE_LABELS[i] for i in range(4)],
                    'Probability': probabilities
                })
                
                fig_prob = px.bar(
                    prob_df,
                    x='Price Range',
                    y='Probability',
                    color='Probability',
                    color_continuous_scale='Blues',
                    title='Confidence Distribution'
                )
                fig_prob.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_prob, use_container_width=True)
            
            st.divider()
            
            # SHAP Explanation
            st.markdown("### 🧠 Features Driving this Price Prediction")
            st.markdown("**SHAP analysis showing feature contributions**")
            
            with st.spinner("Generating explainability analysis..."):
                # Create SHAP explainer
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_scaled)
                
                # Get SHAP values for the predicted class
                if isinstance(shap_values, list):
                    shap_vals_class = shap_values[prediction][0]
                    expected_val = explainer.expected_value[prediction]
                else:
                    shap_vals_class = shap_values[0, :, prediction]
                    expected_val = explainer.expected_value[prediction]
                
                # Create waterfall plot (better for Streamlit than force plot)
                shap_explanation = shap.Explanation(
                    values=shap_vals_class,
                    base_values=expected_val,
                    data=input_df.iloc[0].values,
                    feature_names=input_df.columns.tolist()
                )
                
                # Display waterfall plot
                fig_shap, ax = plt.subplots(figsize=(10, 8))
                shap.plots.waterfall(shap_explanation, show=False)
                st.pyplot(fig_shap, bbox_inches='tight')
                plt.close()
            
            st.divider()
            
            # Competitor Analysis
            st.markdown("### 🏆 Closest Competitors in this Price Segment")
            st.markdown(f"**Finding similar devices in the {PRICE_LABELS[prediction]} category**")
            
            # Filter data by predicted price range
            same_price_data = data[data['price_range'] == prediction].copy()
            
            if len(same_price_data) > 0:
                # Prepare same price data (drop price_range for distance calculation)
                X_competitors = same_price_data.drop('price_range', axis=1)
                X_competitors_scaled = scaler.transform(X_competitors)
                
                # Calculate Euclidean distances
                distances = euclidean_distances(input_scaled, X_competitors_scaled)[0]
                
                # Get top 3 closest competitors
                top_indices = np.argsort(distances)[:3]
                
                competitors = same_price_data.iloc[top_indices].copy()
                competitors['Similarity Score'] = 100 - (distances[top_indices] / distances.max() * 100)
                
                # Display competitors
                display_cols = ['ram', 'battery_power', 'int_memory', 'px_density', 
                              'total_cameras', 'screen_area', 'Similarity Score']
                
                competitors_display = competitors[display_cols].copy()
                competitors_display.columns = ['RAM (MB)', 'Battery (mAh)', 'Memory (GB)', 
                                              'Pixel Density', 'Cameras (MP)', 'Screen Area (cm²)', 
                                              'Similarity (%)']
                
                competitors_display = competitors_display.round(2)
                
                st.dataframe(competitors_display, use_container_width=True, hide_index=True)
                
                # Competitive positioning chart
                st.markdown("#### 🔍 Competitive Positioning")
                
                comp_plot_data = pd.concat([
                    pd.DataFrame({
                        'RAM': [user_inputs['ram']],
                        'Battery Power': [user_inputs['battery_power']],
                        'Type': ['Your Device']
                    }),
                    pd.DataFrame({
                        'RAM': competitors['ram'].values,
                        'Battery Power': competitors['battery_power'].values,
                        'Type': ['Competitor'] * len(competitors)
                    })
                ])
                
                fig_comp = px.scatter(
                    comp_plot_data,
                    x='RAM',
                    y='Battery Power',
                    color='Type',
                    size=[500] + [300]*len(competitors),
                    color_discrete_map={'Your Device': '#ff7f0e', 'Competitor': '#1f77b4'},
                    title='Your Device vs Closest Competitors'
                )
                fig_comp.update_layout(height=400)
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.warning("No competitors found in this price range.")
        
        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {str(e)}")
            st.info("Please check that your model and scaler are compatible with the input data.")
    
    else:
        with col2:
            st.info("👈 Configure phone specifications and click 'Predict Price Range' to see results")