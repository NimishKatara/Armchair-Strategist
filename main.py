import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')



st.set_page_config(
    page_title="F1 Tire Degradation Predictor Claude 5",
    page_icon="🏎️",
    layout="wide"
)


st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #DC143C 0%, #000000 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    .future-prediction {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 20px 0;
    }
    .stButton > button {
        background-color: #DC143C;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .metric-card {
        background: #f0f0f0;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #DC143C;
        margin: 10px 0;
    }
    .linkedin-container {
        display: flex;
        align-items: center;
        padding: 10px;
        margin: 10px 0;
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    .linkedin-link {
        text-decoration: none;
        color: #0077b5;
        font-weight: bold;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .linkedin-link:hover {
        color: #005885;
    }
    
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h1>🏎️ F1 Tire Degradation Predictor</h1><p>Predict remaining tire grip and optimal pit stop windows</p></div>', unsafe_allow_html=True)

class F1TirePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_trained = False
        self.combined_data = None
        
        self.compound_factors = {
            'SOFT': 0.8,
            'MEDIUM': 0.9,
            'HARD': 1.0,
            'INTERMEDIATE': 0.6,
            'WET': 0.4
        }
        
    def create_sample_data(self, n_samples=200):
        """Create realistic F1 sample data"""
        np.random.seed(42)
        
        drivers = ['HAM', 'VER', 'LEC', 'RUS', 'NOR', 'PER', 'SAI', 'OCO', 'ALO', 'BOT']
        teams = ['Mercedes', 'Red Bull', 'Ferrari', 'McLaren', 'Alpine', 'Aston Martin']
        compounds = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']
        tracks = ['Monaco', 'Silverstone', 'Spa', 'Monza', 'Bahrain', 'Austria']
        
        data = pd.DataFrame({
            'driver': np.random.choice(drivers, n_samples),
            'team': np.random.choice(teams, n_samples),
            'compound': np.random.choice(compounds, n_samples),
            'stint_length': np.random.randint(5, 35, n_samples),
            'tyre_age': np.random.randint(1, 40, n_samples),
            'track_temp': np.random.uniform(25, 50, n_samples),
            'air_temp': np.random.uniform(15, 35, n_samples),
            'lap_time': np.random.uniform(75, 95, n_samples),
            'track_name': np.random.choice(tracks, n_samples),
            'degradation_rate': np.random.uniform(0.01, 0.3, n_samples),
        })
        
        for idx, row in data.iterrows():
            compound = row['compound']
            tire_age = row['tyre_age']
            track_temp = row['track_temp']
            
           
            if compound == 'SOFT':
                compound_grip_loss = min(tire_age * 2.5, 30) + (max(0, tire_age - 12) * 1.5)
            elif compound == 'MEDIUM':
                compound_grip_loss = tire_age * 1.8
            elif compound == 'HARD':
                compound_grip_loss = tire_age * 1.2
            elif compound == 'INTERMEDIATE':
                performance_loss_pct = abs(track_temp - 30) / 30
                compound_grip_loss = tire_age * 2.0 + performance_loss_pct * 10
            elif compound == 'WET':
                performance_loss_pct = abs(track_temp - 25) / 25
                compound_grip_loss = tire_age * 3.0 + performance_loss_pct * 20
            else:
                compound_grip_loss = tire_age * 1.5
            
            # Temperature factor
            temp_factor = abs(track_temp - 35) * 0.3
            
            # Calculate remaining grip
            remaining_grip = 100 - compound_grip_loss - temp_factor + np.random.normal(0, 2)
            data.at[idx, 'remaining_grip'] = max(10, min(100, remaining_grip))
        
        return data
    
    def combine_csv_files(self, uploaded_files):
        """Combine multiple CSV files into one dataset"""
        combined_data = []
        
        for uploaded_file in uploaded_files:
            try:
                df = pd.read_csv(uploaded_file)
                df['source_file'] = uploaded_file.name
                combined_data.append(df)
            except Exception as e:
                st.error(f"Error reading {uploaded_file.name}: {e}")
                return None
        
        if combined_data:
            self.combined_data = pd.concat(combined_data, ignore_index=True)
            return self.combined_data
        return None
    
    def prepare_features(self, data):
        """Prepare features for ML model"""
        df = data.copy()
        
        categorical_cols = ['driver', 'team', 'compound', 'track_name']
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col])
                else:
                    unique_values = set(df[col].unique())
                    known_values = set(self.label_encoders[col].classes_)
                    new_values = unique_values - known_values
                    
                    if new_values:
                        # Add new classes to existing encoder
                        all_classes = list(self.label_encoders[col].classes_) + list(new_values)
                        self.label_encoders[col].classes_ = np.array(all_classes)
                    
                    df[col] = df[col].map(lambda x: self.label_encoders[col].transform([x])[0] 
                                         if x in self.label_encoders[col].classes_ else -1)
        
        # Feature engineering
        df['temp_diff'] = df['track_temp'] - df['air_temp']
        df['age_intensity'] = df['tyre_age'] / df['stint_length']
        
        # Select numeric features
        feature_cols = ['stint_length', 'tyre_age', 'track_temp', 'air_temp', 
                       'lap_time', 'degradation_rate', 'temp_diff', 'age_intensity']
        
        if categorical_cols[0] in df.columns: 
            feature_cols.extend(categorical_cols)
        
        return df[feature_cols]
    
    def train(self, data):
        """Train the model"""
        X = self.prepare_features(data)
        y = data['remaining_grip']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.is_trained = True
        return {'rmse': np.sqrt(mse), 'r2': r2}
    
    def predict_single_driver(self, driver_data):
        """Make prediction for a single driver"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        X = self.prepare_features(driver_data)
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        
        return np.clip(prediction, 0, 100)
    
    def calculate_driver_performance(self, driver_data, predicted_grip):
        """Calculate comprehensive driver performance metrics"""
        row = driver_data.iloc[0]
        
        laps_remaining = self.estimate_remaining_laps(predicted_grip, row['degradation_rate'])
        pit_recommendation = self.get_pit_recommendation(predicted_grip, laps_remaining)
        performance_score = self.calculate_performance_score(row, predicted_grip)
        
        track_insights = self.get_track_insights(row['track_name'], row['compound'])
        
        return {
            'predicted_grip': predicted_grip,
            'laps_remaining': laps_remaining,
            'pit_recommendation': pit_recommendation,
            'performance_score': performance_score,
            'track_insights': track_insights,
            'compound_efficiency': self.get_compound_efficiency(row['compound'], row['tyre_age']),
            'temperature_impact': self.get_temperature_impact(row['track_temp'], row['air_temp'])
        }
    
    def estimate_remaining_laps(self, grip, degradation_rate):
        """Estimate how many more laps the tire can handle - Fixed overflow error"""
        if degradation_rate <= 0 or degradation_rate > 1:
            degradation_rate = 0.05 
        
        if grip <= 20:
            return 0
        elif grip <= 30:
            denominator = degradation_rate * 100
            if denominator <= 0:
                return 0
            laps = (grip - 20) / denominator
            return max(0, min(50, int(laps))) 
        elif grip <= 50:
            denominator = degradation_rate * 80
            if denominator <= 0:
                return 5
            laps = (grip - 30) / denominator
            return max(0, min(50, int(laps)))
        else:
            denominator = degradation_rate * 60
            if denominator <= 0:
                return 10
            laps = (grip - 50) / denominator
            return max(0, min(50, int(laps)))
    
    def get_pit_recommendation(self, grip, laps_remaining):
        """Get detailed pit recommendation"""
        if grip <= 30:
            return {
                'action': '🔴 PIT NOW',
                'urgency': 'CRITICAL',
                'reason': 'Tire grip critically low',
                'strategy': 'Immediate pit stop required'
            }
        elif grip <= 50:
            return {
                'action': '🟡 PIT SOON',
                'urgency': 'HIGH',
                'reason': f'~{laps_remaining} laps remaining',
                'strategy': 'Plan pit stop within next 3-5 laps'
            }
        elif grip <= 70:
            return {
                'action': '🟢 MONITOR',
                'urgency': 'MEDIUM',
                'reason': f'~{laps_remaining} laps remaining',
                'strategy': 'Continue monitoring, pit in 8-12 laps'
            }
        else:
            return {
                'action': '✅ CONTINUE',
                'urgency': 'LOW',
                'reason': f'~{laps_remaining} laps remaining',
                'strategy': 'Tires in good condition'
            }
    
    def calculate_performance_score(self, row, predicted_grip):
        """Calculate overall performance score (0-100)"""
    
        grip_score = predicted_grip
        
        compound_multiplier = self.compound_factors.get(row['compound'], 1.0)
        
        age_penalty = min(row['tyre_age'] * 0.5, 20)
        
       
        temp_efficiency = 100 - abs(row['track_temp'] - 35) * 0.5
        
        final_score = (grip_score * compound_multiplier - age_penalty + temp_efficiency * 0.1)
        return np.clip(final_score, 0, 100)
    
    def get_track_insights(self, track_name, compound):
        """Get track-specific insights"""
        insights = {
            'Monaco': f'{compound} compound on Monaco: High precision required, low degradation',
            'Silverstone': f'{compound} compound on Silverstone: Fast corners, medium degradation',
            'Spa': f'{compound} compound on Spa: High speed track, watch for graining',
            'Monza': f'{compound} compound on Monza: Low downforce, tire temperature critical',
            'Bahrain': f'{compound} compound on Bahrain: Hot conditions, watch for overheating',
            'Austria': f'{compound} compound on Austria: Short lap, multiple stint strategy'
        }
        return insights.get(track_name, f'{compound} compound selected for this track')
    
    def get_compound_efficiency(self, compound, age):
        """Calculate compound efficiency based on age with enhanced compound logic"""
        if compound == 'SOFT':
            return max(100 - age * 3, 0)
        elif compound == 'MEDIUM':
            return max(100 - age * 2, 0)
        elif compound == 'HARD':
            return max(100 - age * 1.5, 0)
        elif compound == 'INTERMEDIATE':
            return max(100 - age * 2.5, 0)
        elif compound == 'WET':
            return max(100 - age * 3.5, 0)
        else:
            return max(100 - age * 2, 0)
    
    def get_temperature_impact(self, track_temp, air_temp):
        """Analyze temperature impact on performance"""
        optimal_track_temp = 35
        temp_diff = abs(track_temp - optimal_track_temp)
        
        if temp_diff < 5:
            return "Optimal temperature conditions"
        elif temp_diff < 10:
            return "Slightly suboptimal temperature"
        else:
            return "Challenging temperature conditions"

@st.cache_resource
def load_predictor():
    predictor = F1TirePredictor()
    return predictor

predictor = load_predictor()

st.sidebar.title("🏁 Configuration")

st.sidebar.markdown("""
<div class="linkedin-container">
    <a href="https://www.linkedin.com/in/nimish-katara-460622283" target="_blank" class="linkedin-link">
        <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="20" height="20">
        Nimish Katara
    </a>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

prediction_mode = st.sidebar.radio("Select Prediction Mode:", ["Basic Analysis", "Enhanced Race Strategy"])

st.sidebar.markdown("---")


data_option = st.sidebar.radio("Data Source:", ["Generate Sample Data", "Upload CSV Files"])

if data_option == "Generate Sample Data":
    n_samples = st.sidebar.slider("Number of samples:", 50, 500, 200)
    data = predictor.create_sample_data(n_samples)
    st.sidebar.success(f"Generated {len(data)} sample records")
else:
    uploaded_files = st.sidebar.file_uploader(
        "Upload CSV files", 
        type=['csv'], 
        accept_multiple_files=True,
        help="Upload multiple CSV files from different races (e.g., Monaco 2023, Monaco 2022, etc.)"
    )
    
    if uploaded_files:
        data = predictor.combine_csv_files(uploaded_files)
        if data is not None:
            st.sidebar.success(f"Loaded {len(uploaded_files)} files with {len(data)} total records")
            
        
            with st.sidebar.expander("File Details"):
                file_counts = data['source_file'].value_counts()
                for file, count in file_counts.items():
                    st.sidebar.write(f"📄 {file}: {count} records")
        else:
            st.sidebar.error("Error loading CSV files")
            st.stop()
    else:
        st.sidebar.warning("Please upload CSV files or use sample data")
        st.stop()


if prediction_mode == "Basic Analysis":
    # Main content for Basic Analysis
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Data Overview")
        
        # Display data info
        st.info(f"Dataset contains {len(data)} records with {len(data.columns)} features")
        
        # Show sample data
        with st.expander("View Sample Data"):
            st.dataframe(data.head(10))
        
        # Train model button
        if st.button("🚀 Train Model", use_container_width=True):
            with st.spinner("Training model on combined dataset..."):
                results = predictor.train(data)
                st.success("Model trained successfully!")
                
                # Display training results
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    st.metric("RMSE", f"{results['rmse']:.2f}")
                with col_r2:
                    st.metric("R² Score", f"{results['r2']:.3f}")
    
    with col2:
        st.subheader("🔧 Model Status")
        
        if predictor.is_trained:
            st.success("✅ Model is ready")
        else:
            st.warning("⚠️ Model not trained")
        
        # Feature importance
        if predictor.is_trained:
            st.subheader("📈 Feature Importance")
            feature_names = list(predictor.prepare_features(data.head(1)).columns)
            importance = predictor.model.feature_importances_[:len(feature_names)]
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h')
            st.plotly_chart(fig, use_container_width=True)
    
    # Driver Analysis Section
    st.markdown("---")
    st.subheader("🎯 Individual Driver Analysis")
    
    if predictor.is_trained:
        # Driver selection
        available_drivers = sorted(data['driver'].unique())
        selected_driver = st.selectbox("Select Driver for Analysis:", available_drivers)
        
        if selected_driver:
            # Get driver's latest data
            driver_data = data[data['driver'] == selected_driver].tail(1)
            
            if not driver_data.empty:
                col_pred1, col_pred2 = st.columns([1, 2])
                
                with col_pred1:
                    if st.button("🔍 Analyze Driver Performance", use_container_width=True):
                        with st.spinner(f"Analyzing {selected_driver}'s performance..."):
                            # Get prediction and analysis
                            predicted_grip = predictor.predict_single_driver(driver_data)
                            analysis = predictor.calculate_driver_performance(driver_data, predicted_grip)
                            
                            # Display results
                            st.subheader(f"Analysis for {selected_driver}")
                            
                            # Key metrics
                            col_m1, col_m2, col_m3 = st.columns(3)
                            with col_m1:
                                st.metric("Tire Grip", f"{analysis['predicted_grip']:.1f}%")
                            with col_m2:
                                st.metric("Est. Laps Left", f"{analysis['laps_remaining']}")
                            with col_m3:
                                st.metric("Performance Score", f"{analysis['performance_score']:.1f}")
                            
                            # Detailed recommendations
                            st.subheader("🏁 Strategy Recommendations")
                            pit_rec = analysis['pit_recommendation']
                            st.markdown(f"**Action:** {pit_rec['action']}")
                            st.markdown(f"**Urgency:** {pit_rec['urgency']}")
                            st.markdown(f"**Reason:** {pit_rec['reason']}")
                            st.markdown(f"**Strategy:** {pit_rec['strategy']}")
                            
                            # Additional insights
                            st.subheader("📋 Detailed Insights")
                            st.markdown(f"**Track Insight:** {analysis['track_insights']}")
                            st.markdown(f"**Compound Efficiency:** {analysis['compound_efficiency']:.1f}%")
                            st.markdown(f"**Temperature Impact:** {analysis['temperature_impact']}")
                
                with col_pred2:
                    # Driver details table
                    st.subheader("Driver Current Status")
                    display_data = driver_data[['driver', 'team', 'compound', 'tyre_age', 'track_temp', 'degradation_rate']].T
                    display_data.columns = ['Value']
                    st.dataframe(display_data)
                    
                    # Create performance heatmap for all drivers
                    st.subheader("🔥 Driver Performance Heatmap")
                    
                    # Sample data for heatmap (last 10 drivers)
                    sample_for_heatmap = data.groupby('driver').tail(1).head(10)
                    
                    if len(sample_for_heatmap) > 0:
                        # Predict for all drivers in sample
                        heatmap_data = []
                        for _, row in sample_for_heatmap.iterrows():
                            try:
                                single_row_df = pd.DataFrame([row])
                                pred_grip = predictor.predict_single_driver(single_row_df)
                                analysis = predictor.calculate_driver_performance(single_row_df, pred_grip)
                                
                                heatmap_data.append({
                                    'Driver': row['driver'],
                                    'Tire Grip': pred_grip,
                                    'Performance Score': analysis['performance_score'],
                                    'Compound Efficiency': analysis['compound_efficiency'],
                                    'Estimated Laps': analysis['laps_remaining']
                                })
                            except Exception as e:
                                continue
                        
                        if heatmap_data:
                            heatmap_df = pd.DataFrame(heatmap_data)
                            
                            # Create heatmap
                            fig = px.imshow(
                                heatmap_df.set_index('Driver').T,
                                aspect='auto',
                                color_continuous_scale='RdYlGn',
                                title="Driver Performance Metrics"
                            )
                            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("Please train the model first to perform driver analysis")

elif prediction_mode == "Enhanced Race Strategy":
    # Train the model first if not trained
    if not predictor.is_trained:
        with st.spinner("Training model for enhanced predictions..."):
            results = predictor.train(data)
            st.success("Model trained successfully for enhanced predictions!")
    
    # Enhanced Race Strategy Interface
    st.markdown('<div class="enhanced-strategy-container">', unsafe_allow_html=True)
    st.subheader("🎯 Enhanced Race Strategy Configuration")
    
    col_strategy1, col_strategy2 = st.columns(2)
    
    with col_strategy1:
        # Track selection
        available_tracks = sorted(data['track_name'].unique())
        selected_future_track = st.selectbox("Select Track:", available_tracks)
        
        # Year selection
        future_race_year = st.slider("Target Race Year:", 2025, 2030, 2025)
        
        # Weather conditions
        st.markdown("### 🌤️ Expected Conditions")
        expected_track_temp = st.slider("Track Temperature (°C):", 20, 60, 35)
        expected_air_temp = st.slider("Air Temperature (°C):", 15, 45, 25)
    
    with col_strategy2:
        # Race parameters
        st.markdown("### 🏁 Race Parameters")
        total_race_laps = st.number_input("Total Race Laps:", min_value=50, max_value=70, value=58)
        
        st.markdown("### 📊 Current Dataset")
        st.info(f"Using {len(data)} records from {len(data['track_name'].unique())} tracks")
    
    # Generate predictions button
    if st.button("🚀 Generate Enhanced Predictions", use_container_width=True, type="primary"):
        with st.spinner(f"Generating enhanced predictions for {selected_future_track} {future_race_year}..."):
            
            # Get historical data for the track
            track_history = data[data['track_name'] == selected_future_track]
            
            if not track_history.empty:
                # Enhanced prediction logic
                enhanced_predictions = []
                
                # Get unique drivers from historical data
                historical_drivers = track_history['driver'].unique()
                
                for driver in historical_drivers:
                    driver_history = track_history[track_history['driver'] == driver]
                    
                    # Calculate driver's average performance at this track
                    avg_performance = {
                        'avg_lap_time': driver_history['lap_time'].mean(),
                        'avg_degradation': abs(driver_history['degradation_rate'].mean()),
                        'best_compound': driver_history['compound'].mode().iloc[0] if not driver_history.empty else 'MEDIUM',
                        'team': driver_history['team'].mode().iloc[0] if not driver_history.empty else 'Unknown'
                    }
                    
                    # Test different strategies
                    strategies = []
                    compounds = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']
                    
                    for compound in compounds:
                        for pit_lap in range(15, min(total_race_laps-10, 45), 5):
                            # First stint
                            first_stint = pd.DataFrame({
                                'driver': [driver],
                                'team': [avg_performance['team']],
                                'compound': [compound],
                                'stint_length': [pit_lap],
                                'tyre_age': [pit_lap],
                                'track_temp': [expected_track_temp],
                                'air_temp': [expected_air_temp],
                                'lap_time': [avg_performance['avg_lap_time']],
                                'track_name': [selected_future_track],
                                'degradation_rate': [avg_performance['avg_degradation']]
                            })
                            
                            try:
                                first_stint_grip = predictor.predict_single_driver(first_stint)
                                first_analysis = predictor.calculate_driver_performance(first_stint, first_stint_grip)
                                
                                # Second stint (remaining laps)
                                remaining_laps = total_race_laps - pit_lap
                                second_compound = 'HARD' if compound == 'SOFT' else 'MEDIUM'
                                
                                second_stint = pd.DataFrame({
                                    'driver': [driver],
                                    'team': [avg_performance['team']],
                                    'compound': [second_compound],
                                    'stint_length': [remaining_laps],
                                    'tyre_age': [remaining_laps],
                                    'track_temp': [expected_track_temp],
                                    'air_temp': [expected_air_temp],
                                    'lap_time': [avg_performance['avg_lap_time']],
                                    'track_name': [selected_future_track],
                                    'degradation_rate': [avg_performance['avg_degradation']]
                                })
                                
                                second_stint_grip = predictor.predict_single_driver(second_stint)
                                second_analysis = predictor.calculate_driver_performance(second_stint, second_stint_grip)
                                
                                # Calculate overall strategy score
                                strategy_score = (first_analysis['performance_score'] + second_analysis['performance_score']) / 2
                                
                                # Calculate pit window efficiency
                                pit_efficiency = 100 - abs(pit_lap - 25) * 2  # Optimal pit around lap 25
                                
                                strategies.append({
                                    'driver': driver,
                                    'team': avg_performance['team'],
                                    'strategy': f"{compound} → {second_compound}",
                                    'pit_lap': pit_lap,
                                    'first_stint_grip': first_stint_grip,
                                    'second_stint_grip': second_stint_grip,
                                    'strategy_score': strategy_score,
                                    'pit_efficiency': max(0, pit_efficiency),
                                    'total_score': (strategy_score + pit_efficiency) / 2
                                })
                            
                            except Exception as e:
                                continue
                    
                    # Get best strategy for this driver
                    if strategies:
                        best_strategy = max(strategies, key=lambda x: x['total_score'])
                        enhanced_predictions.append(best_strategy)
                
                # Display enhanced results
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="future-prediction"><h2>🏁 {selected_future_track} {future_race_year} - Enhanced Analysis Results</h2></div>', unsafe_allow_html=True)
                
                if enhanced_predictions:
                    # Sort by total score
                    enhanced_predictions.sort(key=lambda x: x['total_score'], reverse=True)
                    
                    # Top 3 strategies
                    st.subheader("🏆 Top 3 Recommended Strategies")
                    
                    for i, strategy in enumerate(enhanced_predictions[:3]):
                        with st.expander(f"#{i+1} - {strategy['driver']} ({strategy['team']}) - Score: {strategy['total_score']:.1f}"):
                            col_strat1, col_strat2, col_strat3 = st.columns(3)
                            
                            with col_strat1:
                                st.metric("Strategy", strategy['strategy'])
                                st.metric("Pit Lap", strategy['pit_lap'])
                            
                            with col_strat2:
                                st.metric("1st Stint Grip", f"{strategy['first_stint_grip']:.1f}%")
                                st.metric("2nd Stint Grip", f"{strategy['second_stint_grip']:.1f}%")
                            
                            with col_strat3:
                                st.metric("Strategy Score", f"{strategy['strategy_score']:.1f}")
                                st.metric("Pit Efficiency", f"{strategy['pit_efficiency']:.1f}%")
                    
                    # Detailed comparison table
                    st.subheader("📊 Complete Strategy Analysis")
                    
                    comparison_df = pd.DataFrame(enhanced_predictions)
                    display_columns = ['driver', 'team', 'strategy', 'pit_lap', 'first_stint_grip', 
                                     'second_stint_grip', 'strategy_score', 'total_score']
                    
                    # Format the dataframe for better display
                    formatted_df = comparison_df[display_columns].copy()
                    formatted_df['first_stint_grip'] = formatted_df['first_stint_grip'].round(1)
                    formatted_df['second_stint_grip'] = formatted_df['second_stint_grip'].round(1)
                    formatted_df['strategy_score'] = formatted_df['strategy_score'].round(1)
                    formatted_df['total_score'] = formatted_df['total_score'].round(1)
                    
                    st.dataframe(formatted_df, use_container_width=True)
                    
                    # Strategy visualization
                    st.subheader("📈 Strategy Performance Visualization")
                    
                    # Create scatter plot
                    fig = px.scatter(
                        comparison_df,
                        x='pit_lap',
                        y='total_score',
                        color='driver',
                        size='strategy_score',
                        hover_data=['strategy', 'first_stint_grip', 'second_stint_grip'],
                        title=f"Strategy Analysis for {selected_future_track} {future_race_year}"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Race timeline visualization
                    st.subheader("🏁 Optimal Race Timeline")
                    
                    best_overall = enhanced_predictions[0]
                    
                    # Create timeline data
                    timeline_data = []
                    
                    # First stint
                    for lap in range(1, best_overall['pit_lap'] + 1):
                        grip_degradation = best_overall['first_stint_grip'] - (lap * 2)  # Simplified degradation
                        timeline_data.append({
                            'Lap': lap,
                            'Tire_Grip': max(20, grip_degradation),
                            'Stint': 'First Stint',
                            'Compound': best_overall['strategy'].split(' → ')[0]
                        })
                    
                    # Pit stop
                    timeline_data.append({
                        'Lap': best_overall['pit_lap'],
                        'Tire_Grip': 100,
                        'Stint': 'Pit Stop',
                        'Compound': 'PIT'
                    })
                    
                    # Second stint
                    for lap in range(best_overall['pit_lap'] + 1, total_race_laps + 1):
                        laps_on_tire = lap - best_overall['pit_lap']
                        grip_degradation = best_overall['second_stint_grip'] - (laps_on_tire * 1.5)
                        timeline_data.append({
                            'Lap': lap,
                            'Tire_Grip': max(20, grip_degradation),
                            'Stint': 'Second Stint',
                            'Compound': best_overall['strategy'].split(' → ')[1]
                        })
                    
                    timeline_df = pd.DataFrame(timeline_data)
                    
                    # Create timeline chart
                    fig_timeline = px.line(
                        timeline_df,
                        x='Lap',
                        y='Tire_Grip',
                        color='Stint',
                        title=f"Optimal Race Strategy Timeline - {best_overall['driver']} ({best_overall['strategy']})",
                        markers=True
                    )
                    fig_timeline.add_hline(y=30, line_dash="dash", line_color="red", 
                                         annotation_text="Critical Grip Level")
                    fig_timeline.update_layout(height=400)
                    st.plotly_chart(fig_timeline, use_container_width=True)
                    
                    # Weather impact analysis
                    st.subheader("🌡️ Temperature Impact Analysis")
                    
                    col_temp1, col_temp2 = st.columns(2)
                    
                    with col_temp1:
                        st.markdown("**Track Temperature Impact:**")
                        temp_impact = "Optimal" if 30 <= expected_track_temp <= 40 else "Challenging"
                        st.info(f"Expected: {expected_track_temp}°C - {temp_impact} conditions")
                        
                        if expected_track_temp > 45:
                            st.warning("⚠️ High track temperature may cause tire overheating")
                        elif expected_track_temp < 25:
                            st.warning("⚠️ Low track temperature may affect tire warm-up")
                    
                    with col_temp2:
                        st.markdown("**Strategy Adjustments:**")
                        if expected_track_temp > 40:
                            st.markdown("• Consider harder compounds")
                            st.markdown("• Earlier pit stops may be needed")
                            st.markdown("• Monitor tire pressure closely")
                        else:
                            st.markdown("• Standard compound selection optimal")
                            st.markdown("• Normal pit windows applicable")
                            st.markdown("• Focus on tire warming strategies")

                else:
                    st.error("No valid strategies could be generated for the selected parameters")
            
            else:
                st.error(f"No historical data available for {selected_future_track}")

# Data visualization section (only for Basic Analysis)
if prediction_mode == "Basic Analysis" and predictor.is_trained:
    st.markdown("---")
    st.subheader("📈 Data Visualizations")
    
    # Tire compound analysis
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        st.subheader("Compound Performance")
        compound_performance = data.groupby('compound').agg({
            'remaining_grip': 'mean',
            'tyre_age': 'mean',
            'lap_time': 'mean'
        }).reset_index()
        
        fig = px.bar(compound_performance, x='compound', y='remaining_grip', 
                    title="Average Remaining Grip by Compound")
        st.plotly_chart(fig, use_container_width=True)
    
    with col_viz2:
        st.subheader("Temperature vs Performance")
        fig = px.scatter(data, x='track_temp', y='remaining_grip', color='compound',
                        title="Track Temperature vs Tire Grip")
        st.plotly_chart(fig, use_container_width=True)
    
    # Track analysis
    st.subheader("🏁 Track Performance Analysis")
    
    track_stats = data.groupby('track_name').agg({
        'remaining_grip': ['mean', 'std'],
        'lap_time': 'mean',
        'degradation_rate': 'mean'
    }).round(2)
    
    track_stats.columns = ['Avg Grip', 'Grip Std', 'Avg Lap Time', 'Avg Degradation']
    st.dataframe(track_stats, use_container_width=True)

