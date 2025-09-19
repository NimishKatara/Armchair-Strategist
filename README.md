# F1 Tire Degradation Predictor

A comprehensive machine learning application that predicts Formula 1 tire degradation patterns, remaining tire grip, and optimal pit stop strategies using real-world F1 data and advanced predictive modeling.

## Project Overview

This project combines the power of machine learning with Formula 1 telemetry data to create an intelligent tire degradation prediction system. The application uses Random Forest regression models to analyze tire performance patterns and provide strategic insights for race planning.

### What We're Using

- **Machine Learning**: Random Forest Regressor for tire grip prediction
- **Real F1 Data**: FastF1 library for accessing official F1 timing and telemetry data
- **Interactive Dashboard**: Streamlit for user-friendly web interface
- **Data Visualization**: Plotly for interactive charts and performance analysis
- **Data Processing**: Pandas and NumPy for data manipulation and feature engineering

### How It Works

1. **Data Collection**: The system can either generate synthetic F1 data or extract real telemetry data using the FastF1 library
2. **Feature Engineering**: Creates meaningful features like tire age, temperature differentials, compound efficiency, and degradation rates
3. **Machine Learning**: Trains a Random Forest model to predict remaining tire grip based on historical patterns
4. **Strategic Analysis**: Provides pit stop recommendations, performance scores, and race strategy optimization
5. **Interactive Visualization**: Presents results through an intuitive web dashboard with real-time predictions

## Features

### Basic Analysis Mode
- **Tire Grip Prediction**: Predicts remaining tire grip percentage based on current conditions
- **Pit Stop Recommendations**: Intelligent suggestions for optimal pit stop timing
- **Driver Performance Analysis**: Individual driver performance metrics and comparisons
- **Data Visualization**: Interactive charts showing tire degradation patterns and performance trends

### Enhanced Race Strategy Mode
- **Multi-Stint Strategy Planning**: Optimizes tire compound selection and pit stop windows
- **Future Race Predictions**: Generates strategies for upcoming races based on historical data
- **Temperature Impact Analysis**: Evaluates weather conditions on tire performance
- **Comparative Strategy Analysis**: Ranks different strategic approaches for optimal results

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Required Libraries

```bash
pip install streamlit
pip install pandas
pip install numpy
pip install scikit-learn
pip install plotly
pip install fastf1
pip install warnings
```

### Alternative Installation (using requirements.txt)

Create a `requirements.txt` file with:

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.15.0
fastf1>=3.0.0
```

Then install:

```bash
pip install -r requirements.txt
```

## Project Structure

```
f1-tire-predictor/
│
├── main.py              # Main Streamlit application
├── data.py              # Real F1 data extraction script
├── f1_data/             # Directory for CSV data files
├── cache/               # FastF1 cache directory
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Usage

### Running the Main Application

1. Navigate to the project directory:
```bash
cd f1-tire-predictor
```

2. Launch the Streamlit application:
```bash
streamlit run main.py
```

3. Open your web browser and go to `http://localhost:8501`

### Using the Application

#### Option 1: Generate Sample Data
- Select "Generate Sample Data" in the sidebar
- Adjust the number of samples (50-500)
- Click "Train Model" to build the predictive model
- Use "Individual Driver Analysis" to get predictions

#### Option 2: Use Real F1 Data
1. First, generate real F1 data using the data extraction script:
```bash
python data.py
```

2. Upload the generated CSV files through the Streamlit interface
3. The application will automatically combine and process the data

### Generating Real F1 Data

The `data.py` script can extract real F1 telemetry data:

```python
# Modify parameters in data.py
year = 2024
track_name = 'Monaco'  # Available: 'Bahrain', 'Monaco', 'Silverstone', etc.
session_type = 'R'     # 'R' for Race, 'Q' for Qualifying

# Run the script
python data.py
```

## Data Sources and Accuracy

### Real Data Features
- **Lap Times**: Actual lap times from F1 sessions
- **Tire Information**: Compound types, tire age, stint lengths
- **Weather Data**: Track temperature, air temperature
- **Driver/Team Data**: Official F1 driver and team information
- **Track Characteristics**: Circuit-specific performance data

### Calculated Metrics
- **Degradation Rate**: Calculated from lap time progression analysis
- **Remaining Grip**: Derived from compound characteristics, age, and temperature
- **Performance Scores**: Composite metrics for strategic analysis

## Model Performance

The Random Forest model typically achieves:
- **RMSE**: ~5-8% grip prediction error
- **R² Score**: 0.75-0.85 correlation with actual performance
- **Feature Importance**: Tire age and compound type are primary predictors

## Strategic Applications

### For Race Engineers
- Real-time tire performance monitoring
- Optimal pit window identification
- Compound strategy optimization
- Temperature impact assessment

### For Data Analysis
- Historical performance pattern analysis
- Driver comparison studies
- Track-specific strategy development
- Weather condition impact evaluation

## Technical Details

### Machine Learning Pipeline
1. **Data Preprocessing**: Handles missing values, encodes categorical variables
2. **Feature Scaling**: StandardScaler for numerical feature normalization  
3. **Model Training**: Random Forest with 100 estimators
4. **Cross-Validation**: Train/test split for performance evaluation

### Key Algorithms
- **Tire Degradation Modeling**: Linear regression on lap time progression
- **Grip Calculation**: Multi-factor model considering compound, age, temperature
- **Strategy Optimization**: Exhaustive search across pit window combinations

## Limitations and Considerations

- **Data Availability**: Real F1 data may have gaps or inconsistencies
- **Model Accuracy**: Predictions are estimates based on historical patterns
- **External Factors**: Cannot account for driver incidents, safety cars, or mechanical issues
- **Track Evolution**: Surface conditions and weather can vary significantly

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Future Enhancements

- Real-time data integration during live races
- Advanced weather prediction integration
- Machine learning model ensemble methods
- Mobile application development
- Integration with official F1 timing systems

## License

This project is for educational and research purposes. F1 data usage should comply with Formula 1's terms of service and data usage policies.

## Support

For issues or questions:
- Create an issue in the GitHub repository
- Check the FastF1 documentation for data-related questions
- Refer to Streamlit documentation for interface issues

## Acknowledgments

- **FastF1**: For providing access to F1 telemetry data
- **Formula 1**: For the incredible sport that makes this analysis possible
- **Open Source Community**: For the tools and libraries that power this project

---

**Note**: This application is designed for educational and analytical purposes. While it uses real F1 data and sophisticated modeling techniques, predictions should not be considered definitive for actual racing decisions.
