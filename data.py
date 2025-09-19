import fastf1
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def calculate_degradation_rate(stint_data):
    """
    Calculate tire degradation rate for a stint.
    Returns degradation in seconds per lap.
    """
    if len(stint_data) < 3:  # Need at least 3 laps for meaningful degradation
        return np.nan
    
    try:
        # Convert LapTime to seconds if it's not already
        if hasattr(stint_data['LapTime'].iloc[0], 'total_seconds'):
            lap_times_seconds = stint_data['LapTime'].dt.total_seconds()
        else:
            lap_times_seconds = pd.to_numeric(stint_data['LapTime'], errors='coerce')
        
        # Remove NaN values
        valid_times = lap_times_seconds.dropna()
        if len(valid_times) < 3:
            return np.nan
        
        # Remove outliers (laps that are >20% slower than median)
        median_time = valid_times.median()
        filtered_times = valid_times[valid_times <= median_time * 1.2]
        
        if len(filtered_times) < 3:
            return np.nan
        
        # Calculate degradation using linear regression
        lap_numbers = np.arange(len(filtered_times))
        
        # Linear regression to find degradation rate
        coefficients = np.polyfit(lap_numbers, filtered_times, 1)
        degradation_rate = coefficients[0]  # Slope = degradation per lap
        
        return degradation_rate
        
    except Exception as e:
        print(f"Error calculating degradation rate: {e}")
        return np.nan

def generate_f1_data(year, track_name, session_type='R'):
    """
    Generate F1 tire degradation data for a specific year and track.
    
    Parameters:
    year (int): Year of the race
    track_name (str): Name of the track (e.g., 'Bahrain', 'Monaco', 'Silverstone')
    session_type (str): 'R' for Race, 'Q' for Qualifying, 'FP1', 'FP2', 'FP3' for Practice
    
    Returns:
    pandas.DataFrame: DataFrame with tire degradation data
    """
    
    print(f"Loading {year} {track_name} {session_type} data...")
    
    try:
        # Enable cache for faster subsequent loads
        fastf1.Cache.enable_cache('cache')
        
        # Load session data
        session = fastf1.get_session(year, track_name, session_type)
        session.load()
        
        # Get all laps data
        laps = session.laps
        
        # Filter out invalid laps (outlaps, inlaps, pit laps, etc.)
        valid_laps = laps[
            (laps['LapTime'].notna()) & 
            (laps['Compound'].notna()) &
            (laps['TyreLife'].notna())
        ]
        
        # Additional filtering if IsAccurate column exists
        if 'IsAccurate' in valid_laps.columns:
            valid_laps = valid_laps[valid_laps['IsAccurate'] == True]
        
        print(f"Processing {len(valid_laps)} valid laps...")
        
        data_records = []
        
        # Group by driver and stint to calculate degradation
        for driver in valid_laps['Driver'].unique():
            driver_laps = valid_laps[valid_laps['Driver'] == driver]
            
            # Group by compound and consecutive tire life to identify stints
            driver_laps = driver_laps.sort_values('LapNumber')
            
            # Identify stint changes
            stint_changes = (driver_laps['TyreLife'].diff() < 0) | (driver_laps['Compound'].shift() != driver_laps['Compound'])
            stint_ids = stint_changes.cumsum()
            
            for stint_id in stint_ids.unique():
                stint_laps = driver_laps[stint_ids == stint_id]
                
                if len(stint_laps) < 2:  # Skip very short stints
                    continue
                
                # Calculate degradation rate for this stint
                degradation_rate = calculate_degradation_rate(stint_laps)
                
                # Get driver info
                try:
                    driver_info = session.get_driver(driver)
                    team_name = driver_info.get('TeamName', 'Unknown')
                except:
                    team_name = 'Unknown'
                
                # Process each lap in the stint
                for _, lap in stint_laps.iterrows():
                    try:
                        # Get weather data for this lap
                        weather_data = session.laps.get_weather_data()
                        if len(weather_data) > 0:
                            # Find closest weather data point
                            if hasattr(lap['Time'], 'total_seconds'):
                                lap_time_seconds = lap['Time'].total_seconds()
                            else:
                                lap_time_seconds = float(lap['Time'])
                            
                            weather_times = weather_data['Time'].dt.total_seconds()
                            closest_weather_idx = np.argmin(np.abs(weather_times - lap_time_seconds))
                            track_temp = weather_data.iloc[closest_weather_idx]['TrackTemp']
                            air_temp = weather_data.iloc[closest_weather_idx]['AirTemp']
                        else:
                            track_temp = np.nan
                            air_temp = np.nan
                        
                        # Convert lap time to seconds
                        if hasattr(lap['LapTime'], 'total_seconds'):
                            lap_time_seconds = lap['LapTime'].total_seconds()
                        else:
                            lap_time_seconds = float(lap['LapTime'])
                        
                        record = {
                            'driver': driver,
                            'team': team_name,
                            'compound': lap['Compound'],
                            'stint_length': len(stint_laps),
                            'tyre_age': int(lap['TyreLife']) if pd.notna(lap['TyreLife']) else 0,
                            'track_temp': track_temp,
                            'air_temp': air_temp,
                            'lap_time': lap_time_seconds,
                            'track_name': track_name,
                            'degradation_rate': degradation_rate
                        }
                        
                        data_records.append(record)
                        
                        
                    except Exception as e:
                        print(f"Error processing lap for {driver}: {e}")
                        continue
        
        # Convert to DataFrame
        df = pd.DataFrame(data_records)
        
        if len(df) == 0:
            print("No valid data found!")
            return pd.DataFrame()
        
        # Clean up data
        df['compound'] = df['compound'].str.upper()  # Normalize compound names to uppercase
        df = df.dropna(subset=['lap_time', 'compound', 'tyre_age'])

        # Calculate remaining_grip using the same logic as in cluade.py
        # Map compounds to factors
        compound_factor = df['compound'].map({'SOFT': 0.8, 'MEDIUM': 0.9, 'HARD': 1.0, 'INTERMEDIATE': 0.6, 'WET': 0.4})
        temp_factor = (df['track_temp'] - 35) * 0.5
        age_factor = df['tyre_age'] * 1.2
        # If degradation_rate is NaN, treat as 0 for grip calculation
        degradation_rate = df['degradation_rate'].fillna(0)
        n_samples = len(df)
        np.random.seed(42)  # For reproducibility
        df['remaining_grip'] = (
            100 - age_factor - temp_factor -
            degradation_rate * 50 +
            compound_factor * 5 +
            np.random.normal(0, 3, n_samples)
        )
        df['remaining_grip'] = np.clip(df['remaining_grip'], 10, 100)

        # Drop rows with NaN in any critical columns after calculation
        df = df.dropna(subset=['lap_time', 'compound', 'tyre_age', 'degradation_rate', 'remaining_grip'])

        # Print unique compound values and rows with NaN remaining_grip for debugging
        print("Unique compounds:", df['compound'].unique())
        if df['remaining_grip'].isna().any():
            print("Rows with NaN remaining_grip:")
            print(df[df['remaining_grip'].isna()])

        print(f"Generated {len(df)} records for {track_name} {year}")
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def save_to_csv(df, year, track_name, session_type='R'):
    """Save DataFrame to CSV file with descriptive filename."""
    if df.empty:
        print("No data to save!")
        return
    
    # Create filename
    filename = f"f1_tire_data_{year}_{track_name.replace(' ', '_')}_{session_type}.csv"
    
    # Create directory if it doesn't exist
    os.makedirs('f1_data', exist_ok=True)
    filepath = os.path.join('f1_data', filename)
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Data saved to: {filepath}")
    
    # Print summary statistics
    print("\nData Summary:")
    print(f"Total laps: {len(df)}")
    print(f"Drivers: {df['driver'].nunique()}")
    print(f"Teams: {df['team'].nunique()}")
    print(f"Compounds: {df['compound'].unique()}")
    print(f"Average degradation rate: {df['degradation_rate'].mean():.4f} seconds per lap")

def main():
    """Main function to run the data generation."""
    
    # Example usage - modify these parameters as needed
    year = 2024
    track_name = 'Belgian'  # Can be: 'Bahrain', 'Saudi Arabia', 'Australia', 'Monaco', etc.
    session_type = 'R'  # 'R' for Race, 'Q' for Qualifying, 'FP1', 'FP2', 'FP3' for Practice
    
    print("F1 Tire Degradation Data Generator")
    print("=" * 40)
    
    # Generate data
    df = generate_f1_data(year, track_name, session_type)
    
    if not df.empty:
        # Save to CSV
        save_to_csv(df, year, track_name, session_type)
        
        # Display first few rows
        print("\nFirst 5 rows of data:")
        print(df.head())
        
        # Display data types
        print("\nData types:")
        print(df.dtypes)
    else:
        print("No data generated!")

def generate_multiple_tracks(year, tracks, session_type='R'):
    """Generate data for multiple tracks in a single year."""
    all_data = []
    
    for track in tracks:
        print(f"\nProcessing {track}...")
        df = generate_f1_data(year, track, session_type)
        if not df.empty:
            all_data.append(df)
    
    if all_data:
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Save combined data
        filename = f"f1_tire_data_{year}_all_tracks_{session_type}.csv"
        os.makedirs('f1_data', exist_ok=True)
        filepath = os.path.join('f1_data', filename)
        combined_df.to_csv(filepath, index=False)
        
        print(f"\nCombined data saved to: {filepath}")
        print(f"Total records: {len(combined_df)}")
        
        return combined_df
    else:
        print("No data generated for any track!")
        return pd.DataFrame()

if __name__ == "__main__":
    # Example 1: Generate data for a single track
    main()
    
    # Example 2: Generate data for multiple tracks (uncomment to use)
    # tracks = ['Bahrain', 'Saudi Arabia', 'Australia', 'Monaco', 'Spain']
    # generate_multiple_tracks(2023, tracks, 'R')
    
    # Example 3: Generate data for multiple years (uncomment to use)
    # for year in [2022, 2023]:
    #     df = generate_f1_data(year, 'Silverstone', 'R')
    #     if not df.empty:
    #         save_to_csv(df, year, 'Silverstone', 'R')

# Available track names (some examples):
# 'Bahrain', 'Saudi Arabia', 'Australia', 'Emilia Romagna', 'Monaco', 
# 'Spain', 'Canada', 'Austria', 'Great Britain', 'Hungary', 'Belgium', 
# 'Netherlands', 'Italy', 'Singapore', 'Japan', 'Qatar', 'United States', 
# 'Mexico', 'Brazil', 'Las Vegas', 'Abu Dhabi'