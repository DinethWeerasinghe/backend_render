#!/usr/bin/env python3
"""
Python Bridge for Node.js Backend
==================================

This script is called by the Node.js server to make predictions
using the pickle model.

Usage:
  python3 predict.py <datetime> <flare_probability>
  python3 predict.py batch <csv_path>

Returns JSON output for Node.js to parse.
"""

import sys
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime


def add_features(df, model_package):
    """Add required features to dataframe."""
    
    # Solar cycle features
    solar_min = pd.Timestamp(model_package['solar_cycle_info']['solar_minimum_date'])
    df['years_from_minimum'] = (df['DateTime'] - solar_min).dt.days / 365.25
    df['solar_cycle_phase'] = np.clip(df['years_from_minimum'] / 5.5, 0, 1)
    
    cycle_rad = 2 * np.pi * df['years_from_minimum'] / 11
    df['solar_cycle_sin'] = np.sin(cycle_rad)
    df['solar_cycle_cos'] = np.cos(cycle_rad)
    
    # Hour encoding
    df['hour'] = df['DateTime'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Interaction
    df['flare_prob_x_phase'] = df['flare_probability'] * df['solar_cycle_phase']
    
    return df


def categorize_risk(probability):
    """Categorize probability into risk level."""
    if probability < 0.1:
        return 'Very Low'
    elif probability < 0.3:
        return 'Low'
    elif probability < 0.5:
        return 'Moderate'
    elif probability < 0.7:
        return 'High'
    else:
        return 'Very High'


def predict_single(datetime_str, flare_prob, model_path):
    """Make a single prediction."""
    
    try:
        # Load model
        with open(model_path, 'rb') as f:
            pkg = pickle.load(f)
        
        # Prepare data
        df = pd.DataFrame({
            'DateTime': [pd.to_datetime(datetime_str)],
            'flare_probability': [float(flare_prob)]
        })
        
        # Add features
        df = add_features(df, pkg)
        
        # Predict
        X = df[pkg['feature_names']].values
        X_scaled = pkg['scaler'].transform(X)
        prob = pkg['model'].predict_proba(X_scaled)[0, 1]
        
        # Result
        result = {
            'success': True,
            'datetime': datetime_str,
            'flare_probability': float(flare_prob),
            'blackout_probability': float(prob),
            'risk_level': categorize_risk(prob),
            'solar_cycle_phase': float(df['solar_cycle_phase'].iloc[0])
        }
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def predict_batch(csv_path, model_path):
    """Make batch predictions from CSV file."""
    
    try:
        # Load model
        with open(model_path, 'rb') as f:
            pkg = pickle.load(f)
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Handle different CSV formats
        if 'DateTime' in df.columns:
            df['DateTime'] = pd.to_datetime(df['DateTime'])
        elif 'date' in df.columns and 'hour' in df.columns:
            df['DateTime'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
        else:
            raise ValueError("CSV must have 'DateTime' column or 'date' and 'hour' columns")
        
        # Handle different probability column names
        if 'solar_flare_probability' in df.columns:
            df = df.rename(columns={'solar_flare_probability': 'flare_probability'})
        elif 'flare_probability' not in df.columns:
            raise ValueError("CSV must have 'solar_flare_probability' or 'flare_probability' column")
        
        # Add features
        df = add_features(df, pkg)
        
        # Predict
        X = df[pkg['feature_names']].values
        X_scaled = pkg['scaler'].transform(X)
        probabilities = pkg['model'].predict_proba(X_scaled)[:, 1]
        
        # Add results
        df['blackout_probability'] = probabilities
        df['risk_level'] = [categorize_risk(p) for p in probabilities]
        
        # Create results
        results = []
        for _, row in df.iterrows():
            results.append({
                'datetime': str(row['DateTime']),
                'flare_probability': float(row['flare_probability']),
                'blackout_probability': float(row['blackout_probability']),
                'risk_level': row['risk_level'],
                'solar_cycle_phase': float(row['solar_cycle_phase'])
            })
        
        return {
            'success': True,
            'count': len(results),
            'predictions': results,
            'summary': {
                'mean_blackout_probability': float(np.mean(probabilities)),
                'max_blackout_probability': float(np.max(probabilities)),
                'risk_distribution': {
                    'Very Low': int((df['risk_level'] == 'Very Low').sum()),
                    'Low': int((df['risk_level'] == 'Low').sum()),
                    'Moderate': int((df['risk_level'] == 'Moderate').sum()),
                    'High': int((df['risk_level'] == 'High').sum()),
                    'Very High': int((df['risk_level'] == 'Very High').sum())
                }
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def main():
    """Main entry point."""
    
    if len(sys.argv) < 2:
        print(json.dumps({
            'success': False,
            'error': 'Usage: python3 predict.py <datetime> <flare_probability> OR python3 predict.py batch <csv_path>'
        }))
        sys.exit(1)
    
    # Model path (adjust if needed)
    model_path = 'hf_blackout_model.pkl'
    
    if sys.argv[1] == 'batch':
        # Batch prediction
        if len(sys.argv) < 3:
            print(json.dumps({
                'success': False,
                'error': 'Usage: python3 predict.py batch <csv_path>'
            }))
            sys.exit(1)
        
        csv_path = sys.argv[2]
        result = predict_batch(csv_path, model_path)
        
    else:
        # Single prediction
        if len(sys.argv) < 3:
            print(json.dumps({
                'success': False,
                'error': 'Usage: python3 predict.py <datetime> <flare_probability>'
            }))
            sys.exit(1)
        
        datetime_str = sys.argv[1]
        flare_prob = sys.argv[2]
        result = predict_single(datetime_str, flare_prob, model_path)
    
    # Output JSON
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
