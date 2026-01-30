"""
Data loading utilities for the data analyst portfolio
"""
import streamlit as st
import pandas as pd
import os

@st.cache_data
def load_stock_data():
    """Load stock price data with caching"""
    try:
        # Try to load from data_analyst/data folder first
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'stock_price.csv')
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
        else:
            # Fallback to parent directory
            parent_path = os.path.join(os.path.dirname(__file__), '..', '..', 'stock_price.csv')
            if os.path.exists(parent_path):
                df = pd.read_csv(parent_path)
            else:
                # Show helpful error message
                st.error("📁 **Data File Not Found**")
                st.warning("""
                The `stock_price.csv` file is not included in this repository due to GitHub's file size limits.
                
                **To use this app:**
                1. Download the stock price dataset
                2. Upload it to a cloud storage (Google Drive, Dropbox, etc.)
                3. Update this function to load from URL
                
                **Example:**
                ```python
                df = pd.read_csv('https://your-storage-url/stock_price.csv')
                ```
                
                See `data/README.md` for more information.
                """)
                return None
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date')
        
        return df
    except Exception as e:
        st.error(f"❌ Error loading stock data: {e}")
        st.info("💡 Check that the CSV file exists and is properly formatted.")
        return None

@st.cache_data
def load_credit_card_data(sample_size=50000):
    """
    Load credit card fraud data with sampling for performance
    
    Args:
        sample_size: Number of rows to sample (default 50000)
    """
    try:
        # Try to load from data_analyst/data folder first
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'creditcard.csv')
        if os.path.exists(data_path):
            # For large files, use chunksize and sample
            df = pd.read_csv(data_path, nrows=sample_size)
        else:
            # Fallback to parent directory
            parent_path = os.path.join(os.path.dirname(__file__), '..', '..', 'creditcard.csv')
            if os.path.exists(parent_path):
                df = pd.read_csv(parent_path, nrows=sample_size)
            else:
                # Show helpful error message
                st.error("📁 **Data File Not Found**")
                st.warning("""
                The `creditcard.csv` file is not included in this repository due to GitHub's 100 MB file size limit.
                This file is 143.84 MB and cannot be stored directly in GitHub.
                
                **Options to use this app:**
                
                **Option 1: Use Public Dataset**
                Download from Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
                
                **Option 2: Cloud Storage**
                1. Upload the CSV to Google Drive, Dropbox, or AWS S3
                2. Get a public URL
                3. Update this function to load from URL
                
                **Example:**
                ```python
                df = pd.read_csv('https://your-storage-url/creditcard.csv', nrows=sample_size)
                ```
                
                See `data/README.md` for detailed instructions.
                """)
                return None
        
        return df
    except Exception as e:
        st.error(f"❌ Error loading credit card data: {e}")
        st.info("💡 Check that the CSV file exists and has the correct format.")
        return None

def get_data_info(df):
    """Get basic information about the dataset"""
    if df is None:
        return None
    
    info = {
        'rows': len(df),
        'columns': len(df.columns),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        'missing_values': df.isnull().sum().sum(),
        'dtypes': df.dtypes.value_counts().to_dict()
    }
    
    return info

def preprocess_stock_data(df):
    """Preprocess stock data for analysis"""
    if df is None:
        return None
    
    # Calculate additional metrics
    df['daily_return'] = df['last_value'].pct_change()
    df['price_range'] = df['high_value'] - df['low_value']
    df['price_change'] = df['last_value'] - df['open_value']
    
    # Calculate moving averages
    df['ma_7'] = df['last_value'].rolling(window=7).mean()
    df['ma_30'] = df['last_value'].rolling(window=30).mean()
    
    # Calculate volatility (rolling standard deviation)
    df['volatility'] = df['daily_return'].rolling(window=30).std()
    
    return df

def preprocess_fraud_data(df):
    """Preprocess credit card fraud data"""
    if df is None:
        return None
    
    # Add hour of day from Time column
    df['Hour'] = (df['Time'] / 3600) % 24
    
    # Create bins for amount
    df['Amount_Bin'] = pd.cut(df['Amount'], 
                               bins=[0, 10, 50, 100, 500, float('inf')],
                               labels=['0-10', '10-50', '50-100', '100-500', '500+'])
    
    return df
