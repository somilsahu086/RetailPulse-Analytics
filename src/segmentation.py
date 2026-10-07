import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def compute_rfm_and_clusters(df, n_clusters=4):
    
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    if 'TotalAmount' not in df.columns:
        df['TotalAmount'] = df['Quantity'] * df['Price']
        
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'Invoice': 'nunique',
        'TotalAmount': 'sum'
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    rfm['Avg_Order_Value'] = rfm['Monetary'] / rfm['Frequency']
    rfm['Churn'] = (rfm['Recency'] > 90).astype(int)
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    return rfm

if __name__ == "__main__":
    df_clean = pd.read_excel('../data/cleaned_data.xlsx')
    rfm_df = compute_rfm_and_clusters(df_clean)
    rfm_df.to_csv('../data/customer_segments.csv', index=False)
    print("✅ Segmentation script successfully executed and saved.")