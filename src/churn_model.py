import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_and_save_model(rfm_df, model_save_path='../model/churn_model.pkl'):
    
    X = rfm_df[['Frequency', 'Monetary', 'Avg_Order_Value', 'Cluster']]
    y = rfm_df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Model Training Complete. Test Accuracy: {acc * 100:.2f}%")
    
    joblib.dump(model, model_save_path)
    print(f"Trained model saved at: {model_save_path}")
    return model

def predict_churn(input_features, model_path='../src/churn_model.pkl'):
    
    model = joblib.load(model_path)
    prediction = model.predict(input_features)
    probability = model.predict_proba(input_features)[:, 1]
    return prediction, probability

if __name__ == "__main__":
    # Test training pipeline locally
    rfm_data = pd.read_csv('../data/customer_segments.csv')
    train_and_save_model(rfm_data)