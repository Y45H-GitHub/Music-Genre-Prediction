import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
import os

def load_dataset():
    """Load dataset from local CSV file with fallback to Kaggle if needed"""
    local_path = "./Data/features_3_sec.csv"
    
    if os.path.exists(local_path):
        print("📁 Loading dataset from local file...")
        return pd.read_csv(local_path)
    else:
        try:
            print("🌐 Downloading dataset from Kaggle...")
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(
                "andradaolteanu/gtzan-dataset-music-genre-classification",
                path=".",
                unzip=True
            )
            return pd.read_csv(local_path)
        except Exception as e:
            raise Exception(f"Failed to load dataset: {str(e)}")

def preprocess_data(df):
    """Preprocess the dataset and return features/labels"""
    print("🔄 Preprocessing data...")
    
    # Convert categorical labels
    df['label'] = df['label'].astype('category')
    df['class_label'] = df['label'].cat.codes
    
    # Create genre mapping dictionary
    genre_mapping = dict(zip(df.class_label.unique(), df.label.unique()))
    
    # Prepare features and labels
    X = df.drop(['filename', 'label', 'class_label'], axis=1)  # Exclude non-feature columns
    y = df['class_label']
    
    return X, y, genre_mapping

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate machine learning models"""
    print("🤖 Training models...")
    
    # Normalize features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    pickle.dump(scaler, open('scaler.pkl', 'wb'))
    
    # Train KNN model
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train_scaled, y_train)
    knn_pred = knn.predict(X_test_scaled)
    knn_acc = accuracy_score(y_test, knn_pred)
    
    # Train SVM model
    svm = SVC(kernel='linear', C=10, probability=True)  # Added probability for potential future use
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)
    svm_acc = accuracy_score(y_test, svm_pred)
    
    # Save models
    pickle.dump(knn, open('model_knn.pkl', 'wb'))
    pickle.dump(svm, open('model_svm.pkl', 'wb'))
    
    # Print evaluation results
    print("\n📊 Model Performance:")
    print(f"KNN Accuracy: {knn_acc:.2%}")
    print(classification_report(y_test, knn_pred, target_names=list(genre_mapping.values())))
    
    print(f"\nSVM Accuracy: {svm_acc:.2%}")
    print(classification_report(y_test, svm_pred, target_names=list(genre_mapping.values())))
    
    return knn, svm

if __name__ == "__main__":
    try:
        # Load and preprocess data
        df = load_dataset()
        X, y, genre_mapping = preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42,
            stratify=y  # Maintain class distribution
        )
        
        # Train and evaluate models
        knn_model, svm_model = train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        # Save genre mapping for later use
        pickle.dump(genre_mapping, open('genre_mapping.pkl', 'wb'))
        
        print("✅ All operations completed successfully!")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")