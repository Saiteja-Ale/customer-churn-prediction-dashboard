import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictionModel:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_importance = {}
        
    def load_and_preprocess_data(self, file_path='customer_churn_data.csv'):
        """Load and preprocess the customer data"""
        print("📂 Loading customer data...")
        df = pd.read_csv(file_path)
        
        # Handle missing values (if any)
        df = df.dropna()
        
        # Convert TotalCharges to numeric (handle any string values)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df = df.dropna()
        
        print(f"✅ Loaded {len(df)} customer records")
        return df
    
    def feature_engineering(self, df):
        """Create additional features and encode categorical variables"""
        print("🔧 Engineering features...")
        
        # Create new features
        df['MonthlyCharges_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
        df['TotalCharges_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1)
        df['HighValue_Customer'] = (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.75)).astype(int)
        df['LowTenure'] = (df['tenure'] <= 12).astype(int)
        df['HighSupport'] = (df['SupportCalls'] > 3).astype(int)
        
        # Service count features
        service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        df['TotalServices'] = 0
        for col in service_cols:
            if col in df.columns:
                df['TotalServices'] += (df[col] == 'Yes').astype(int)
        
        return df
    
    def encode_categorical_features(self, df, is_training=True):
        """Encode categorical variables"""
        print("🏷️ Encoding categorical features...")
        
        categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                           'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                           'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                           'PaperlessBilling', 'PaymentMethod']
        
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col in df.columns:
                if is_training:
                    # Fit and transform during training
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    # Only transform during prediction
                    if col in self.label_encoders:
                        # Handle unseen categories
                        unique_values = set(df_encoded[col].astype(str))
                        known_values = set(self.label_encoders[col].classes_)
                        
                        # Replace unseen values with most frequent class
                        if not unique_values.issubset(known_values):
                            most_frequent = self.label_encoders[col].classes_[0]
                            df_encoded[col] = df_encoded[col].astype(str).apply(
                                lambda x: x if x in known_values else most_frequent
                            )
                        
                        df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def prepare_features_target(self, df):
        """Prepare features and target variable"""
        # Target variable
        y = (df['Churn'] == 'Yes').astype(int)
        
        # Features (exclude ID and target)
        exclude_cols = ['customerID', 'Churn']
        X = df.drop(columns=exclude_cols)
        
        return X, y
    
    def train_models(self, X_train, y_train):
        """Train multiple models and compare performance"""
        print("🤖 Training machine learning models...")
        
        # Initialize models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
        }
        
        model_scores = {}
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            model_scores[name] = cv_scores.mean()
            
            # Fit the model
            model.fit(X_train, y_train)
            self.models[name] = model
            
            # Store feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                self.feature_importance[name] = np.abs(model.coef_[0])
        
        # Select best model
        best_model_name = max(model_scores, key=model_scores.get)
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        print(f"\n📊 Model Performance (ROC-AUC):")
        for name, score in model_scores.items():
            print(f"{name}: {score:.4f}")
        
        print(f"\n🏆 Best Model: {best_model_name} (ROC-AUC: {model_scores[best_model_name]:.4f})")
        
        return model_scores
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the best model on test set"""
        print(f"\n🎯 Evaluating {self.best_model_name} on test set...")
        
        # Predictions
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = self.best_model.score(X_test, y_test)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC-AUC: {auc_score:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
        
        return accuracy, auc_score, y_pred_proba
    
    def plot_feature_importance(self, feature_names, top_n=15):
        """Plot feature importance"""
        if self.best_model_name in self.feature_importance:
            importance = self.feature_importance[self.best_model_name]
            
            # Create dataframe for plotting
            feature_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(top_n)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=feature_df, x='importance', y='feature')
            plt.title(f'Top {top_n} Feature Importance - {self.best_model_name}')
            plt.xlabel('Feature Importance')
            plt.tight_layout()
            plt.show()
    
    def generate_predictions_for_dashboard(self, df, X_test, y_test, y_pred_proba):
        """Generate predictions with customer details for dashboard"""
        print("📈 Generating predictions for dashboard...")
        
        # Create results dataframe
        test_indices = X_test.index
        results_df = df.loc[test_indices].copy()
        
        # Add predictions
        results_df['Churn_Probability'] = y_pred_proba
        results_df['Risk_Score'] = np.round(y_pred_proba * 100, 1)
        results_df['Actual_Churn'] = y_test.values
        results_df['Predicted_Churn'] = (y_pred_proba > 0.5).astype(int)
        
        # Create risk categories
        results_df['Risk_Category'] = pd.cut(results_df['Risk_Score'], 
                                           bins=[0, 30, 60, 80, 100], 
                                           labels=['Low', 'Medium', 'High', 'Critical'])
        
        # Calculate customer lifetime value (simplified)
        results_df['CLV_Estimate'] = results_df['MonthlyCharges'] * 24  # 2-year estimate
        results_df['Revenue_at_Risk'] = results_df['CLV_Estimate'] * results_df['Churn_Probability']
        
        return results_df

# Main execution
def main():
    # Initialize the model
    churn_model = ChurnPredictionModel()
    
    # Load and preprocess data
    df = churn_model.load_and_preprocess_data()
    
    # Feature engineering
    df = churn_model.feature_engineering(df)
    
    # Encode categorical variables
    df_encoded = churn_model.encode_categorical_features(df, is_training=True)
    
    # Prepare features and target
    X, y = churn_model.prepare_features_target(df_encoded)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Churn rate in training: {y_train.mean():.3f}")
    print(f"Churn rate in test: {y_test.mean():.3f}")
    
    # Train models
    model_scores = churn_model.train_models(X_train, y_train)
    
    # Evaluate best model
    accuracy, auc_score, y_pred_proba = churn_model.evaluate_model(X_test, y_test)
    
    # Plot feature importance
    churn_model.plot_feature_importance(X.columns)
    
    # Generate dashboard data
    dashboard_data = churn_model.generate_predictions_for_dashboard(df, X_test, y_test, y_pred_proba)
    
    # Save results
    dashboard_data.to_csv('churn_predictions_dashboard.csv', index=False)
    print("\n💾 Dashboard data saved as 'churn_predictions_dashboard.csv'")
    
    # Summary statistics for dashboard
    print("\n📊 Dashboard Summary:")
    print(f"Total customers analyzed: {len(dashboard_data)}")
    print(f"High-risk customers (>60%): {sum(dashboard_data['Risk_Score'] > 60)}")
    print(f"Critical-risk customers (>80%): {sum(dashboard_data['Risk_Score'] > 80)}")
    print(f"Total revenue at risk: ${dashboard_data['Revenue_at_Risk'].sum():,.2f}")
    print(f"Average customer lifetime value: ${dashboard_data['CLV_Estimate'].mean():,.2f}")
    
    print("\nRisk Category Distribution:")
    print(dashboard_data['Risk_Category'].value_counts())
    
    return churn_model, dashboard_data

if __name__ == "__main__":
    model, predictions = main()