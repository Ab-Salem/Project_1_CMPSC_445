from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_models(df):
    # Prepare features and target
    features = [col for col in df.columns if col.startswith('requires_') or 
                col.startswith('title_') or col.startswith('state_') or 
                col in ['is_senior', 'is_remote']]
    
    X = df[features]
    y = df['numeric_salary']
    
    # Drop rows with missing salary information
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model 1: Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Model 2: Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate models
    models = {
        'Linear Regression': lr_model,
        'Random Forest': rf_model
    }
    
    results = {}
    for name, model in models.items():
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        results[name] = {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    
    return models, results, X.columns
