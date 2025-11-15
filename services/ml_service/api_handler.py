import logging
import pandas as pd
import joblib
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)

class FastAPIHandler():
    def __init__(self, model_path: str = '/app/car_price_model.pkl'):
        logger.info(' Initializing FastAPIHandler...')
        self.model = None
        self.expected_features = None
        
        try:
            if os.path.exists(model_path):
                logger.info(f' Loading model from: {model_path}')
                self.model = joblib.load(model_path)
                logger.info(' Model loaded successfully')
                
                
                if hasattr(self.model, 'feature_names_in_'):
                    self.expected_features = self.model.feature_names_in_.tolist()
                    logger.info(f' Model expects features: {self.expected_features}')
                else:
                    logger.warning(' Model does not have feature_names_in_ attribute')
                    
                    if hasattr(self.model, 'feature_importances_'):
                        num_features = len(self.model.feature_importances_)
                        self.expected_features = [f'feature_{i}' for i in range(num_features)]
                        logger.info(f' Assuming {num_features} features')
            else:
                logger.error(f' Model file not found at: {model_path}')
                logger.info(f' Current directory: {os.getcwd()}')
                logger.info(f' Directory contents: {os.listdir("/")}')
                if os.path.exists('/app'):
                    logger.info(f' App directory contents: {os.listdir("/app")}')
                
        except Exception as e:
            logger.error(f' Error loading model: {e}')
            self.model = None
            self.expected_features = None

    def convert_api_to_model_features(self, api_features):
        
        return {
            'Car_Name': f"{api_features['brand']} {api_features['model']}",
            'Year': api_features['year'],
            'Present_Price': api_features['engine_volume'] * 5,  # пример преобразования
            'Driven_kms': api_features['mileage'],
            'Fuel_Type': api_features['fuel_type'],
            'Selling_type': 'Dealer',  # значение по умолчанию
            'Transmission': api_features['transmission'],
            'Owner': 0,  # значение по умолчанию
            'Car_Age': 2024 - api_features['year']  # вычисляем возраст
        }

    def predict(self, item_features: Dict[str, Any]):
        if self.model is None:
            raise Exception("Model not loaded. Please check if model file exists.")
        
        try:
            logger.info(f' Making prediction with API features: {list(item_features.keys())}')
            
            
            model_features = self.convert_api_to_model_features(item_features)
            logger.info(f' Converted to model features: {list(model_features.keys())}')
            
            if self.expected_features:
                
                ordered_features = {}
                missing_features = []
                
                for feature in self.expected_features:
                    if feature in model_features:
                        ordered_features[feature] = model_features[feature]
                    else:
                        missing_features.append(feature)
                
                if missing_features:
                    raise Exception(f"Missing required features: {missing_features}")
                
                item_df = pd.DataFrame([ordered_features])
                item_df = item_df[self.expected_features]  
            else:
                item_df = pd.DataFrame([model_features])
                logger.warning(' Using features in arbitrary order')
            
            logger.info(f' DataFrame shape: {item_df.shape}')
            logger.info(f' DataFrame columns: {item_df.columns.tolist()}')
            logger.info(f' DataFrame values: {item_df.values.tolist()}')
            
            prediction = self.model.predict(item_df)
            logger.info(f' Prediction result: {prediction[0]}')
            
            return prediction
            
        except Exception as e:
            logger.error(f' Prediction error: {e}')
            raise e

    def get_expected_features(self):
        return self.expected_features
