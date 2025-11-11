import logging
import pandas as pd
import joblib
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)

class FastAPIHandler():
    def __init__(self, model_path: str = '/models/car_price_model.pkl'):
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
                if os.path.exists('/models'):
                    logger.info(f' Models directory contents: {os.listdir("/models")}')
                
        except Exception as e:
            logger.error(f' Error loading model: {e}')
            self.model = None
            self.expected_features = None

    def predict(self, item_features: Dict[str, Any]):
        if self.model is None:
            raise Exception("Model not loaded. Please check if model file exists.")
        
        try:
            logger.info(f' Making prediction with features: {list(item_features.keys())}')
            
            
            if self.expected_features:
                
                ordered_features = {}
                missing_features = []
                
                for feature in self.expected_features:
                    if feature in item_features:
                        ordered_features[feature] = item_features[feature]
                    else:
                        missing_features.append(feature)
                
                if missing_features:
                    raise Exception(f"Missing required features: {missing_features}")
                
                item_df = pd.DataFrame([ordered_features])
                item_df = item_df[self.expected_features]  
            else:
                
                item_df = pd.DataFrame([item_features])
                logger.warning(' Using features in arbitrary order')
            
            logger.info(f' DataFrame shape: {item_df.shape}')
            logger.info(f' DataFrame columns: {item_df.columns.tolist()}')
            
            
            prediction = self.model.predict(item_df)
            logger.info(f' Prediction result: {prediction[0]}')
            
            return prediction
            
        except Exception as e:
            logger.error(f' Prediction error: {e}')
            raise e

    def get_expected_features(self):
        """Возвращает список ожидаемых моделью фич"""
        return self.expected_features