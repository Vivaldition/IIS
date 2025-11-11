import mlflow.sklearn
import joblib
import os

def download_model():
    """
    Загружает модель из MLflow по run_id
    """
    
    run_id = "a0c1e800cf7c4171847779c1deb7ca27"
    
    model_uri = f"runs:/{run_id}/model"
    
    try:
        print(f" Загружаем модель из MLflow...")
        print(f"   Run ID: {run_id}")
        print(f"   Model URI: {model_uri}")
        
        
        model = mlflow.sklearn.load_model(model_uri)
        
        
        model_path = os.path.join(os.path.dirname(__file__), "car_price_model.pkl")
        joblib.dump(model, model_path)
        
        print(f" Модель успешно загружена и сохранена!")
        print(f" Путь: {model_path}")
        print(f" Размер файла: {os.path.getsize(model_path)} байт")
        
        
        print(f" Тип модели: {type(model)}")
        if hasattr(model, 'feature_names_in_'):
            print(f" Ожидаемые фичи: {model.feature_names_in_.tolist()}")
        
        return model_path
        
    except Exception as e:
        print(f" Ошибка при загрузке модели: {e}")
        return None

if __name__ == "__main__":
    download_model()