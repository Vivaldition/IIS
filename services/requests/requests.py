import urllib.request
import json
import time
import random
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PREDICTION_SERVICE_URL = os.getenv('PREDICTION_SERVICE_URL', 'http://car-price-predict:8000')

def generate_car_data():
    return {
        'year': random.randint(2000, 2023),
        'mileage': random.randint(0, 300000),
        'engine_volume': round(random.uniform(1.0, 3.0), 1),
        'horsepower': random.randint(80, 300),
        'brand': random.choice(['toyota', 'honda', 'ford', 'bmw', 'mercedes']),
        'model': random.choice(['camry', 'civic', 'focus', 'x5', 'c-class']),
        'transmission': random.choice(['automatic', 'manual']),
        'fuel_type': random.choice(['petrol', 'diesel', 'hybrid'])
    }

def send_request():
    try:
        data = generate_car_data()
        logger.info('Sending request: %s', data)
        
        # Используем urllib вместо requests
        req = urllib.request.Request(
            f'{PREDICTION_SERVICE_URL}/predict',
            data=json.dumps(data).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode())
            logger.info(' Prediction: %s', result)
            
    except Exception as e:
        logger.error(' Request failed: %s', e)

def main():
    logger.info(' Starting requests service (using urllib)')
    while True:
        send_request()
        sleep_time = random.uniform(1, 5)
        time.sleep(sleep_time)

if __name__ == '__main__':
    main()
