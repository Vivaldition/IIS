#!/bin/bash
export MLFLOW_TRACKING_URI="sqlite:///$PWD/mlflow/mlruns.db"
mlflow ui --backend-store-uri $MLFLOW_TRACKING_URI --port 5000 --host 127.0.0.1
