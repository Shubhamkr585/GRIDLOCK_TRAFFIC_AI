# GridLock AI: ML Engine

This directory contains the Python Machine Learning microservice.

## How to Run the API Server
By default, the Docker container is configured to boot up the FastAPI server, serving predictions via the saved `lgbm_model.pkl` file.
```bash
# From the root of the project:
docker-compose up --build
```
The API will be available at: `http://localhost:8000`

## How to Retrain the Model
The model does **not** train automatically when the container starts (to save time and memory). If you get a new dataset and need to retrain the model, follow these steps:

1. Ensure your new raw CSV dataset is placed in the `../dataset/` folder.
2. Update the `DATASET_PATH` in `config.py` if the filename changed.
3. Open a terminal and run the training script manually inside the Docker container:
```bash
docker exec -it gridlock-ml-engine python train_model.py
```
This will process the new dataset, overwrite the `lgbm_model.pkl` file with the updated model, and you will be good to go!
