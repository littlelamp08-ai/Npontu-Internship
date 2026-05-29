# Iris Classification Pipeline & Inference API

This repository contains a production-ready machine learning pipeline that handles data preprocessing, model training, and performance evaluation using the public Iris dataset. The final trained pipeline is exposed via a high-performance REST API built with FastAPI.

## Project Structure
```text
├── app.py                 # Core application script (Pipeline + FastAPI Service)
├── requirements.txt       # Production library dependencies
└── README.md              # Project setup and monitoring documentation

1. Local Setup & Execution Instructions
Follow these steps to configure your environment and deploy the inference service locally:

Step 1: Install Dependencies
Ensure you have Python 3.9+ installed. Run the following command to install the required production libraries:

Bash
pip install -r requirements.txt

Step 2: Launch the REST API
Run the Uvicorn ASGI server to initialize the model training pipeline and spin up the production API instance:

Bash
uvicorn app:app --reload
Note: On its initial execution, the script automatically triggers the model training pipeline to generate and serialize the model_pipeline.pkl artifact safely before starting the web server workers.

Step 3: Verify the Deployment
Open your web browser and navigate to:

Interactive API Documentation (Swagger UI): http://127.0.0.1:8000/docs

This interface allows you to execute test payloads directly against the live /predict endpoint.



