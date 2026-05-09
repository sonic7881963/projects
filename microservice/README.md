# Text Classification Microservice

A FastAPI microservice that serves a packaged NLP text classifier. The service loads a `joblib` model artifact, exposes health and prediction endpoints, and returns both the predicted class and class probabilities.

## What This Demonstrates

- Packaging a trained ML model behind an API.
- FastAPI request/response modeling with Pydantic.
- Loading persisted model artifacts and metadata at service startup.
- Returning structured prediction output suitable for client applications.

## Model

The included metadata identifies the model as `module3_tfidf_logistic_regression`. It classifies text into four labels:

- `comp.graphics`
- `rec.sport.baseball`
- `sci.med`
- `talk.politics.misc`

The stored metadata reports a test accuracy of `0.8969210174029452` for the packaged model.

## API Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Basic service status and endpoint list. |
| `GET` | `/health` | Model name and label metadata. |
| `POST` | `/predict` | Predicts a label for an input text string. |
| `GET` | `/docs` | FastAPI-generated API documentation. |

## Run Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"The team won the baseball game.\"}"
```

## Project Structure

```text
microservice/
  main.py
  requirements.txt
  artifacts/
    metadata.json
    text_classifier.joblib
```
