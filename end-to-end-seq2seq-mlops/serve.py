from fastapi import FastAPI
import uvicorn
import mlflow.pyfunc

app = FastAPI()

# Load model from registry (placeholder)
model = None

@app.post("/predict")
def predict(text: str):
    return {"translation": "Bonjour le monde"}  # Placeholder

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
