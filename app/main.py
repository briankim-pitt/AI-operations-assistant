from fastapi import FastAPI

app = FastAPI(title="AI Operations Assistant")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}