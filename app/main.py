import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.model_loader import model_loader
from app.schemas import ParseRequest, ParseResponse


app = FastAPI(
    title="Lightweight Command Parsing API",
    version=os.getenv(
        "MODEL_VERSION",
        "v1"
    )
)


@app.on_event("startup")
def startup_event():
    """
    Load the model when the FastAPI application starts.
    """

    model_loader.load()


@app.middleware("http")
async def add_model_headers(
    request: Request,
    call_next
):
    """
    Add model version and serving-instance headers
    to every API response.
    """

    response = await call_next(request)

    response.headers[
        "X-Model-Version"
    ] = model_loader.model_version

    response.headers[
        "X-Served-By"
    ] = model_loader.served_by

    return response


@app.get("/healthz")
def healthz():
    """
    Kubernetes liveness probe.

    This checks whether the application process is alive.
    """

    return {
        "status": "ok"
    }


@app.get("/readyz")
def readyz():
    """
    Kubernetes readiness probe.

    The application is ready only after the model
    has successfully loaded.
    """

    if not model_loader.is_ready:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready"
            }
        )

    return {
        "status": "ready"
    }


@app.post(
    "/parse",
    response_model=ParseResponse
)
def parse_command(
    request: ParseRequest
):
    """
    Parse a natural-language command using
    the distilled Pythia-70M model.
    """

    if not model_loader.is_ready:
        raise HTTPException(
            status_code=503,
            detail="model_not_ready"
        )

    try:
        # Run model inference.
        result = model_loader.generate(
            request.text
        )

        # Validate the model output against
        # the required API schema.
        parsed = ParseResponse.model_validate(
            result
        )

        return parsed

    except ValueError as exc:
        print(
            f"[ERROR] Invalid model output: {exc}"
        )

        # This is explicitly required by
        # the assignment for invalid model output.
        return JSONResponse(
            status_code=500,
            content={
                "error": "invalid_model_output"
            }
        )

    except Exception as exc:
        print(
            f"[ERROR] Inference failure: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="inference_error"
        ) from exc