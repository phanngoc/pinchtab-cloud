# Pinchtab Cloud SG — Backend image.
#
# Runs the FastAPI control plane. Talks to a `pinchtab` daemon over a docker
# internal network (resolved via service DNS, never bound to host).
#
# Build: docker compose build backend
# Run:   docker compose up -d
#
# Layer caching: deps first, code last.

FROM python:3.12-slim AS deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install only deps first so the layer stays cached across code edits.
COPY pyproject.toml ./
RUN pip install \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.32.0" \
    "sqlalchemy>=2.0.36" \
    "alembic>=1.14.0" \
    "pydantic>=2.10.0" \
    "pydantic-settings>=2.7.0" \
    "httpx>=0.28.0" \
    "itsdangerous>=2.2.0" \
    "stripe>=11.4.0" \
    "python-dotenv>=1.0.1" \
    "tldextract>=5.1.3" \
    "limits>=4.0.0" \
    "anthropic>=0.45.0" \
    "pyyaml>=6.0" \
    "python-multipart>=0.0.20"

# ---- runtime image ----
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --shell /bin/false app

WORKDIR /app

# Copy installed site-packages from the deps stage.
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Application code.
COPY --chown=app:app backend ./backend
COPY --chown=app:app core ./core
COPY --chown=app:app pyproject.toml ./

USER app

EXPOSE 8000

# Healthcheck pings the control plane.
HEALTHCHECK --interval=20s --timeout=3s --retries=3 --start-period=10s \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).status==200 else 1)"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
