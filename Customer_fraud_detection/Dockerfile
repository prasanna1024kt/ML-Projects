FROM python:3.13-slim

# install build deps needed for some packages (lightgbm/xgboost) and pipenv runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libomp-dev git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy Pipfile & lock and install in-system
# copy requirements (generate locally)
COPY Pipfile Pipfile.lock ./
RUN pip install --upgrade pip setuptools wheel pipenv \
 && pipenv install --system --deploy --ignore-pipfile

# copy datasets (if present) and model code
# If datasets are large you may prefer to mount them with -v at runtime instead
COPY datasets/ ./datasets/
COPY model/ ./model/

EXPOSE 9696

# Use correct gunicorn bind flag and module path (module:app)
# Adjust "model.predict:app" to the actual module that defines `app`
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "model.predict:app"]