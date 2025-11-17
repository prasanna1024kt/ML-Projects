FROM python:3.12-slim

RUN pip install pipenv 

workdir /app

copy ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

RUN mkdir -p Datasets

COPY ["./Datasets/heart_failure_clinical_records.csv", "./Datasets/"]

RUN mkdir -p model

COPY ["./model/predict.py", "./model/heart_failure_model_xgboost.bin", "./model/"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind==0.0.0.0:9696", "predict:app"]
