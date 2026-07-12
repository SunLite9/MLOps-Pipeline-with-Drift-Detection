FROM apache/airflow:2.9.3-python3.11

# Installed without Airflow's own constraints file: requirements.txt pulls in
# newer mlflow/pandas/numpy than Airflow 2.9.3 was tested against, and the
# constraint file would force a version conflict rather than resolve one.
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

ENV PYTHONPATH="/opt/airflow/project:${PYTHONPATH}"
ENV PYTHONIOENCODING=utf-8
