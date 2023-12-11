FROM python:3.11-slim

WORKDIR /usr/src/app

COPY *.py .

# Adjust dataset name and path, if required
COPY datasets/cop/all_data.csv datasets/cop/all_data.csv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

VOLUME ["/output"]

CMD ["python", "-u","/usr/src/app/main.py"]