FROM python:3.11.4

WORKDIR /CarBrandClass

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /CarBrandClass/requirements.txt

COPY ./app /CarBrandClass/app
COPY ./model /CarBrandClass/model

ENV PYTHONPATH "${PYTHONPATH}:/CarBrandClass"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]