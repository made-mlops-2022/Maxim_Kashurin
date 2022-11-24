FROM python:3.9

RUN mkdir -p /project && mkdir /project/model

WORKDIR /project

COPY requirements.txt /project
RUN pip install --no-cache-dir -r requirements.txt
COPY models/logregmodel.pkl /model/model.pkl
COPY inference.py /project/inference.py
ENV PORT 8080
ENV PATH_TO_MODEL "/model/model.pkl"

COPY . /project

CMD ["python", "inference.py"]