FROM tiangolo/uwsgi-nginx-flask:python3.7

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

RUN ls
COPY ./app /app
COPY ./data /data