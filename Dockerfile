FROM python:3.12

WORKDIR /opt

RUN apt-get update && apt-get install -y sudo libhdf5-dev

RUN pip install --upgrade pip

COPY requirements.txt /opt/
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /work

COPY appfile /work/appfile

CMD ["streamlit", "run", "appfile/pages/main.py", "--server.port", "8888"]