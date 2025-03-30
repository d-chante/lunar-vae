FROM python:3.11-slim

WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt

RUN apt-get update && apt-get install -y bash
RUN python -m ensurepip --upgrade \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

CMD ["bash"]
