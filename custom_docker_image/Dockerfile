FROM python:3

COPY requirements.txt /tmp
RUN /usr/local/bin/pip install --no-cache-dir -r /tmp/requirements.txt \
        && rm /tmp/requirements.txt

ENTRYPOINT ["/usr/local/bin/python"]