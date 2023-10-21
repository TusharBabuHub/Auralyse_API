# syntax=docker/dockerfile:1.4
FROM --platform=$BUILDPLATFORM tensorflow/tensorflow:2.14.0

# Set the working directory in the container to /app
WORKDIR /

# Copy the current directory contents into the container at /app
ADD . /

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install the requirements
COPY requirements.txt /
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000/tcp

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]
