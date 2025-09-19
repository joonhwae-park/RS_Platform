FROM mambaorg/micromamba:1.5.8
USER root

RUN apt-get update && apt-get install -y \
    git build-essential cmake ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY p5.yml /tmp/p5.yml
RUN sed -i '/^prefix:/d' /tmp/p5.yml

# p5 yml 
RUN micromamba create -y -n p5 -f /tmp/p5.yml && \
    micromamba clean -a -y

ENV MAMBA_DOCKERFILE_ACTIVATE=1
SHELL ["/bin/bash", "-lc"]

RUN git clone https://github.com/jeykigung/P5.git /workspace/P5-main
ENV P5_ROOT=/workspace/P5-main
WORKDIR /workspace

RUN pip install --no-cache-dir fastapi uvicorn[standard] supabase

COPY app.py /workspace/app.py

# Cloud Run Port
EXPOSE 8080

ENTRYPOINT ["micromamba","run","-n","detic","--no-capture-output","uvicorn","app:app","--host","0.0.0.0","--port","8080"]
