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

RUN micromamba run -n p5 python -m pip install --upgrade pip && \
    micromamba run -n p5 pip install \
      --index-url https://download.pytorch.org/whl/cu117 \
      torch==2.0.1+cu117 torchvision==0.15.2+cu117

RUN git clone https://github.com/joonhwae-park/P5_mod.git /workspace/P5-main
ENV P5_ROOT=/workspace/P5-main
WORKDIR /workspace

RUN pip install --no-cache-dir fastapi uvicorn[standard] supabase

COPY app.py /workspace/app.py

# Cloud Run Port
EXPOSE 8080

ENTRYPOINT ["micromamba","run","-n","p5","--no-capture-output","uvicorn","app:app","--host","0.0.0.0","--port","8080"]
