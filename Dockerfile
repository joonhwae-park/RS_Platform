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

RUN micromamba run -n p5 pip install --no-cache-dir \
      fastapi "uvicorn[standard]" supabase \
      peft==0.6.2 transformers==4.36.2 accelerate==0.24.1 sentencepiece==0.1.96 scikit-learn==1.6.1

RUN git clone https://github.com/joonhwae-park/P5_mod.git /workspace/P5-main
ENV P5_ROOT=/workspace/P5-main
WORKDIR /workspace

# Copy application code
COPY app.py /workspace/app.py

# Copy t5-small model files to HuggingFace cache location
# This prevents runtime downloads from HuggingFace
RUN mkdir -p /root/.cache/huggingface/hub/models--google-t5--t5-small/snapshots/default
COPY models/t5-small/ /root/.cache/huggingface/hub/models--google-t5--t5-small/snapshots/default/

# Cloud Run Port
EXPOSE 8080

ENTRYPOINT ["micromamba","run","-n","p5","uvicorn","app:app","--host","0.0.0.0","--port","8080","--log-level","info"]
