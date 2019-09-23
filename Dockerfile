FROM python:3.7.4

ARG APP_ROOT="/app"
WORKDIR ${APP_ROOT}

RUN \
    apt-get update && \
    apt-get install -y python-opengl xvfb ffmpeg

COPY ./requirements.txt $APP_ROOT/requirements.txt
RUN \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    git clone https://github.com/MultiAgentLearning/playground && \
    pip install ./playground && \
    rm -rf ./playground

ENV TF_XLA_FLAGS "--tf_xla_cpu_global_jit $APP_ROOT"
ENV TF_CPP_MIN_LOG_LEVEL 2

COPY . $APP_ROOT

# CMD ["jupyter", "notebook", "--allow-root", "--ip=0.0.0.0", "--port=8888"]
