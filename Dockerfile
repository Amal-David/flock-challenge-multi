FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    bubblewrap \
    build-essential \
    ca-certificates \
    clang \
    cmake \
    curl \
    git \
    jq \
    libssl-dev \
    pkg-config \
    python3 \
    tini \
    util-linux \
    && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    PATH=/opt/cargo/bin:${PATH}

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain 1.97.0 \
    && rustc --version \
    && ldd --version | head -n 1

WORKDIR /workspace

COPY scripts/render_job_entrypoint.sh /usr/local/bin/flock-render-job
RUN chmod 0755 /usr/local/bin/flock-render-job

CMD ["sleep", "infinity"]
