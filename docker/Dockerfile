FROM ubuntu:22.04 AS base

ARG HOST_UID=1000
ARG HOST_GID=1000

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

# System dependencies
RUN apt-get update && apt-get -y dist-upgrade && apt-get install -y --no-install-recommends \
    apt-utils \
    bash \
    bash-completion \
    bc \
    build-essential \
    bzip2 \
    ca-certificates \
    cmake \
    curl \
    dialog \
    diffutils \
    findutils \
    git \
    gnupg \
    gosu \
    gnupg2 \
    gpgsm \
    hostname \
    iproute2 \
    iputils-ping \
    keyutils \
    language-pack-en \
    less \
    libcap2-bin \
    libegl-mesa0 \
    libegl1 \
    libegl1-mesa \
    libgl1 \
    libgl1-mesa-glx \
    libglx-mesa0 \
    libglu1-mesa \
    libglx0 \
    libglib2.0-bin \
    libkrb5-3 \
    libnss-mdns \
    libnss-myhostname \
    libnss-sss \
    libsm6 \
    libx11-6 \
    libxext6 \
    libxft2 \
    libxrender1 \
    libxt6 \
    libvte-2.91-common \
    libvte-common \
    libvulkan1 \
    locales \
    lsof \
    man-db \
    manpages \
    mesa-vulkan-drivers \
    mtr \
    ncurses-base \
    netbase \
    openssh-client \
    passwd \
    pigz \
    pinentry-curses \
    procps \
    quota \
    rsync \
    sudo \
    tcpdump \
    time \
    traceroute \
    tree \
    tzdata \
    unzip \
    util-linux \
    vulkan-tools \
    wget \
    xauth \
    xz-utils \
    zip \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh

ENV PATH=${CONDA_DIR}/bin:$PATH

# Accept Anaconda channel ToS (required for non-interactive builds)
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create conda env
RUN conda create -y -n sim python=3.11 && conda clean -afy
ENV CONDA_DEFAULT_ENV=sim
ENV PATH=${CONDA_DIR}/envs/sim/bin:$PATH
ENV LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/opt/conda/envs/sim/lib
ENV ISAACSIM_PATH=/workspace/isaacsim
ENV PYTHONPATH=${ISAACSIM_PATH}
ENV VIRTUAL_ENV=/opt/venv
ENV OMNI_VENV_REQUIREMENTS=/opt/venv_requirements.txt
ENV PATH=${VIRTUAL_ENV}/bin:$PATH
ENV HOST_UID=${HOST_UID}
ENV HOST_GID=${HOST_GID}

WORKDIR /workspace

RUN cat > /usr/local/bin/omni_entrypoint.sh <<'SH'
#!/usr/bin/env bash
set -e

echo "[entrypoint] starting as uid=$(id -u) gid=$(id -g) HOME=${HOME}"

export ISAACSIM_PATH="${ISAACSIM_PATH:-/workspace/isaacsim}"
export PYTHONPATH="${ISAACSIM_PATH}:${PYTHONPATH:-}"
export VIRTUAL_ENV="${VIRTUAL_ENV:-/opt/venv}"
export OMNI_VENV_REQUIREMENTS="${OMNI_VENV_REQUIREMENTS:-/opt/venv_requirements.txt}"

if [ ! -f "${VIRTUAL_ENV}/bin/activate" ]; then
  echo "Bootstrapping venv at ${VIRTUAL_ENV}..."
  /opt/conda/envs/sim/bin/python -m venv "${VIRTUAL_ENV}"
  # shellcheck disable=SC1091
  source "${VIRTUAL_ENV}/bin/activate"
  if [ -f "${OMNI_VENV_REQUIREMENTS}" ]; then
    pip install --no-cache-dir -r "${OMNI_VENV_REQUIREMENTS}"
  else
    echo "WARNING: ${OMNI_VENV_REQUIREMENTS} not found; venv is empty." >&2
  fi
fi

if [ "$(id -u)" -eq 0 ] && [ -n "${HOST_UID:-}" ] && [ -n "${HOST_GID:-}" ] && [ -d "${VIRTUAL_ENV}" ]; then
  current_uid="$(stat -c %u "${VIRTUAL_ENV}")"
  current_gid="$(stat -c %g "${VIRTUAL_ENV}")"
  if [ "$current_uid" != "$HOST_UID" ] || [ "$current_gid" != "$HOST_GID" ]; then
    echo "Fixing ${VIRTUAL_ENV} ownership to ${HOST_UID}:${HOST_GID}..."
    chown -R "${HOST_UID}:${HOST_GID}" "${VIRTUAL_ENV}"
  fi
fi

if [ -f "${VIRTUAL_ENV}/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${VIRTUAL_ENV}/bin/activate"
  echo "Activated venv: ${VIRTUAL_ENV}"
else
  echo "No venv found at ${VIRTUAL_ENV}."
  echo "Run /workspace/omni_drones/docker/install to create it and install Python deps."
fi

if [ -f "${ISAACSIM_PATH}/setup_conda_env.sh" ]; then
  source "${ISAACSIM_PATH}/setup_conda_env.sh"
fi

# Source .bashrc to pick up user env vars and functions (suppress errors since
# isaac sim / venv may not be installed yet on first run)
if [ -f "${HOME}/.bashrc" ]; then
  set +e
  # shellcheck disable=SC1091
  source "${HOME}/.bashrc" || true
  set -e
else
  echo "ERROR: ${HOME}/.bashrc not found. Is the .home directory mounted correctly?" >&2
  exit 1
fi

# Drop from root to the host user so file ownership matches
if [ "$(id -u)" -eq 0 ] && [ -n "${HOST_UID:-}" ] && [ -n "${HOST_GID:-}" ]; then
  # If the host runs as root, keep root (already correct)
  if [ "${HOST_UID}" -eq 0 ] && [ "${HOST_GID}" -eq 0 ]; then
    exec "$@"
  fi
  # Ensure the target user/group exist, creating them if needed
  if ! getent group "${HOST_GID}" > /dev/null 2>&1; then
    groupadd -g "${HOST_GID}" hostgroup
  fi
  if ! getent passwd "${HOST_UID}" > /dev/null 2>&1; then
    # Register /workspace/omni_drones as home so passwd and $HOME agree
    useradd -u "${HOST_UID}" -g "${HOST_GID}" -d /workspace/omni_drones -M -s /bin/bash hostuser
  fi

  # Debug: log state before privilege drop
  echo "[entrypoint] dropping privileges: uid=${HOST_UID} gid=${HOST_GID}"
  echo "[entrypoint] HOME=${HOME}"
  echo "[entrypoint] .bashrc exists at HOME: $([ -f "${HOME}/.bashrc" ] && echo YES || echo NO)"
  echo "[entrypoint] passwd home for uid ${HOST_UID}: $(getent passwd "${HOST_UID}" | cut -d: -f6)"

  # Explicitly carry HOME through gosu so /etc/profile.d scripts cannot override it
  exec gosu "${HOST_UID}:${HOST_GID}" env HOME=/workspace/omni_drones "$@"
fi

exec "$@"
SH
RUN chmod +x /usr/local/bin/omni_entrypoint.sh
ENTRYPOINT ["/usr/local/bin/omni_entrypoint.sh"]
CMD ["bash"]
VOLUME ["/opt/venv"]

FROM base AS dev

FROM base AS eecs

COPY --from=isaacsim / /workspace/isaacsim
COPY docker/install /tmp/omni_install.sh

RUN --mount=type=ssh \
    chmod +x /tmp/omni_install.sh && \
    /tmp/omni_install.sh && \
    rm -f /tmp/omni_install.sh
