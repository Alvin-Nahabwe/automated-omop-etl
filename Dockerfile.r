# syntax=docker/dockerfile:1.7
# omop-agent with R, Achilles and the official OHDSI DataQualityDashboard (quality.use_r_dqd).
# Same command line and web app as Dockerfile, on Ubuntu 24.04 where the R recipe is tested; about
# 1 GB larger. deploy/compose.yaml builds it when OMOP_AGENT_DOCKERFILE=Dockerfile.r (docs/deploy.md).
#
# Behind a proxy that inspects TLS, pass its CA certificate:
#   docker build -f Dockerfile.r --secret id=extra_ca,src=/path/to/ca.crt --build-arg HTTPS_PROXY=... .
ARG UBUNTU_IMAGE=ubuntu:24.04
FROM ${UBUNTU_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/opt/omop-agent/bin:$PATH \
    OMOP_AGENT_PROJECTS=/srv/projects \
    OMOP_AGENT_DATA=/srv/data \
    DATABASECONNECTOR_JAR_FOLDER=/opt/jdbc \
    OMOP_AGENT_USE_R_DQD=true \
    OMOP_AGENT_REQUIRE_R_DQD=true

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# R, Java, the PostgreSQL JDBC driver and the OHDSI packages. The compilers are only needed for the
# few CRAN packages built from source, so they are removed afterwards.
COPY deploy/r/install_r_dqd.sh /tmp/install_r_dqd.sh
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -s /run/secrets/extra_ca ]; then export CURL_CA_BUNDLE=/run/secrets/extra_ca; fi; \
    sh /tmp/install_r_dqd.sh \
    && apt-get purge -y --auto-remove r-base-dev \
    && rm -rf /var/lib/apt/lists/* /tmp/install_r_dqd.sh /tmp/Rtmp* \
    && Rscript -e 'stopifnot(all(vapply(c("DatabaseConnector","Achilles","DataQualityDashboard"), requireNamespace, logical(1), quietly=TRUE)))'

WORKDIR /src
COPY pyproject.toml README.md ./
COPY omop_agent ./omop_agent
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -s /run/secrets/extra_ca ]; then export PIP_CERT=/run/secrets/extra_ca; fi; \
    python3 -m venv /opt/omop-agent \
    && pip install --no-cache-dir ".[web]" \
    && rm -rf /src \
    && useradd --create-home --uid 10001 omop \
    && mkdir -p /srv/projects /srv/data /srv/athena /srv/source \
    && chown omop:omop /srv/projects /srv/data /srv/athena

LABEL org.opencontainers.image.title="omop-agent (R)" \
      org.opencontainers.image.description="AI agent for automated OMOP ETL, with the review web app, Achilles and the official DataQualityDashboard" \
      org.opencontainers.image.licenses="Apache-2.0"
WORKDIR /srv
USER omop
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python3", "-c", "import urllib.request as u; u.build_opener(u.ProxyHandler({})).open('http://127.0.0.1:8000/healthz', timeout=4)"]
ENTRYPOINT ["omop-agent"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
