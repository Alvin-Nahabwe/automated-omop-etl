# syntax=docker/dockerfile:1.7
# omop-agent: the pipeline command line and the review web app, in one image.
# Built and run by deploy/compose.yaml; see docs/deploy.md.
#
# Behind a proxy that inspects TLS, pass its CA certificate:
#   docker build --secret id=extra_ca,src=/path/to/ca.crt --build-arg HTTPS_PROXY=... .
# If Docker Hub limits your pulls, use a mirror of the same image:
#   --build-arg PYTHON_IMAGE=mirror.gcr.io/library/python:3.12-slim
ARG PYTHON_IMAGE=python:3.12-slim

FROM ${PYTHON_IMAGE} AS build
WORKDIR /src
COPY pyproject.toml README.md ./
COPY omop_agent ./omop_agent
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -s /run/secrets/extra_ca ]; then export PIP_CERT=/run/secrets/extra_ca; fi; \
    pip wheel --no-cache-dir --wheel-dir /wheels ".[web]"

FROM ${PYTHON_IMAGE}
LABEL org.opencontainers.image.title="omop-agent" \
      org.opencontainers.image.description="AI agent for automated OMOP ETL, with a review web app" \
      org.opencontainers.image.licenses="Apache-2.0"
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMOP_AGENT_PROJECTS=/srv/projects \
    OMOP_AGENT_DATA=/srv/data
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links /wheels "omop-agent[web]" \
    && rm -rf /wheels \
    && useradd --create-home --uid 10001 omop \
    && mkdir -p /srv/projects /srv/data /srv/athena /srv/source \
    && chown omop:omop /srv/projects /srv/data /srv/athena
WORKDIR /srv
USER omop
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request as u; u.build_opener(u.ProxyHandler({})).open('http://127.0.0.1:8000/healthz', timeout=4)"]
ENTRYPOINT ["omop-agent"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
