# Runtime for the alerting demo stack.
#
# Installs the repo's declared dependencies but NOT mlflow itself -- the repo is
# mounted at /mlflow/home and put on PYTHONPATH, so the container always runs the
# working tree rather than a snapshot baked at build time.
FROM python:3.11

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# psycopg2 for Postgres; huey is declared by the project but listed here too so a
# dependency reshuffle cannot silently disable the job scheduler.
RUN uv pip install --system psycopg2-binary "huey<4,>=2.5.4"

COPY pyproject.toml /tmp/pyproject.toml
RUN python -c "\
import tomllib;\
d = tomllib.load(open('/tmp/pyproject.toml','rb'));\
print('\n'.join(d['project']['dependencies']))" > /tmp/requirements.txt \
    && uv pip install --system -r /tmp/requirements.txt

WORKDIR /mlflow/home
ENV PYTHONPATH=/mlflow/home
ENV PYTHONUNBUFFERED=1
