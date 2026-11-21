# Built by the scorer on linux/amd64. A match runs it with no network, a read-only root
# (writable /tmp), 1 CPU, 512 MB, as uid 65534, with the map mounted at /arena.
FROM python:3.13.5-slim
ENV PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir numpy==2.2.6
WORKDIR /bot
COPY bot/ /bot/bot/
RUN python3 -m compileall -q /bot/bot && python3 -c "import numpy; print('numpy', numpy.__version__)"
ENTRYPOINT ["python3", "-u", "/bot/bot/main.py"]
