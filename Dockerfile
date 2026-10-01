FROM python:3.11-slim

WORKDIR /app

# 核心仅 numpy/scipy/scikit-learn；纯 CPU，无 GPU 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

# 默认跑端到端 benchmark
CMD ["python", "-m", "robustforge.cli", "run", "--out", "benchmark.json"]
