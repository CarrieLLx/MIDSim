# YuLan-OneSim 统一容器镜像
# 包含前端、后端、CLI 所有功能，开箱即用

FROM node:18-slim AS frontend-builder

# 构建前端
WORKDIR /frontend
COPY src/frontend/package*.json ./
RUN npm install
COPY src/frontend/ ./
RUN npm run build

# 主镜像
FROM python:3.10-slim

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    nginx \
    supervisor \
    curl \
    procps \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装Python依赖
COPY setup.py README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# 复制前端构建文件到nginx目录
COPY --from=frontend-builder /frontend/dist /var/www/html

# 复制配置文件
COPY config/ ./config/

# 创建必要目录
RUN mkdir -p /app/logs /var/log/supervisor

# 复制nginx配置
COPY docker/nginx.conf /etc/nginx/nginx.conf

# 复制supervisor配置
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# 创建启动脚本
COPY docker/start.sh /start.sh
RUN chmod +x /start.sh

# 暴露端口
EXPOSE 80

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost/health || exit 1

# 启动
CMD ["/start.sh"] 