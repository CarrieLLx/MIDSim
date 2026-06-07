# YuLan-OneSim Makefile

IMAGE_NAME ?= ptss/yulan-onesim
CONTAINER_NAME ?= yulan-onesim
TAG ?= latest

.PHONY: help build run stop clean logs shell test

# 默认目标
help:
	@echo "🚀 YuLan-OneSim Docker 操作命令："
	@echo ""
	@echo "  build     - 构建Docker镜像"
	@echo "  run       - 启动容器"
	@echo "  stop      - 停止容器"
	@echo "  restart   - 重启容器"
	@echo "  logs      - 查看日志"
	@echo "  shell     - 进入容器Shell"
	@echo "  clean     - 清理容器和镜像"
	@echo "  test      - 测试容器是否正常工作"
	@echo ""
	@echo "使用示例："
	@echo "  make build && make run"

# 构建镜像
build:
	@echo "🔨 构建 YuLan-OneSim 镜像..."
	docker build -t $(IMAGE_NAME):$(TAG) .

# 启动容器
run:
	@echo "🚀 启动 YuLan-OneSim 容器..."
	docker run -d --name $(CONTAINER_NAME) -p 8000:80 -v ./config:/app/config $(IMAGE_NAME):$(TAG)
	@echo "✅ 容器已启动！"
	@echo "🌐 Web界面: http://localhost:8000"
	@echo "📚 API文档: http://localhost:8000/docs"

# 停止容器
stop:
	@echo "🛑 停止容器..."
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)
	@echo "✅ 容器已停止"

# 重启容器
restart: stop run

# 查看日志
logs:
	@echo "📄 查看容器日志..."
	docker logs -f $(CONTAINER_NAME)

# 进入Shell
shell:
	@echo "💻 进入容器Shell..."
	docker exec -it $(CONTAINER_NAME) bash

# 清理
clean: stop
	@echo "🧹 清理镜像..."
	-docker rmi $(IMAGE_NAME):$(TAG)
	@echo "✅ 清理完成"

# 测试
test:
	@echo "🧪 测试容器功能..."
	@sleep 5
	@if curl -f http://localhost:8000/health > /dev/null 2>&1; then \
		echo "✅ 容器运行正常"; \
	else \
		echo "❌ 容器运行异常"; \
		exit 1; \
	fi

# 一键部署（构建+运行+测试）
deploy: build run
	@echo "⏳ 等待服务启动..."
	@sleep 10
	@make test
	@echo "🎉 部署完成！" 