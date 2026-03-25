.PHONY: build up down logs clean restart

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

clean:
	docker-compose down -v
	rm -rf orchestrator/app/assets/confirmations/*.wav

restart:
	docker-compose restart

health:
	@echo "Checking service health..."
	@curl -s http://localhost:8001/health || echo "Wakeword: DOWN"
	@curl -s http://localhost:8002/health || echo "STT: DOWN"
	@curl -s http://localhost:8003/health || echo "LLM: DOWN"
	@curl -s http://localhost:8004/health || echo "TTS: DOWN"
