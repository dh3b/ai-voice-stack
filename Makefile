.PHONY: build up down logs clean restart health

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
	rm -rf orchestrator/assets/confirmations/*.wav

restart:
	docker-compose restart

health:
	@echo "Checking service health..."
	@curl -sf http://localhost:8001/health && echo " Wakeword: OK" || echo " Wakeword: DOWN"
	@curl -sf http://localhost:8002/health && echo " STT: OK"      || echo " STT: DOWN"
	@curl -sf http://localhost:8003/health && echo " LLM: OK"      || echo " LLM: DOWN"
	@curl -sf http://localhost:8004/health && echo " TTS: OK"      || echo " TTS: DOWN"
