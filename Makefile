.PHONY: up down logs check test
up:
	docker compose up --build -d --wait
down:
	docker compose down
logs:
	docker compose logs -f --tail=100
check:
	docker compose config --quiet
test:
	python -m unittest discover -s tests -v
