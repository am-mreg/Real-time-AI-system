.PHONY: help install lint test run docker-build docker-run

help:
	@echo "Available targets: install lint test run docker-build docker-run"

	install:
		python -m pip install --upgrade pip
			pip install -r requirements.txt

			lint:
				ruff src tests || true
					python -m pyflakes src || true

					test:
						pytest -q

						run:
							uvicorn src.app:app --host 0.0.0.0 --port 8000

							docker-build:
								docker build -t yolo-live:latest .

								docker-run:
									docker run --device /dev/video0 -p 8000:8000 -v $(PWD)/weights:/app/weights yolo-live:latest