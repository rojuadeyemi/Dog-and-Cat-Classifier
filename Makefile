SHELL := /bin/bash

.PHONY: all deploy develop

all: requirements.txt
	@if not exist .venv\Scripts ( \
		python -m venv .venv \
	)
	@if exist .venv\Scripts ( \
		.venv\Scripts\activate && pip install -r requirements.txt \
	) else ( \
		source .venv/bin/activate && pip install -r requirements.txt \
	)

develop:
	@if exist .venv\Scripts\activate ( \
		.venv\Scripts\activate && \
		set FLASK_APP=app.py && \
		set FLASK_ENV=development && \
		set FLASK_DEBUG=1 && \
		flask run \
	) else \
		( \
		source .venv/bin/activate && \
		export FLASK_APP=app.py && \
		export FLASK_ENV=development && \
		export FLASK_DEBUG=1 && \
		flask run \
	)

deploy:
	@if exist .venv\Scripts\activate ( \
		.venv\Scripts\activate \
		&& \
		waitress-serve --listen 0.0.0.0:5000 app:app \
	) else ( \
		source .venv/bin/activate \
				&& \
		waitress-serve --listen 0.0.0.0:5000 app:app \
	)
