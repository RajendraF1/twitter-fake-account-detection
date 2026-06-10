.PHONY: install run train streamlit api test

install:
	pip install -r requirements.txt

run:
	python -m scripts.run_pipeline

train:
	python -m scripts.train --all

streamlit:
	streamlit run app/streamlit_app.py

api:
	uvicorn app.api:app --reload

test:
	python -m pytest tests/ -v
