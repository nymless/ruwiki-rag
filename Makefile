.PHONY: venv deps deps_manual app_init data_load data_unzip extract_docs extract_json qdrant_pull qdrant_run

venv:
	uv venv --python=3.13

deps:
	uv sync

deps_manual:
	uv pip install torch --index-url https://download.pytorch.org/whl/cu129
	uv pip install transformers==4.51.0 sentence-transformers==5.1.1 flash-attn langchain_community langchain_huggingface langchain_gigachat
	uv pip install langchain-qdrant qdrant-client datasets einops accelerate lxml jupyter wikiextractor

app_init:
	mkdir -p ./data/packed
	mkdir -p ./data/unpacked
	mkdir -p ./data/docs
	mkdir -p ./data/json

data_load:
	wget -O ./data/packed/ruwiki-latest-pages-articles.xml.bz2 https://dumps.wikimedia.org/ruwiki/latest/ruwiki-latest-pages-articles.xml.bz2

data_unzip:
	bunzip2 -v -v ./data/packed/ruwiki-latest-pages-articles.xml.bz2

extract_docs:
	python -m wikiextractor.WikiExtractor -o ./data/unpacked ./data/packed/ruwiki-latest-pages-articles.xml

extract_json:
	python -m wikiextractor.WikiExtractor -o ./data/unpacked --json ./data/packed/ruwiki-latest-pages-articles.xml

qdrant_pull:
	docker pull qdrant/qdrant

qdrant_run:
	docker run --rm -p 6333:6333 -p 6334:6334 -v "$(HOME)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant
