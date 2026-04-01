# AI_NER Service

Backend service for PostTop music metadata NER. Includes dataset pipeline, model training/compilation, and a production Flask inference API.

## Stack

- Python 3.11
- Flask + Waitress
- Transformers (BERT tokenizer)
- NumPy + scikit-learn
- PostgreSQL (`psycopg2`) for dataset fetch
- Docker

## Requirements

- Python 3.11+
- pip
- PostgreSQL (only required for `fetch` pipeline step)

## Environment Variables

Create a `.env` file in the project root.

- `POSTGRES_DB` - PostgreSQL database name
- `POSTGRES_USER` - PostgreSQL user
- `POSTGRES_PASSWORD` - PostgreSQL password
- `POSTGRES_HOST` - PostgreSQL host
- `POSTGRES_PORT` - PostgreSQL port

## Run

```bash
pip install -r requirements.txt
python src/prod.py
```

Server starts on `http://localhost:5000`.

## Scripts

- `python src/cli.py fetch` - fetch dataset rows from PostgreSQL and save `dataset/videos.json`
- `python src/cli.py split` - split raw dataset into train/validation sets
- `python src/cli.py preprocess` - clean and normalize dataset text/labels
- `python src/cli.py tokenize` - tokenize samples for model input
- `python src/cli.py feature` - run feature extraction pipeline
- `python src/cli.py train` - train the NER model
- `python src/cli.py compile` - compile and export runtime model artifact
- `python src/cli.py fetch split preprocess tokenize feature train compile` - run full end-to-end pipeline

API endpoint:

- `POST /predict`

Request body:

```json
{
  "title": "Song title",
  "description": "Video description",
}
```

## Docker

```bash
docker build -t ai-ner .
docker run --rm -p 5000:5000 ai-ner
```

Or pull and run the published image:

```bash
docker pull ghcr.io/posttop/ai-ner:latest
docker run --rm -p 5000:5000 ghcr.io/posttop/ai-ner:latest
```