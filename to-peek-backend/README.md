# To-Peek Backend

Backend FastAPI pour l'analyse de topics dans les commentaires YouTube.

## Installation

```bash
cd to-peek-backend

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Ou avec les extras ML
pip install -e ".[ml]"
```

## Lancement

```bash
# Mode développement avec hot-reload
uvicorn app.main:app --reload --port 8000

# Ou directement
python -m app.main
```

## API Endpoints

### Datasets
- `GET /api/v1/datasets/stats` - Statistiques du dataset
- `POST /api/v1/datasets/create` - Créer le dataset depuis les données
- `GET /api/v1/datasets/sample?n=100` - Échantillon de commentaires

### Channels
- `GET /api/v1/channels/` - Liste des chaînes
- `GET /api/v1/channels/summary` - Résumé avec totaux
- `GET /api/v1/channels/{folder}` - Détails d'une chaîne
- `GET /api/v1/channels/{folder}/videos` - Vidéos d'une chaîne

### Topics
- `GET /api/v1/topics/` - Derniers résultats d'extraction
- `GET /api/v1/topics/hierarchical` - Résultats hiérarchiques
- `GET /api/v1/topics/files` - Liste des fichiers de résultats
- `GET /api/v1/topics/stopwords` - Stopwords détectés
- `GET /api/v1/topics/visualizations` - Visualisations disponibles

## Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

