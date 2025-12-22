# Idées & Découvertes

---

- on commence par une stratégie EDA (Exploratory Data Analysis)

- logique hiérarchique : démarrer avec peu de topics, trouver ceux avec grosse couverture dimensionnelle, rentrer dedans pour jouer sur la granularité

- l'idée c'est qu'il y a un côté hiérarchique à la data, on peut sous-découper mais uniquement sur le subset qu'on a choisi

- utiliser silhouette par cluster + variance intra-cluster pour détecter les topics "fourre-tout"
- si un cluster a silhouette basse (<0.1) et variance haute → candidat pour re-clustering itératif
- workflow : extraire les docs de ce cluster, relancer BERTopic dessus avec params plus fins

- c-TF-IDF (top words par défaut de BERTopic) = fréquentiel, même avec KeyBERTInspired qui re-ranke juste les candidats TF-IDF
- solution : approche centroïde → vocabulaire = embedding du centroïde du cluster, puis cosine similarity avec tous les mots du vocab
- ça donne des mots vraiment sémantiquement proches du topic, pas juste fréquents

- problème : mots similaires dans les top words (vidéo, vidéos, video, videos)
- solution : MMR (Maximal Marginal Relevance) pour diversifier
- MMR_LAMBDA = 0.7 → 70% pertinence, 30% diversité
- sélection itérative : pénalise les mots trop proches de ceux déjà sélectionnés

---

## Paramètres choisis (sweep 22/12/2024)

- Run 30 sélectionné parmi 72 configs testées
- UMAP: n_neighbors=15, n_components=5, min_dist=0.0
- HDBSCAN: min_cluster_size=50, min_samples=10
- Résultat: 6 topics, 38% outliers, silhouette 0.471
- Pourquoi: meilleur compromis entre nombre de topics exploitables (pas 2-3 inutiles, pas 40+ fragmentés) et qualité des clusters (silhouette élevé pour cette plage)
- Alternative si besoin plus granulaire: Run 20 (22 topics, 52% outliers, silhouette 0.489) avec min_cluster_size=10

---

## Notes techniques

- le nombre de clusters scale avec la taille du dataset
- min_cluster_size=10 sur 1k docs → ~100 clusters max
- min_cluster_size=10 sur 100k docs → ~10k clusters max
- solutions : min_cluster_size proportionnel (1% du dataset) ou nr_topics fixe

---

## Optimisations performance

**cProfile révèle les bottlenecks :**
- ~20s imports de modules (inévitable, one-time)
- ~10s UMAP fit_transform (cœur de l'algo)
- ~8s Numba JIT compilation (one-time par session)
- ~5-6s embedding des documents

**MMR optimisé :**
- pré-filtrer top 100 candidats avant MMR (au lieu de tout le vocab)
- pré-calculer matrice de similarité 100×100 une seule fois
- lookup vectorisé au lieu de boucle Python avec cosine_similarity à chaque itération

**Cache embeddings vocabulaire :**
- `modeling/cache/vocab_embeddings.parquet`
- clé = mot/ngram, valeur = embedding 384D
- encode seulement les nouveaux mots, lookup pour les autres
- grandit au fil des exécutions

**Cache embeddings commentaires :**
- `modeling/cache/comments_embeddings.parquet`
- clé = MD5 hash du texte, valeur = embedding 384D
- passer embeddings pré-calculés à BERTopic via `embeddings=` param
- IMPORTANT : passer aussi `embedding_model=` explicitement sinon KeyBERTInspired crash (il a besoin du modèle pour encoder les mots candidats)

**Autres :**
- `TOKENIZERS_PARALLELISM=false` pour éviter warnings HuggingFace sur fork
- batch_size=128 pour encoder vocab (plus rapide que un par un)

---

