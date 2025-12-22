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

## Commentaires représentatifs

- par défaut BERTopic prend les premiers commentaires du cluster (pas forcément représentatifs)
- solution : centroïde+MMR sur les commentaires aussi
- calculer similarité de chaque commentaire au centroïde du cluster
- prendre top 20 candidats, puis MMR pour diversifier
- résultat : commentaires typiques ET diversifiés (pas 5x le même truc reformulé)

---

## Détection automatique des stop words

**Problème initial :**
- lister manuellement les stop words FR/EN = fastidieux et incomplet
- approche IDF pure = sensible au déséquilibre linguistique du corpus
- si 90% anglais, 10% français → les stop words français ont IDF plus élevé que certains mots de contenu anglais

**Solution en deux couches :**

1. **NLTK comme base** : charger les stop words connus pour FR, EN, ES, DE, PT, IT
   - filet de sécurité pour les classiques (de, la, le, the, is, are...)

2. **Entropie inter-cluster** : détection des stop words corpus-spécifiques
   - clustering rapide (K-means, 10 clusters)
   - pour chaque mot, calculer sa distribution à travers les clusters
   - entropie haute = distribution uniforme = stop word
   - entropie basse = concentré dans certains clusters = mot de contenu

**Raffinement max_ratio :**
- problème : "vidéo" peut être partout (entropie haute) MAIS avec un pic dans un cluster "montage"
- solution : calculer max_ratio = max(cluster_freq) / mean(cluster_freq)
- si max_ratio > 2.0 → le mot a un pic significatif → on le "sauve"
- stop word = entropie haute ET max_ratio bas

**Robustesse multi-langue :**
- l'approche entropie fonctionne si chaque langue est distribuée sur plusieurs thématiques
- si une langue est concentrée dans un seul sujet → ses stop words ne seront pas détectés
- NLTK couvre ce cas comme filet de sécurité

**Script :** `detect_stopwords.py` → `cache/detected_stopwords.json`

---

## Simplifications

- KeyBERTInspired retiré : on utilise notre propre centroïde+MMR pour les top words
- plus besoin du re-ranking TF-IDF, on va direct au sémantique
- code refactorisé dans `lib/` : config, cache, mmr, data
- `extract_topics.py` reste lisible (~300 lignes)

---

