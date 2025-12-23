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

## Le conflit HDBSCAN : topics vs outliers (12/2024)

**Le problème fondamental :**
- HDBSCAN a un conflit intrinsèque entre nombre de topics et taux d'outliers
- ↑ `min_cluster_size` → moins de topics MAIS plus d'outliers (petits groupes deviennent bruit)
- ↓ `min_cluster_size` → plus de topics MAIS moins d'outliers
- impossible d'avoir "peu de topics" ET "peu d'outliers" en tunant HDBSCAN seul

**Stratégie "Permissive → Reduce" :**

1. **Clustering permissif** : accepter beaucoup de micro-topics
   - `min_cluster_size` bas (10-15)
   - `min_samples` bas (3-5)
   - résultat : 300-500 topics, <10% outliers

2. **Réduction post-hoc** : `reduce_topics(nr_topics=15)`
   - BERTopic fusionne itérativement les topics les plus similaires (cosine sur c-TF-IDF)
   - résultat : 15 topics stables

3. **Réassignation des outliers** : `reduce_outliers(strategy="embeddings")`
   - pour chaque outlier, calcule similarité cosine avec les centroïdes des topics
   - si sim > threshold → assigne au topic le plus proche
   - résultat : outliers passent de 41% → 15%

**Pourquoi ça marche :**
- HDBSCAN fait ce qu'il sait faire : trouver la densité locale
- BERTopic fait ce qu'il sait faire : fusionner/organiser les topics
- on découple "détection de structure" de "granularité souhaitée"

---

## Évolution de la métrique de split (12/2024)

### Étape 1 : Silhouette (point de départ)

**Idée initiale :**
- silhouette score = métrique classique de qualité de cluster
- formule : `(b - a) / max(a, b)` où a = intra-cluster, b = inter-cluster
- si silhouette < 0.1 → cluster "fourre-tout" → candidat au split

**Problème découvert :**
- silhouette assume des clusters globulaires/convexes
- HDBSCAN trouve des clusters de densité variable, formes arbitraires
- silhouette pénalise les clusters allongés même s'ils sont cohérents
- pas adapté à notre use case

### Étape 2 : DBCV + Persistence (tentative)

**DBCV (Density-Based Cluster Validation) :**
- conçu pour clusters density-based
- mesure la "séparation relative de densité"
- fourni par `hdbscan.validity_index()`
- **problème** : donne un score GLOBAL, pas per-cluster

**Persistence (`cluster_persistence_`) :**
- mesure intrinsèque HDBSCAN : durée de vie du cluster dans la hiérarchie
- persistence haute = cluster stable
- persistence basse = cluster instable → candidat au split
- **disponible per-cluster** ✓

**Test empirique (DefendIntelligence, ~9000 comments) :**
```
Gros clusters "fourre-tout" :
  - "IA, désinformation et éthique" (2458 comments) : pers=0.69
  - "Appréciation des Explications" (2421 comments) : pers=0.82

Petits clusters précis :
  - "Calculs de pages et mots" (17 comments) : pers=0.12
  - "Activation of Windows Licenses" (16 comments) : pers=0.08
```

**Conclusion : persistence corrélée à la TAILLE, pas à la cohérence**
- les gros clusters ont plus de "masse" → survivent plus longtemps dans la hiérarchie
- les petits clusters précis ont basse persistence simplement parce qu'ils sont petits
- **inverser la logique** (splitter haute persistence) ne marche pas non plus

### Étape 3 : Mean Distance au centroïde (solution retenue)

**Intuition :**
- un cluster cohérent = points regroupés autour du centroïde
- un cluster fourre-tout = points dispersés loin du centroïde
- la taille n'a pas d'impact : 2500 comments peuvent être très proches si le sujet est homogène

**Calcul :**
```python
centroid = np.mean(embeddings, axis=0)
distances = np.linalg.norm(embeddings - centroid, axis=1)
mean_distance = np.mean(distances)
```

**Test empirique (même dataset) :**
```
Gros clusters "fourre-tout" :
  - "IA, désinformation et éthique" : dist=0.83 → SPLIT
  - "Appréciation des Explications" : dist=0.82 → SPLIT

Petits clusters précis :
  - "Calculs de pages et mots" : dist=0.54 → PROTÉGÉ
  - "Activation of Windows Licenses" : dist=0.69 → PROTÉGÉ
```

**Seuil retenu : `MEAN_DISTANCE_THRESHOLD = 0.75`**
- dist > 0.75 → cluster dispersé → split
- dist < 0.75 → cluster serré → protégé

**Pourquoi ça marche :**
- indépendant de la taille du cluster
- mesure directement ce qu'on veut : la dispersion dans l'espace sémantique
- interprétable (distance euclidienne dans l'espace d'embeddings normalisés)

---

## Paramètres HDBSCAN avancés

**`cluster_selection_method` :**
- `"eom"` (Excess of Mass) : sélectionne les clusters qui maximisent la "masse" totale
  - tend à donner des clusters plus stables
  - évite l'hyper-fragmentation
  - **recommandé pour topic modeling**
- `"leaf"` : sélectionne les feuilles de l'arbre hiérarchique
  - donne plus de petits clusters
  - utile si on veut une granularité très fine

**`cluster_selection_epsilon` :**
- distance en-dessous de laquelle les clusters sont fusionnés automatiquement
- `epsilon=0.0` : pas de fusion (défaut)
- `epsilon=0.05` : fusionne les clusters très proches
- utile pour éviter les micro-séparations dues au bruit

**Interaction avec UMAP :**
- HDBSCAN travaille sur l'espace réduit par UMAP
- la "densité" dépend de la projection UMAP
- `min_dist=0.0` dans UMAP → clusters très denses → HDBSCAN plus sensible
- modifier UMAP peut avoir plus d'impact que modifier HDBSCAN

---

