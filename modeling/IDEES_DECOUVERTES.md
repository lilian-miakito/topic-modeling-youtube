# Idées & Découvertes

---

- on commence par une stratégie EDA (Exploratory Data Analysis)

- logique hiérarchique : démarrer avec peu de topics, trouver ceux avec grosse couverture dimensionnelle, rentrer dedans pour jouer sur la granularité

- l'idée c'est qu'il y a un côté hiérarchique à la data, on peut sous-découper mais uniquement sur le subset qu'on a choisi

- utiliser silhouette par cluster + variance intra-cluster pour détecter les topics "fourre-tout"
- si un cluster a silhouette basse (<0.1) et variance haute → candidat pour re-clustering itératif
- workflow : extraire les docs de ce cluster, relancer BERTopic dessus avec params plus fins

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

