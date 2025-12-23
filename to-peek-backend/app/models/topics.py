"""
Pydantic models for topic extraction.
"""
from pydantic import BaseModel


class TopicExtractionParams(BaseModel):
    """Parameters for topic extraction."""
    sample_size: int = 1000
    
    # UMAP params
    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    umap_min_dist: float = 0.0
    
    # HDBSCAN params
    hdbscan_min_cluster_size: int = 50
    hdbscan_min_samples: int = 10
    
    # BERTopic params
    min_topic_size: int = 5
    nr_topics: int | None = None


class TopicWordDetail(BaseModel):
    """Word with similarity score."""
    word: str
    similarity: float


class TopicCommentDetail(BaseModel):
    """Comment with similarity score."""
    comment: str
    similarity: float


class TopicInfo(BaseModel):
    """Information about a single topic."""
    id: int
    count: int
    persistence: float | None = None
    variance: float | None = None
    max_distance: float | None = None
    mean_distance: float | None = None
    top_words_centroid_mmr: list[str] = []
    top_words_centroid_mmr_detail: list[TopicWordDetail] = []
    example_comments_original: list[str] = []
    example_comments_centroid_mmr: list[str] = []
    example_comments_centroid_mmr_detail: list[TopicCommentDetail] = []
    generated_name: str | None = None
    
    # Hierarchical structure
    depth: int = 0
    parent_id: int | str | None = None
    parent_name: str | None = None
    is_hierarchical: bool = False
    children: list["TopicInfo"] = []


class TopicExtractionResult(BaseModel):
    """Result of topic extraction."""
    generated_at: str
    sample_size: int
    num_topics: int
    params: dict
    topics: list[TopicInfo]
    filename: str | None = None


class HierarchicalTopicResult(BaseModel):
    """Result of hierarchical topic extraction."""
    generated_at: str
    pipeline: str = "hierarchical"
    config: dict
    num_topics: int
    num_hierarchical: int
    num_subtopics: int
    topics: list[TopicInfo]
    filename: str | None = None

