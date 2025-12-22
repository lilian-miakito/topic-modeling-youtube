#!/usr/bin/env python3
"""
Visualize topics extracted by BERTopic.
Generates interactive HTML visualizations of the topic embeddings.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

from bertopic import BERTopic

DATASETS_DIR = Path(__file__).parent / "datasets"
OUTPUT_DIR = Path(__file__).parent / "visualizations"


def load_model_and_docs():
    """Load the saved BERTopic model and documents."""
    model_dir = DATASETS_DIR / "bertopic_model"
    docs_file = DATASETS_DIR / "bertopic_docs.json"
    
    if not model_dir.exists():
        print(f"Model not found: {model_dir}")
        print("Run extract_topics.py first!")
        return None, None, None
    
    if not docs_file.exists():
        print(f"Documents not found: {docs_file}")
        print("Run extract_topics.py first!")
        return None, None, None
    
    print("Loading BERTopic model...")
    topic_model = BERTopic.load(model_dir)
    
    print("Loading documents...")
    with open(docs_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = data["documents"]
    topics = data["topics"]
    
    return topic_model, documents, topics


def main():
    parser = argparse.ArgumentParser(description="Visualize BERTopic results")
    parser.add_argument("--type", choices=["all", "documents", "topics", "hierarchy", "heatmap", "barchart"],
                        default="all", help="Type of visualization to generate")
    args = parser.parse_args()
    
    print("=" * 60)
    print("BERTopic Visualization")
    print("=" * 60)
    
    # Load model and docs
    topic_model, documents, topics = load_model_and_docs()
    
    if topic_model is None:
        return
    
    print(f"\nLoaded {len(documents):,} documents with {len(set(topics)) - 1} topics")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    visualizations = []
    
    # 1. Document embeddings visualization (the "galaxy")
    if args.type in ["all", "documents"]:
        print("\n1. Generating document embeddings visualization (galaxy)...")
        try:
            fig_docs = topic_model.visualize_documents(
                documents,
                topics=topics,
                hide_annotations=False,
                hide_document_hover=False,
                width=1200,
                height=800
            )
            doc_file = OUTPUT_DIR / f"galaxy_{timestamp}.html"
            fig_docs.write_html(doc_file)
            print(f"   Saved: {doc_file}")
            visualizations.append(("Galaxy (Documents)", doc_file))
        except Exception as e:
            print(f"   Error: {e}")
            print("   Trying with reduced embeddings...")
            try:
                # Fallback: reduce embeddings first
                embeddings = topic_model._extract_embeddings(documents, method="document")
                fig_docs = topic_model.visualize_documents(
                    documents,
                    topics=topics,
                    embeddings=embeddings,
                    hide_annotations=False,
                    width=1200,
                    height=800
                )
                doc_file = OUTPUT_DIR / f"galaxy_{timestamp}.html"
                fig_docs.write_html(doc_file)
                print(f"   Saved: {doc_file}")
                visualizations.append(("Galaxy (Documents)", doc_file))
            except Exception as e2:
                print(f"   Could not generate document visualization: {e2}")
    
    # 2. Topic visualization (inter-topic distance)
    if args.type in ["all", "topics"]:
        print("\n2. Generating topic distance visualization...")
        try:
            fig_topics = topic_model.visualize_topics(
                width=1000,
                height=800
            )
            topics_file = OUTPUT_DIR / f"topics_distance_{timestamp}.html"
            fig_topics.write_html(topics_file)
            print(f"   Saved: {topics_file}")
            visualizations.append(("Topic Distance", topics_file))
        except Exception as e:
            print(f"   Could not generate topic visualization: {e}")
    
    # 3. Hierarchical clustering
    if args.type in ["all", "hierarchy"]:
        print("\n3. Generating hierarchy visualization...")
        try:
            fig_hierarchy = topic_model.visualize_hierarchy(
                width=1000,
                height=600
            )
            hierarchy_file = OUTPUT_DIR / f"hierarchy_{timestamp}.html"
            fig_hierarchy.write_html(hierarchy_file)
            print(f"   Saved: {hierarchy_file}")
            visualizations.append(("Hierarchy", hierarchy_file))
        except Exception as e:
            print(f"   Could not generate hierarchy visualization: {e}")
    
    # 4. Topic similarity heatmap
    if args.type in ["all", "heatmap"]:
        print("\n4. Generating topic heatmap...")
        try:
            fig_heatmap = topic_model.visualize_heatmap(
                width=800,
                height=800
            )
            heatmap_file = OUTPUT_DIR / f"heatmap_{timestamp}.html"
            fig_heatmap.write_html(heatmap_file)
            print(f"   Saved: {heatmap_file}")
            visualizations.append(("Heatmap", heatmap_file))
        except Exception as e:
            print(f"   Could not generate heatmap: {e}")
    
    # 5. Bar chart of top words per topic
    if args.type in ["all", "barchart"]:
        print("\n5. Generating bar chart of top words...")
        try:
            fig_barchart = topic_model.visualize_barchart(
                top_n_topics=15,
                n_words=8,
                width=400,
                height=300
            )
            barchart_file = OUTPUT_DIR / f"barchart_{timestamp}.html"
            fig_barchart.write_html(barchart_file)
            print(f"   Saved: {barchart_file}")
            visualizations.append(("Bar Chart", barchart_file))
        except Exception as e:
            print(f"   Could not generate bar chart: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("VISUALIZATION SUMMARY")
    print("=" * 60)
    
    if visualizations:
        print(f"\nGenerated {len(visualizations)} visualization(s):")
        for name, path in visualizations:
            print(f"  • {name}: {path}")
        
        print(f"\nOpen in browser: file://{visualizations[0][1].absolute()}")
    else:
        print("\nNo visualizations were generated.")


if __name__ == "__main__":
    main()

