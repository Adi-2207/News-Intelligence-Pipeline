#!/usr/bin/env python3
"""
Main pipeline script for News Intelligence System
Runs the complete pipeline from data ingestion to API serving
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion import DataIngestion
from src.preprocessing import TextPreprocessor
from src.embeddings import EmbeddingGenerator
from src.clustering import ArticleClusterer
from src.categorization import ArticleCategorizer
from src.evaluation import PipelineEvaluator
from src.api import create_api_server

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NewsIntelligencePipeline:
    """End-to-end news intelligence pipeline"""
    
    def __init__(self, config_path='config/config.yaml'):
        """Initialize pipeline with configuration"""
        self.config = self._load_config(config_path)
        self.data_ingestion = DataIngestion()
        self.preprocessor = TextPreprocessor()
        self.embedding_generator = EmbeddingGenerator(
            model_name=self.config.get('embedding_model', 'all-MiniLM-L6-v2')
        )
        self.clusterer = ArticleClusterer()
        self.categorizer = ArticleCategorizer()
        self.evaluator = PipelineEvaluator()
        
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except:
            # Default configuration
            return {
                'sample_size': 1000,
                'embedding_model': 'all-MiniLM-L6-v2',
                'min_cluster_size': 20,
                'categories': [
                    'Politics', 'Business', 'Technology', 'Sports',
                    'Health', 'Entertainment', 'Science', 'World News',
                    'Environment', 'Education', 'Lifestyle'
                ]
            }
    
    def run(self, input_path, output_dir='output', sample_size=None):
        """Run complete pipeline"""
        logger.info("="*60)
        logger.info("NEWS INTELLIGENCE PIPELINE")
        logger.info("="*60)
        
        start_time = datetime.now()
        
        try:
            # 1. Data Ingestion
            logger.info("Step 1: Data Ingestion")
            df = self.data_ingestion.load_data(
                input_path, 
                sample_size=sample_size or self.config.get('sample_size', 1000)
            )
            logger.info(f"Loaded {len(df)} articles")
            
            # 2. Preprocessing
            logger.info("\nStep 2: Text Preprocessing")
            df = self.preprocessor.preprocess(df)
            logger.info(f"After preprocessing: {len(df)} articles")
            
            # 3. Embeddings
            logger.info("\nStep 3: Generating Embeddings")
            embeddings = self.embedding_generator.generate(df)
            logger.info(f"Generated embeddings: {embeddings.shape}")
            
            # 4. Clustering
            logger.info("\nStep 4: Clustering Articles")
            df, cluster_metrics = self.clusterer.cluster(df, embeddings)
            logger.info(f"Created {cluster_metrics['n_clusters']} clusters")
            
            # 5. Categorization
            logger.info("\nStep 5: Categorizing Clusters")
            df = self.categorizer.categorize(df, categories=self.config.get('categories'))
            logger.info(f"Assigned {df['category'].nunique()} categories")
            
            # 6. Evaluation
            logger.info("\nStep 6: Evaluating Pipeline")
            metrics = self.evaluator.evaluate(df, embeddings)
            
            # 7. Save Results
            logger.info("\nStep 7: Saving Results")
            self._save_results(df, embeddings, metrics, output_dir)
            
            # Summary
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info("\n" + "="*60)
            logger.info("PIPELINE COMPLETE")
            logger.info("="*60)
            logger.info(f"Time elapsed: {elapsed:.1f} seconds")
            logger.info(f"Articles processed: {len(df)}")
            logger.info(f"Clusters created: {metrics['clustering']['n_clusters']}")
            logger.info(f"Categories assigned: {metrics['categorization']['n_categories']}")
            logger.info(f"Silhouette score: {metrics['clustering']['silhouette_score']:.3f}")
            
            return df, embeddings, metrics
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _save_results(self, df, embeddings, metrics, output_dir):
        """Save all pipeline outputs"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save processed data
        df.to_pickle(os.path.join(output_dir, 'processed_articles.pkl'))
        df.to_csv(os.path.join(output_dir, 'processed_articles.csv'), index=False)
        
        # Save embeddings
        np.save(os.path.join(output_dir, 'embeddings.npy'), embeddings)
        
        # Save metrics
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create summary report
        self._create_summary_report(df, metrics, output_dir)
        
        logger.info(f"Results saved to {output_dir}/")
    
    def _create_summary_report(self, df, metrics, output_dir):
        """Create human-readable summary report"""
        report_path = os.path.join(output_dir, 'summary_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("NEWS INTELLIGENCE PIPELINE - SUMMARY REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("CONFIGURATION\n")
            f.write("-"*30 + "\n")
            f.write(f"Embedding Model: {self.config.get('embedding_model')}\n")
            f.write(f"Sample Size: {len(df)}\n\n")
            
            f.write("RESULTS\n")
            f.write("-"*30 + "\n")
            f.write(f"Total Articles: {metrics['data']['total_articles']}\n")
            f.write(f"Clusters: {metrics['clustering']['n_clusters']}\n")
            f.write(f"Categories: {metrics['categorization']['n_categories']}\n")
            f.write(f"Silhouette Score: {metrics['clustering']['silhouette_score']:.3f}\n\n")
            
            f.write("CATEGORY DISTRIBUTION\n")
            f.write("-"*30 + "\n")
            for cat, count in metrics['categorization']['distribution'].items():
                percentage = count / metrics['data']['total_articles'] * 100
                f.write(f"{cat}: {count} ({percentage:.1f}%)\n")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='News Intelligence Pipeline - Process and categorize news articles'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--output', '-o',
        default='output',
        help='Output directory (default: output)'
    )
    parser.add_argument(
        '--sample', '-s',
        type=int,
        help='Number of articles to sample'
    )
    parser.add_argument(
        '--config', '-c',
        default='config/config.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--serve-api',
        action='store_true',
        help='Start REST API server after processing'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = NewsIntelligencePipeline(config_path=args.config)
    df, embeddings, metrics = pipeline.run(
        input_path=args.input,
        output_dir=args.output,
        sample_size=args.sample
    )
    
    # Optionally start API server
    if args.serve_api:
        logger.info("\nStarting REST API server...")
        app = create_api_server(args.output)
        app.run(debug=True, port=5000)


if __name__ == '__main__':
    main()
