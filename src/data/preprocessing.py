import networkx as nx
import numpy as np
import tensorflow as tf
import pandas as pd
from transformers import DistilBertTokenizer

class DataPreprocessor:
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.node_to_idx = {}
        
    def preprocess_clinical_notes(self, notes, max_length=64):
        """
        Tokenizes clinical notes using Hugging Face tokenizer.
        """
        encodings = self.tokenizer(
            notes,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='tf'
        )
        return encodings

    def preprocess_graph(self, G):
        """
        Converts NetworkX graph to adjacency matrix and node features.
        Returns:
            adjacency_matrix (np.array)
            node_features (np.array): One-hot encoding of node types (drug vs gene).
            node_map (dict): Mapping from node name to index.
        """
        nodes = list(G.nodes(data=True))
        self.node_to_idx = {node[0]: i for i, node in enumerate(nodes)}
        
        num_nodes = len(nodes)
        adj = nx.to_numpy_array(G, nodelist=self.node_to_idx.keys())
        
        # Node features: [is_drug, is_gene]
        features = np.zeros((num_nodes, 2))
        for i, (node_name, data) in enumerate(nodes):
            if data.get('type') == 'drug':
                features[i] = [1, 0]
            else:
                features[i] = [0, 1]
                
        return adj, features, self.node_to_idx

    def get_patient_features(self, df):
        """
        Converts patient DataFrame to numerical features (Exactly 7: age, sex, updrs, 4 genotypes).
        """
        # Ensure we're working with a copy
        df_encoded = df.copy().reset_index(drop=True)
        
        # Build feature array manually to ensure correct shape
        n_samples = len(df_encoded)
        features = np.zeros((n_samples, 7), dtype=np.float32)
        
        # Column 0: Age
        features[:, 0] = df_encoded['age'].values.astype(np.float32)
        
        # Column 1: Sex (0=M, 1=F)
        features[:, 1] = df_encoded['sex'].map({'M': 0, 'F': 1}).values.astype(np.float32)
        
        # Column 2: UPDRS
        features[:, 2] = df_encoded['updrs_score'].values.astype(np.float32)
        
        # Columns 3-6: Genotype one-hot (GBA, LRRK2, SNCA, None)
        genotypes = ['GBA', 'LRRK2', 'SNCA', 'None']
        for i, geno in enumerate(genotypes):
            features[:, 3 + i] = (df_encoded['genotype'] == geno).astype(np.float32).values
        
        return features

if __name__ == "__main__":
    # Test Block
    import pandas as pd # Import pandas here for test block if needed, though used in method
    pass 
