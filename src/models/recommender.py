import tensorflow as tf
from tensorflow.keras import layers, models
from .gnn import DrugGeneGNN

class TreatmentRecommender(models.Model):
    def __init__(self, num_clinical_features=7, gnn_hidden=32, embedding_dim=16):
        super(TreatmentRecommender, self).__init__()
        
        # Sub-modules
        self.gnn = DrugGeneGNN(hidden_units=gnn_hidden, embedding_dim=embedding_dim)
        
        # Patient Feature Processing
        self.patient_dense = layers.Dense(32, activation='relu')
        
        # Note Processing (Simulating Pre-computed Embeddings or simple processing)
        # In a full flow, we'd fine-tune DistilBERT here, but for simplicity/speed 
        # we will assume we pass pooled CLS token embeddings.
        self.note_dense = layers.Dense(32, activation='relu')
        
        # Fusion Layer
        self.concat = layers.Concatenate()
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(1, activation='sigmoid') # Probability of success
        
    def call(self, inputs):
        """
        inputs:
            patient_features: [batch_size, num_clinical_features]
            note_embeddings: [batch_size, 768] (DistilBERT output)
            gnn_graph_data: [adj, node_features] (For GNN extraction)
            drug_indices: [batch_size] (Indices of drugs to score for each patient)
        """
        patient_features, note_embeddings, gnn_adj, gnn_feats, drug_indices = inputs
        
        # 1. GNN Embeddings
        # Note: GNN usually runs on the whole graph. We get all node embeddings.
        node_embeddings = self.gnn([gnn_adj, gnn_feats]) 
        
        # Gather specific drug embeddings for the batch
        # This assumes drug_indices maps to the graph node rows
        batch_drug_embeddings = tf.gather(node_embeddings, drug_indices)
        
        # 2. Patient Features
        p_out = self.patient_dense(patient_features)
        
        # 3. Clinical Notes
        n_out = self.note_dense(note_embeddings)
        
        # 4. Fusion
        combined = self.concat([p_out, n_out, batch_drug_embeddings])
        x = self.fc1(combined)
        output = self.fc2(x)
        
        return output

    def test_run(self):
        """
        Simple method to verify the model builds and runs.
        """
        import numpy as np
        
        # Mock Inputs
        batch_size = 5
        n_feat = 7
        n_note_emb = 768
        n_nodes = 15
        
        p_feats = tf.random.normal((batch_size, n_feat))
        n_emb = tf.random.normal((batch_size, n_note_emb))
        adj = tf.random.uniform((n_nodes, n_nodes))
        g_feats = tf.random.normal((n_nodes, 2))
        drug_idx = tf.random.uniform((batch_size,), maxval=n_nodes, dtype=tf.int32)
        
        out = self.call([p_feats, n_emb, adj, g_feats, drug_idx])
        return f"Model ran successfully. Output shape: {out.shape}"
