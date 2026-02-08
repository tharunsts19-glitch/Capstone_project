import numpy as np
import tensorflow as tf
from src.data.generator import generate_mock_patients, generate_drug_gene_graph, generate_clinical_notes
from src.data.preprocessing import DataPreprocessor
from src.models.recommender import TreatmentRecommender

def main():
    print("--- PD Precision Medicine Backend Pipeline ---")
    
    # 1. Data Generation
    print("[1/4] Generating Mock Data...")
    df_patients = generate_mock_patients(5)
    G = generate_drug_gene_graph()
    notes = generate_clinical_notes(5)
    
    # 2. Preprocessing
    print("[2/4] Preprocessing Data...")
    preprocessor = DataPreprocessor()
    
    patient_features = preprocessor.get_patient_features(df_patients)
    # Using dummy note embeddings for demonstration (768 matches DistilBERT output)
    note_embeddings = tf.random.normal((5, 768)) 
    adj, g_feats, node_map = preprocessor.preprocess_graph(G)
    
    # 3. Model Initialization
    print("[3/4] Initializing Recommender Model...")
    # Get number of clinical features from preprocessor output
    n_feat = patient_features.shape[1]
    recommender = TreatmentRecommender(num_clinical_features=n_feat)
    
    # 4. Inference / Recommendation
    print("[4/4] Running Recommendation Engine...")
    
    # Evaluate first drug in the graph for each patient
    drug_names = [n for n, d in G.nodes(data=True) if d['type'] == 'drug']
    if not drug_names:
        print("No drugs found in graph.")
        return
        
    target_drug = drug_names[0]
    drug_idx = node_map[target_drug]
    batch_drug_indices = tf.constant([drug_idx] * 5, dtype=tf.int32)
    
    # Prepare inputs
    inputs = [
        tf.constant(patient_features, dtype=tf.float32),
        tf.constant(note_embeddings, dtype=tf.float32),
        tf.constant(adj, dtype=tf.float32),
        tf.constant(g_feats, dtype=tf.float32),
        batch_drug_indices
    ]
    
    # Run model
    predictions = recommender(inputs)
    
    print("\nResults for Drug:", target_drug)
    for i, p_id in enumerate(df_patients['patient_id']):
        print(f"Patient {p_id} Success Probability: {predictions[i][0]:.2%}")

if __name__ == "__main__":
    main()
