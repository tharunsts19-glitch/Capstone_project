# Genomics-Driven Personalized Treatment Recommendation System for Parkinson’s Disease

## Overview
This project implements a precision medicine platform that integrates genomic and clinical data to provide personalized treatment recommendations for Parkinson’s Disease (PD) patients. It utilizes Graph Neural Networks (GNNs) to model drug-gene interactions and deep learning for predicting treatment success probabilities.

## Key Features
- **Multi-modal Integration:** Combines patient demographic data, genotypes, and clinical notes.
- **Drug-Gene Interaction Modeling:** Uses GNNs to capture complex relationships between genomic markers and pharmacological treatments.
- **Explainable AI (XAI):** Provides insights into why specific treatments were recommended.
- **Clinician Dashboard:** Interactive Streamlit-based interface for decision support.

## Tech Stack
- **Languages:** Python
- **Frameworks:** TensorFlow, HuggingFace (Transformers), NetworkX
- **Frontend:** Streamlit
- **Deployment:** Docker, Docker-Compose

## Project Structure
```
.
├── src/
│   ├── app/           # Streamlit Frontend
│   ├── data/          # Data generation and preprocessing
│   └── models/         # GNN and Recommender model architectures
├── Dockerfile         # App containerization
├── docker-compose.yml # Service orchestration
└── requirements.txt   # Python dependencies
```

## Getting Started

### Local Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model (optional but recommended for optimized results):
   ```bash
   python src/train.py
   ```
3. Run the application:
   ```bash
   streamlit run src/app/app.py
   ```

### Docker Setup
1. Build and run:
   ```bash
   docker-compose up --build
   ```
2. Access the dashboard at `http://localhost:8501`.

## AI Model Architecture
The system uses a fusion model:
1. **Clinical Features:** Processed via Dense layers.
2. **Clinical Notes:** Tokenized using DistilBERT.
3. **Genomic Interactions:** Embedded using a multi-layer Graph Convolutional Network (GCN).
4. **Fusion Layer:** Concatenates all embeddings to predict success probability for a target drug.
