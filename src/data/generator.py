import pandas as pd
import numpy as np
import networkx as nx
import random

# Set stable seed for consistent patient generation
random.seed(42)
np.random.seed(42)

def generate_mock_patients(n=100):
    """
    Generates mock patient data. Includes 20 specific patients from requested dataset.
    """
    # Specific patients from requested spreadsheet
    fixed_patients = [
        ["P000", "Joseph Brown", "8520-4067-4796", 81, "F", 46, "None", "joseph.garcia77@gmail.com"],
        ["P001", "Elizabeth Garcia", "6465-6500-1791", 55, "F", 37, "None", "william.miller10@gmail.com"],
        ["P002", "Robert Rodriguez", "8624-8260-2433", 61, "F", 45, "None", "patricia.rodriguez41@gmail.com"],
        ["P003", "John Brown", "8014-9974-5334", 84, "M", 23, "None", "elizabeth.smith22@gmail.com"],
        ["P004", "Patricia Rodrigue", "9324-6834-7261", 76, "M", 18, "None", "joseph.williams46@gmail.com"],
        ["P005", "Mary Williams", "3069-4502-5611", 51, "M", 24, "None", "jennifer.williams90@gmail.com"],
        ["P006", "Elizabeth Garcia", "5335-6317-4868", 65, "F", 45, "GBA", "elizabeth.miller85@gmail.com"],
        ["P007", "Elizabeth Garcia", "5658-5005-7806", 69, "F", 25, "None", "william.miller32@gmail.com"],
        ["P008", "Jennifer Davis", "9082-5337-5020", 79, "M", 17, "None", "jane.jones33@gmail.com"],
        ["P009", "Mary Jones", "2010-9168-7318", 87, "F", 20, "None", "william.johnson53@gmail.com"],
        ["P010", "Richard Miller", "4182-9306-6051", 80, "F", 33, "LRRK2", "jennifer.johnson88@gmail.com"],
        ["P011", "William Williams", "3098-5935-2090", 76, "F", 17, "None", "mary.miller46@gmail.com"],
        ["P012", "Patricia Davis", "5428-3188-9100", 77, "F", 22, "None", "william.martinez9@gmail.com"],
        ["P013", "Robert Garcia", "3648-8201-3293", 53, "M", 42, "None", "william.garcia63@gmail.com"],
        ["P014", "Elizabeth Brown", "3123-2847-2534", 56, "M", 13, "None", "richard.jones2@gmail.com"],
        ["P015", "Jane Garcia", "9163-7090-8474", 74, "F", 16, "LRRK2", "mary.johnson83@gmail.com"],
        ["P016", "Joseph Martinez", "5106-8243-5431", 83, "F", 18, "None", "jennifer.williams30@gmail.com"],
        ["P017", "Joseph Martinez", "5775-1939-8154", 57, "M", 14, "None", "joseph.johnson96@gmail.com"],
        ["P018", "Elizabeth Davis", "2789-4116-7249", 83, "F", 33, "None", "jennifer.smith20@gmail.com"],
        ["P019", "John Williams", "9262-7632-9964", 75, "F", 14, "None", "jane.miller96@gmail.com"]
    ]
    
    df_fixed = pd.DataFrame(fixed_patients, columns=['patient_id', 'patient_name', 'aadhar_number', 'age', 'sex', 'updrs_score', 'genotype', 'email'])
    
    if n <= 20:
        return df_fixed.iloc[:n]
    
    # Generate remaining if needed
    n_rem = n - 20
    first_names = ['John', 'Jane', 'Robert', 'Mary', 'William', 'Patricia', 'Richard', 'Jennifer', 'Joseph', 'Elizabeth']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']
    
    names = [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(n_rem)]
    data_rem = {
        'patient_id': [f'P{i:03d}' for i in range(20, n)],
        'patient_name': names,
        'aadhar_number': [f"{random.randint(1000, 9999)}-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}" for _ in range(n_rem)],
        'age': np.random.randint(50, 90, n_rem),
        'sex': np.random.choice(['M', 'F'], n_rem),
        'updrs_score': np.random.randint(10, 50, n_rem),
        'genotype': np.random.choice(['LRRK2', 'GBA', 'SNCA', 'None'], n_rem, p=[0.1, 0.1, 0.05, 0.75]),
        'email': [f"{name.lower().replace(' ', '.')}{random.randint(1, 99)}@gmail.com" for name in names]
    }
    df_rem = pd.DataFrame(data_rem)
    return pd.concat([df_fixed, df_rem]).reset_index(drop=True)

def generate_drug_gene_graph():
    """
    Generates a mock drug-gene interaction graph.
    """
    G = nx.Graph()
    
    genes = ['LRRK2', 'GBA', 'SNCA', 'PARK2', 'PINK1']
    drugs = ['Levodopa', 'Carbidopa', 'Entacapone', 'Rasagiline', 'Amantadine']
    
    G.add_nodes_from(genes, type='gene')
    G.add_nodes_from(drugs, type='drug')
    
    # Add random edges (interactions)
    for drug in drugs:
        for gene in genes:
            if random.random() > 0.6:
                G.add_edge(drug, gene, weight=random.random())
                
    return G

def generate_clinical_notes(n=100):
    """
    Generates mock clinical notes.
    """
    notes = []
    templates = [
        "Patient shows signs of tremor in right hand. Responding well to {drug}.",
        "Bradykinesia observed. Suggest increasing dosage of {drug}.",
        "Complains of dyskinesia. Genomic markers indicate risk with {drug}.",
        "Stable condition. Continue current regimen of {drug}."
    ]
    drugs = ['Levodopa', 'Carbidopa', 'Entacapone']
    
    for _ in range(n):
        drug = random.choice(drugs)
        template = random.choice(templates)
        notes.append(template.format(drug=drug))
        
    return notes

if __name__ == "__main__":
    df = generate_mock_patients(10)
    print("Mock Patients:")
    print(df.head())
    
    G = generate_drug_gene_graph()
    print(f"\nGraph Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    notes = generate_clinical_notes(5)
    print("\nMock Clinical Notes:")
    for note in notes:
        print(f"- {note}")
