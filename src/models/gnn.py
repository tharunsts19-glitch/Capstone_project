import tensorflow as tf
from tensorflow.keras import layers, models

class GraphConvolution(layers.Layer):
    """
    Simple Graph Convolution Layer: H_new = Activation(A * H * W)
    """
    def __init__(self, units, activation=None):
        super(GraphConvolution, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        def extract_last_dim(s):
            # If it's a single integer or None, return it
            if isinstance(s, int):
                return s
            if s is None:
                return None
            # If it has as_list (TensorShape), convert to list
            if hasattr(s, 'as_list'):
                s = s.as_list()
            # If it's a list/tuple
            if isinstance(s, (list, tuple)):
                if len(s) == 0: return None
                # If the last element is an int, that's our dimension
                if isinstance(s[-1], int):
                    return s[-1]
                # Otherwise, recurse into the last element
                return extract_last_dim(s[-1])
            return s

        # For multiple inputs [adj, features], the features tensor is the second one
        if isinstance(input_shape, (list, tuple)) and len(input_shape) > 1:
            # Check if it's a list of shapes
            feat_shape = input_shape[1]
        else:
            feat_shape = input_shape
            
        input_dim = extract_last_dim(feat_shape)
        
        # Final safety check: if it's still not an int, it might be a 0-d or 1-d shape
        if not isinstance(input_dim, int):
            # If somehow we got a list/tuple as the dimension, take its last element
            while isinstance(input_dim, (list, tuple)):
                input_dim = input_dim[-1]
        
        self.w = self.add_weight(shape=(input_dim, self.units),
                                 initializer="random_normal",
                                 trainable=True)

    def call(self, inputs):
        # inputs: [adjacency_matrix, features]
        adj, features = inputs
        
        # Linear transform: H * W
        output = tf.matmul(features, self.w)
        
        # Graph convolution: A * (H * W)
        output = tf.matmul(adj, output)
        
        if self.activation:
            output = self.activation(output)
            
        return output

class DrugGeneGNN(models.Model):
    def __init__(self, hidden_units=32, embedding_dim=16):
        super(DrugGeneGNN, self).__init__()
        self.gc1 = GraphConvolution(hidden_units, activation='relu')
        self.gc2 = GraphConvolution(embedding_dim, activation='relu')
        
    def call(self, inputs):
        # inputs: [adjacency_matrix, node_features]
        x = self.gc1(inputs)
        x = self.gc2([inputs[0], x])
        return x # Node embeddings

if __name__ == "__main__":
    # Test Block
    import numpy as np
    
    adj = tf.constant(np.random.rand(10, 10).astype(np.float32))
    feats = tf.constant(np.random.rand(10, 5).astype(np.float32))
    
    gnn = DrugGeneGNN()
    embeddings = gnn([adj, feats])
    print("GNN Output Shape:", embeddings.shape)
