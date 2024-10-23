import numpy as np
def cosine_distance_matrix(data):
    # Normalize each row to unit length (L2 norm = 1)
    data_norm = data / np.linalg.norm(data, axis=1, keepdims=True)
    
    # Compute the cosine similarity matrix
    cosine_similarity = np.dot(data_norm, data_norm.T)
    
    # Convert cosine similarity to cosine distance
    cosine_distance = 1 - cosine_similarity
    
    return np.mean(cosine_distance)
a=np.load('gs_plus.npy')
print(cosine_distance_matrix(a))
a=np.load('gs.npy')
print(cosine_distance_matrix(a))
a=np.load('randn.npy')
print(cosine_distance_matrix(a))
a=np.load('baseline{args.test_mode}.npy')
print(cosine_distance_matrix(a))

