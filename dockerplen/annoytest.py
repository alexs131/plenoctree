import numpy as np
from annoy import AnnoyIndex

points = [
    np.array([0.65, 0.32, 0.71]),
    np.array([0.95, 0.37, 0.34]),
    np.array([0.17, -0.374, 0.324]),
]

points = np.stack(points)
print(points)
index = AnnoyIndex(3,'angular')
for i,v in enumerate(points):
        index.add_item(i,v)
index.build(10)
index.save("annoy.ann")

query = np.array([0.95, -0.30, 0.25])
print(query)

result = index.get_nns_by_vector(query, 1)
print(result)
index.load("annoy.ann")
result = index.get_nns_by_vector(query, 1)
print(result)
