import puffinn
import numpy as np
import pickle

points = [
    np.array([0.65, 0.32, 0.71]),
    np.array([0.95, 0.37, 0.34]),
    np.array([0.17, -0.374, 0.324]),
]

points = np.stack(points)
print(points)
index = puffinn.Index('angular', dimensions, 4*1024**3)
for v in points:
        index.insert(v)
index.rebuild()

query = np.array([0.64, 0.30, 0.65])
print(query)

result = index.search(query, 1, 0.8)
print(result)

pickle.dump(index, open("puffinndata.pkl", "wb"))
