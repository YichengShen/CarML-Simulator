from collections import defaultdict, Counter
import xml.etree.ElementTree as ET
from math import ceil, sqrt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Load boundaries
tree = ET.parse("osm_boston_common/osm.net.xml")
root = tree.getroot()
bounds = root[0].attrib["convBoundary"].split(",")
x_min = ceil(float(bounds[0]))
y_min = ceil(float(bounds[1]))
x_max = ceil(float(bounds[2]))
y_max = ceil(float(bounds[3]))
print("X-range:", x_min, x_max, "\nY-range:", y_min, y_max, "\n")

x = []
y = []
coord = []
# Calculate traffic on each (x,y)
tree = ET.parse("osm_boston_common/osm_fcd.xml")
root = tree.getroot()
for timestep in root:
    for vehicle in timestep.findall('vehicle'):
        x.append(float(vehicle.attrib['x'])*100)
        y.append(float(vehicle.attrib['y'])*100)
        coord.append([float(vehicle.attrib['x']), float(vehicle.attrib['y'])])

# plt.scatter(x, y, s=0.05)
# plt.show()

db = DBSCAN(eps=10, min_samples=300).fit(coord)
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

dicc = Counter(labels)

print('number of clusters: ', n_clusters_)
print(dicc)

dic = defaultdict(lambda: defaultdict(list))
for i, x in enumerate(labels):
    dic[x]['x'].append(coord[i][0])
    dic[x]['y'].append(coord[i][1])

for i in dic.values():
    plt.scatter(i['x'], i['y'], s=0.05)
plt.show()

# Load Junctions
tree = ET.parse("osm_boston_common/osm.net.xml")
root = tree.getroot()
junction_list = root.findall('junction')

