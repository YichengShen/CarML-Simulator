from collections import defaultdict, Counter
from math import ceil, sqrt
import xml.etree.ElementTree as ET
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
# tree = ET.parse("MonacoST/most_fcd.xml") 
tree = ET.parse("osm_boston_common/osm_fcd.xml") 
root = tree.getroot()
for timestep in root:
    # if float(timestep.attrib['time']) % 10 == 0:
    for vehicle in timestep.findall('vehicle'):
        x.append(float(vehicle.attrib['x'])*100)
        y.append(float(vehicle.attrib['y'])*100)
        coord.append([float(vehicle.attrib['x']), float(vehicle.attrib['y'])])

# Print the location of each vehicle in each time step in a scatter plot
# plt.scatter(x, y, s=0.05)
# plt.show()

# The value of eps and min_samples determines how each cluster is formed
# Higher min_samples or lower eps indicate higher density necessary to form a cluster
# More can be read about them on https://scikit-learn.org/stable/modules/clustering.html#dbscan and https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN
db = DBSCAN(eps=35, min_samples=2000).fit(coord)
labels = db.labels_ # The labels has the same shape as coord, each index i of label tells which cluster index i of coord is in
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# Print the density of each cluster
# Note: The cluster with the key -1 are noise (outliers) and we don't care about it
dicc = Counter(labels)
order = sorted(dicc, key=dicc.get, reverse=True)
print('number of clusters: ', n_clusters_)
print('density of each cluster', dicc)

# Plot the new scatter plot with each cluster colored
dic = defaultdict(lambda: defaultdict(list))
for i, x in enumerate(labels):
    dic[x]['x'].append(coord[i][0])
    dic[x]['y'].append(coord[i][1])

def largestN(n):
    return order[1:n+1]

largest_dic = {}
largest = largestN(4)
for key, value in dic.items():
    if key in largest and key != -1:
        largest_dic[key] = value
        plt.scatter(value['x'], value['y'], s=0.05, c='black')
    else:
        plt.scatter(value['x'], value['y'], s=0.05)
# plt.show()

# Find the center of a cluster
def find_center(value):
    num_points = len(value['x'])
    center_x = sum(value['x']) / num_points
    center_y = sum(value['y']) / num_points
    return center_x, center_y

center_dic = {}
for key, value in largest_dic.items():
    center = find_center(value)
    center_dic[key] = center

print('center of each cluster: ', center_dic)

# Load Junctions
tree = ET.parse("osm_boston_common/osm.net.xml")
root = tree.getroot()
junction_list = root.findall('junction')

closest_junction_distance = {}
junction_dic= {}
for key in center_dic.keys():
    closest_junction_distance[key] = 99999
    junction_dic[key] = None

# Loop through the junctions and find junctions that are closest to each cluster center
for junction in junction_list:
    for key, (center_x, center_y) in center_dic.items():
        distance = sqrt((float(junction.attrib['x']) - center_x) ** 2 + (float(junction.attrib['y']) - center_y) ** 2)
        if distance < closest_junction_distance[key]:
            closest_junction_distance[key] = distance
            junction_dic[key] = junction

for key, junction in junction_dic.items():
    print("Cluster:", key, "Coord:", (float(junction.attrib['x']), float(junction.attrib['y'])), "Traffic Density:", dicc[key])

