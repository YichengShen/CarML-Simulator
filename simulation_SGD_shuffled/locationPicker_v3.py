from collections import defaultdict, Counter
from math import ceil, sqrt
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import hdbscan
import yaml

file = open('config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)

x = []
y = []
coord = []
num_points = 0 # number of rows in fcd file
# Calculate traffic on each (x,y)
# tree = ET.parse("MonacoST/most_fcd.xml")
tree = ET.parse(cfg['simulation']['FCD_FILE'])
root = tree.getroot()
for timestep in root:
    if float(timestep.attrib['time']) % 10 == 0:
        for vehicle in timestep.findall('vehicle'):
            x.append(float(vehicle.attrib['x'])*100)
            y.append(float(vehicle.attrib['y'])*100)
            coord.append([float(vehicle.attrib['x']), float(vehicle.attrib['y'])])
            num_points += 1

# Print the location of each vehicle in each time step in a scatter plot
# plt.scatter(x, y, s=0.05)
# plt.show()

# The value of eps and min_samples determines how each cluster is formed
# Higher min_samples or lower eps indicate higher density necessary to form a cluster
# More can be read about them on https://scikit-learn.org/stable/modules/clustering.html#dbscan and https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN
min_samples = round(num_points * 0.002) # 0.1% of traffic points
eps = 20 # the initial eps value
n_clusters_ = -1
num_rsu = cfg['simulation']['num_rsu']
# Try different eps until we find enough clusters (>= number of rsu we want to place)
# If this fails, we need to lower our RSU number
# while n_clusters_ < num_rsu:
#     print("Testing DBSCAN with min_samples={} and eps={}".format(min_samples, eps))
#     db = DBSCAN(eps=eps, min_samples=min_samples).fit(coord)
#     labels = db.labels_ # The labels has the same shape as coord, each index i of label tells which cluster index i of coord is in
#     n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#     eps += 5
#     # When increasing eps no longer contributes to getting more clusters, break
#     if eps > 100 or n_clusters_ == 1:
#         print("The number of RSUs is much larger than the number of clusters that can be formed.")
#         break

clusterer = hdbscan.HDBSCAN(min_cluster_size=int(num_points*0.0025))
labels = clusterer.fit_predict(coord)

# Print the density of each cluster
# Note: The cluster with the key -1 are noise (outliers) and we don't care about it
dicc = Counter(labels)
order = sorted(dicc, key=dicc.get, reverse=True)
# print('number of clusters: ', n_clusters_)
# print('density of each cluster', dicc)

# Plot the new scatter plot with each cluster colored
dic = defaultdict(lambda: defaultdict(list))
for i, x in enumerate(labels):
    dic[x]['x'].append(coord[i][0])
    dic[x]['y'].append(coord[i][1])

def largestN(n):
    return order[1:n+1]

largest_dic = {}
largest = largestN(num_rsu)
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

# Center_dic contains all clusters with their center as the value
center_dic = {}
for key, value in dic.items():
    center = find_center(value)
    center_dic[key] = center

# print('center of each cluster: ', center_dic)

# Load Junctions
tree = ET.parse(cfg['simulation']['NET_FILE'])
root = tree.getroot()
junction_list = root.findall('junction')


def intersection_area(d, R, r):
    """ Return the area of intersection of two circles.
    The circles have radii R and r, and their centres are separated by d.
    https://scipython.com/book/chapter-8-scipy/problems/p84/overlapping-circles/
    """
    if d <= abs(R-r):
        # One circle is entirely enclosed in the other.
        return np.pi * min(R, r)**2
    if d >= r + R:
        # The circles don't overlap at all.
        return 0
    r2, R2, d2 = r**2, R**2, d**2
    alpha = np.arccos((d2 + r2 - R2) / (2*d*r))
    beta = np.arccos((d2 + R2 - r2) / (2*d*R))
    return r2 * alpha + R2 * beta - 0.5 * (r2 * np.sin(2*alpha) + R2 * np.sin(2*beta))


closest_junction_distance = {}
junction_dic= {}
for key in center_dic.keys():
    closest_junction_distance[key] = 99999
    junction_dic[key] = None

rsu_counter = num_rsu
rsu_range = cfg['comm_range']['v2rsu'] # Later, this should match RSU range defined in our simulator

# Loop starting the densest cluster
for key in order:
    # Break if finished picking all RSUs
    if rsu_counter <= 0:
        break
    if key != -1: # -1 is the noise
        for junction in junction_list:
            overlap = False
            # Compare the current junction with every RSU that has been picked out already
            for junc in junction_dic.values():
                if junc is not None:
                    # d -> distance between two junctions(potential RSU) -> distance between 2 circles
                    d = sqrt((float(junc.attrib['x']) - float(junction.attrib['x'])) ** 2 + (float(junc.attrib['y']) - float(junction.attrib['y'])) ** 2)
                    area_intersect = intersection_area(d, rsu_range, rsu_range)
                    ratio_intersect = area_intersect / (rsu_range**2 * np.pi)
                    if ratio_intersect >= 0.8: # the allowed ratio of overlaping
                        overlap = True
            # If the current junction passes the overlap test, then update junction_dic
            if not overlap:
                center_x, center_y = center_dic[key]
                distance = sqrt((float(junction.attrib['x']) - center_x) ** 2 + (float(junction.attrib['y']) - center_y) ** 2)
                if distance < closest_junction_distance[key]:
                    closest_junction_distance[key] = distance
                    junction_dic[key] = junction
    if junction_dic[key] is not None:
        rsu_counter -= 1

# Loop through the junctions and find junctions that are closest to each cluster center
# for junction in junction_list:
#     for key, (center_x, center_y) in center_dic.items():
#         distance = sqrt((float(junction.attrib['x']) - center_x) ** 2 + (float(junction.attrib['y']) - center_y) ** 2)
#         if distance < closest_junction_distance[key]:
#             closest_junction_distance[key] = distance
#             junction_dic[key] = junction

x_rsu = []
y_rsu = []
total_traffic = 0
output_junctions = []
for key, junction in junction_dic.items():
    if junction is not None:
        total_traffic += dicc[key]
for key, junction in junction_dic.items():
    if junction is not None:
        print("Cluster:", key, "Coord:", (float(junction.attrib['x']), float(junction.attrib['y'])), "Traffic Density:", dicc[key])
        x_rsu.append(float(junction.attrib['x']))
        y_rsu.append(float(junction.attrib['y']))
        output_junctions.append((junction, dicc[key]/total_traffic))


# Plot RSU as red stars
plt.scatter(x_rsu, y_rsu, s=50, c='red', marker='*')
plt.show()