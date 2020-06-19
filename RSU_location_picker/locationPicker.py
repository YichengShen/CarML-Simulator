import xml.etree.ElementTree as ET
from math import ceil, sqrt
import numpy as np
import matplotlib.pyplot as plt
from heapq import nlargest


# Load boundaries
tree = ET.parse("osm_boston_common/osm.net.xml")
root = tree.getroot()
bounds = root[0].attrib["convBoundary"].split(",")
x_min = ceil(float(bounds[0]))
y_min = ceil(float(bounds[1]))
x_max = ceil(float(bounds[2]))
y_max = ceil(float(bounds[3]))
print("X-range:", x_min, x_max, "\nY-range:", y_min, y_max, "\n")

# Create 2D array filled with zeros
traffic_array = np.zeros((x_max + 1, y_max + 1), dtype=int)

# Calculate traffic on each (x,y)
tree = ET.parse("osm_boston_common/osm_fcd.xml")
root = tree.getroot()
for timestep in root:
    for vehicle in timestep.findall('vehicle'):
        x = round(float(vehicle.attrib['x']))
        y = round(float(vehicle.attrib['y']))
        traffic_array[x][y] += 1

# Load Junctions
tree = ET.parse("osm_boston_common/osm.net.xml")
root = tree.getroot()
junction_list = root.findall('junction')

# Calculate traffic around each junction
junction_traffic = {}
for i in range(len(traffic_array)):
    for j in range(len(traffic_array[i])):
        if traffic_array[i][j] > 0:
            for junction in junction_list:
                distance = sqrt((float(junction.attrib['x']) - i) ** 2 + (float(junction.attrib['y']) - j) ** 2)
                if distance <= 200:
                    if junction in junction_traffic:
                        junction_traffic[junction] += traffic_array[i][j]
                    else:
                        junction_traffic[junction] = traffic_array[i][j]
            

# Pick N junctions with most traffic
# current problem is that these junctions can be close to each other
rsu_num = 10
rsu_list = nlargest(rsu_num, junction_traffic, key=junction_traffic.get) 

for rsu in rsu_list:
    print("Coord:", (float(rsu.attrib['x']), float(rsu.attrib['y'])), "Accumulated Traffic Amount:", junction_traffic[rsu])

# Heatmap of all junctions according to traffic
heatmap = np.zeros((x_max + 1, y_max + 1), dtype=int)
for junction in list(junction_traffic.keys()):
    x = round(float(junction.attrib['x']))
    y = round(float(junction.attrib['y']))
    heatmap[x][y] = junction_traffic[junction]

extent = [x_min, x_max, y_min, y_max]
plt.imshow(heatmap, cmap='hot', aspect='auto', extent=extent, interpolation='nearest')
plt.colorbar()
plt.show()