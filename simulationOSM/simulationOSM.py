import numpy as np 
import xml.etree.ElementTree as ET 

class Vehicle:
    def __init__(self, car_id, comp_power, comp_power_std, bandwidth, bandwidth_std):
        self.car_id = car_id
        self.x = 0
        self.y = 0
        self.speed = 0
        self.tasks_remaining = 200
        self.comp_power = np.random.normal(comp_power, comp_power_std)
        self.bandwidth = np.random.normal(bandwidth, bandwidth_std)
    

class RSU:
    def __init__(self, rsu_id, rsu_x, rsu_y, rsu_range):
        self.rsu_id = rsu_id
        self.rsu_x = rsu_x
        self.rsu_y = rsu_y
        self.rsu_range = rsu_range
        self.rsu_x_range = [rsu_x - rsu_range, rsu_x + rsu_range]
        self.rsu_y_range = [rsu_y - rsu_range, rsu_y + rsu_range]

class Dataset:
    def __init__(self, ROU_file):
        self.ROU_file = ROU_file

    def vehicleDict(self, comp_power, comp_power_std, bandwidth, bandwidth_std):
        tree = ET.parse(self.ROU_file)
        root = tree.getroot()
        vehicleDict = {}
        for vehicle in root.findall('trip'):
            vehicleDict[vehicle.attrib['id']] = Vehicle(vehicle.attrib['id'], comp_power, comp_power_std, bandwidth, bandwidth_std)
        return vehicleDict

class Simulation:
    def __init__(self, FCD_file, vehicleDict: dict, rsuList: list, num_tasks):
        self.FCD_file = FCD_file
        self.vehicleDict = vehicleDict
        self.rsuList = rsuList
        self.num_tasks = num_tasks
        self.num_tasks_distributed = num_tasks / len(vehicleDict)

    def merge(self, intervals):
        out = []
        for i in sorted(intervals, key=lambda i: i[0]):
            if out and i[0]<=out[-1][-1]:
                out[-1][-1] = max(out[-1][-1], i[-1])
            else: out+=[i]
        return out

    def simulate(self):
        tree = ET.parse(self.FCD_file)
        root = tree.getroot()
        rsu_x_range = self.merge(map(lambda x: x.rsu_x_range, self.rsuList))
        rsu_y_range = self.merge(map(lambda x: x.rsu_y_range, self.rsuList))
        for timestep in root:
            for vehicle in timestep.findall('vehicle'):
                vehi = self.vehicleDict[vehicle.attrib['id']]
                vehi.x = float(vehicle.attrib['x'])
                vehi.y = float(vehicle.attrib['y'])
                vehi.speed = float(vehicle.attrib['speed'])
                inRangeX = False
                inRangeY = False
                for x_min, x_max in rsu_x_range:
                    if x_min <= vehi.x <= x_max:
                        inRangeX = True
                for y_min, y_max in rsu_y_range:
                    if y_min <= vehi.y <= y_max:
                        inRangeY = True
                if inRangeX and inRangeY:
                    if vehi.tasks_remaining > 0:
                        vehi.tasks_remaining -= vehi.comp_power
                        if vehi.tasks_remaining <= 0:
                            self.num_tasks -= self.num_tasks_distributed
                            if self.num_tasks <= 0:
                                return timestep.attrib['time']
        print('Cannot Process All')
        return timestep.attrib['time']

def main():
    num_tasks = 1000
    data = Dataset('osm_boston_common/osm.passenger.trips.xml')
    vehicleDict = data.vehicleDict(10, 2, 0, 0)
    rsu_1 = RSU('rsu_1', 2500, 3500, 00)
    simulation1 = Simulation('osm_boston_common/osm_fcd.xml', vehicleDict, [rsu_1], num_tasks)
    timeStep = simulation1.simulate()
    print(timeStep)

main()

