import numpy as np
import xml.etree.ElementTree as ET

class Vehicle:
    """
    Vehicle object for Car ML Simulator.
    Attributes:
    - car_id
    - x
    - y
    - speed
    - comp_power
    - tasks_distributed
    - tasks_remaining
    - bandwidth
    - comm_time
    - download_time
    - upload_time
    """
    def __init__(self, car_id, comp_power, comp_power_std, bandwidth, bandwidth_std):
        self.car_id = car_id
        self.x = 0
        self.y = 0
        self.speed = 0
        self.comp_power = np.random.normal(comp_power, comp_power_std)
        self.tasks_distributed = self.comp_power * 20
        self.tasks_remaining = self.tasks_distributed
        self.bandwidth = np.random.normal(bandwidth, bandwidth_std)
        self.comm_time = 10 * (0.5 + 1 / self.bandwidth)
        self.download_time = self.comm_time
        self.upload_time = self.comm_time

    def downloaded(self):
        self.download_time -= 1
        return self.download_time <= 0

    def uploaded(self):
        self.upload_time -= 1
        return self.download_time <= 0

class RSU:
    """
    Road Side Unit object for Car ML Simulator.
    Attributes:
    - rsu_id
    - rsu_x
    - rsu_y
    - rsu_range
    - rsu_x_range
    - rsu_y_range
    """
    def __init__(self, rsu_id, rsu_x, rsu_y, rsu_range):
        self.rsu_id = rsu_id
        self.rsu_x = rsu_x
        self.rsu_y = rsu_y
        self.rsu_range = rsu_range
        self.rsu_x_range = [rsu_x - rsu_range, rsu_x + rsu_range]
        self.rsu_y_range = [rsu_y - rsu_range, rsu_y + rsu_range]


class Dataset:
    """
    Data read from SUMO XML files.
    Attributes:
    - ROU_file
    - NET_file
    """
    def __init__(self, ROU_file, NET_file):
        self.ROU_file = ROU_file
        self.NET_file = NET_file

    def vehicleDict(self, comp_power, comp_power_std, bandwidth, bandwidth_std):
        tree = ET.parse(self.ROU_file)
        root = tree.getroot()
        vehicleDict = {}
        for vehicle in root.findall('trip'):
            vehicleDict[vehicle.attrib['id']] = Vehicle(vehicle.attrib['id'], comp_power, comp_power_std, bandwidth, bandwidth_std)
        return vehicleDict

    def RSUList(self, range, nums):
        tree = ET.parse(self.NET_file)
        root = tree.getroot()
        RSUList = []
        junctionList = np.random.choice(root.findall('junction'), nums, replace=False)
        for i in range(nums):
            id = 'rsu' + str(i)
            RSUList.append(RSU(id, junctionList[i].attrib['x'], junctionList[i].attrib['y'], range))
        return RSUList

    def RSURangeList(self, range, nums):
        tree = ET.parse(self.NET_file)
        root = tree.getroot()
        RSURangeX = []
        RSURangeY = []
        junctionList = np.random.choice(root.findall('junction'), nums, replace=False)
        for junction in junctionList:
            junct = junction.attrib
            RSURangeX.append((float(junct['x']) - range, float(junct['x']) + range))
            RSURangeY.append((float(junct['y']) - range, float(junct['y']) + range))
        return (RSURangeX, RSURangeY)
            
    # def merge(self, intervals):
    #     out = []
    #     for i in sorted(intervals, key=lambda i: i[0]):
    #         if out and i[0]<=out[-1][-1]:
    #             out[-1][-1] = max(out[-1][-1], i[-1])
    #         else: out+=[i]
    #     return out

class Simulation:
    """
    Simulation object for Car ML Simulator.
    Attributes:
    - FCD_file
    - vehicleDict
    - rsu_x_range
    - rsu_y_range
    - num_tasks
    """
    def __init__(self, FCD_file, vehicleDict: dict, rsuRange: tuple, num_tasks):
        self.FCD_file = FCD_file
        self.vehicleDict = vehicleDict
        self.rsu_x_range = rsuRange[0]
        self.rsu_y_range = rsuRange[1]
        self.num_tasks = num_tasks
