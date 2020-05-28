import math
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
    - tasks_assigned
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
        self.tasks_assigned = self.comp_power * 20
        self.tasks_remaining = self.tasks_assigned
        self.bandwidth = np.random.normal(bandwidth, bandwidth_std)
        self.comm_time = 10 * (0.5 + 1 / self.bandwidth)
        self.download_time = self.comm_time
        self.upload_time = self.comm_time

    def download_from_rsu(self, rsuList):
        if self.inRange(rsuList):
            self.download_time -= 1

    def compute(self):
        self.tasks_remaining -= self.comp_power

    def upload_to_rsu(self, rsuList):
        if self.inRange(rsuList):
            self.upload_time -= 1

    def inRange(self, rsuList):
        for rsu in rsuList:
            if math.sqrt((rsu.rsu_x - self.x) ** 2 + (rsu.rsu_y - self.y) ** 2) <= rsu.rsu_range:
                return True
        return False

class RSU:
    """
    Road Side Unit object for Car ML Simulator.
    Attributes:
    - rsu_id
    - rsu_x
    - rsu_y
    - rsu_range
    """
    def __init__(self, rsu_id, rsu_x, rsu_y, rsu_range):
        self.rsu_id = rsu_id
        self.rsu_x = rsu_x
        self.rsu_y = rsu_y
        self.rsu_range = rsu_range


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

    def rsuList(self, rsu_range, rsu_nums):
        tree = ET.parse(self.NET_file)
        root = tree.getroot()
        RSUList = []
        junctionList = np.random.choice(root.findall('junction'), rsu_nums, replace=False)
        for i in range(rsu_nums):
            id = 'rsu' + str(i)
            RSUList.append(RSU(id, float(junctionList[i].attrib['x']), float(junctionList[i].attrib['y']), rsu_range))
        return RSUList

class Simulation:
    """
    Simulation object for Car ML Simulator.
    Attributes:
    - FCD_file
    - vehicleDict
    - rsuList
    - num_tasks
    """
    def __init__(self, FCD_file, vehicleDict: dict, rsuList: list, num_tasks):
        self.FCD_file = FCD_file
        self.vehicleDict = vehicleDict
        self.rsuList = rsuList
        self.num_tasks = num_tasks
