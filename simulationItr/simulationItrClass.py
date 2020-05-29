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
    - rsu_assigned
    - bandwidth
    - computed_array
    """
    def __init__(self, car_id, comp_power, comp_power_std, bandwidth, bandwidth_std):
        self.car_id = car_id
        self.x = 0
        self.y = 0
        self.speed = 0
        self.comp_power = np.random.normal(comp_power, comp_power_std)
        self.tasks_assigned = []
        self.tasks_remaining = []
        self.rsu_assigned = None
        self.bandwidth = np.random.normal(bandwidth, bandwidth_std)
        self.computed_array = []

    def download_from_rsu(self, rsuList):
        rsu = self.in_range(rsuList)
        if rsu:
            if self.rsu_assigned == None:
                self.rsu_assigned = rsu
            if self.rsu_assigned == rsu:
                for _ in range(int(self.bandwidth)):
                    if rsu.tasks_unassigned:
                        task = rsu.tasks_unassigned.pop()
                        self.tasks_assigned.append(task)
                        self.tasks_remaining.append(task)
                        rsu.tasks_assigned.add(task)
                    elif rsu.tasks_assigned:
                        task = rsu.tasks_assigned.pop()
                        self.tasks_assigned.append(task)
                        self.tasks_remaining.append(task)
                        rsu.tasks_assigned.add(task)
                    else:
                        return

    def download_complete(self):
        if len(self.tasks_assigned) < self.comp_power * 20:
            return False
        else:
            for task in self.tasks_assigned:
                self.rsu_assigned.tasks_assigned.discard(task)
                self.rsu_assigned.tasks_downloaded.add(task)
                return True

    def compute(self):
        result = []
        for _ in range(int(self.comp_power)):
            if self.tasks_remaining:
                result.append(self.tasks_remaining.pop())
        self.computed_array.append(result)

    def compute_complete(self):
        return not self.tasks_remaining 

    def upload_to_rsu(self, rsuList):
        rsu = self.in_range(rsuList)
        if rsu:
            rsu.received_results.append(self.computed_array.pop())
    
    def upload_complete(self):
        if not self.computed_array:
            return True
        else:
            return False

    def in_range(self, rsuList):
        shortestDistance = 10000
        closestRsu = None
        for rsu in rsuList:
            distance = math.sqrt((rsu.rsu_x - self.x) ** 2 + (rsu.rsu_y - self.y) ** 2)
            if distance <= rsu.rsu_range and distance < shortestDistance:
                shortestDistance = distance
                closestRsu = rsu
        return closestRsu
    
    def has_data(self, rsu, datum):
        if datum in rsu.sample.sample:
            return True



class RSU:
    """
    Road Side Unit object for Car ML Simulator.
    Attributes:
    - rsu_id
    - rsu_x
    - rsu_y
    - rsu_range
    - tasks_unassigned
    - tasks_assigned
    - tasks_downloaded
    - received_results
    """
    def __init__(self, rsu_id, rsu_x, rsu_y, rsu_range, tasks_unassigned):
        self.rsu_id = rsu_id
        self.rsu_x = rsu_x
        self.rsu_y = rsu_y
        self.rsu_range = rsu_range
        self.tasks_unassigned = tasks_unassigned.sample
        self.tasks_assigned = set()
        self.tasks_downloaded = set()
        self.received_results = []


class Task_Set:
    """The dataset used to learn."""
    def __init__(self, data_list_id, num_tasks):
        self.data_list_id = data_list_id
        self.num_tasks = num_tasks
        self.data_list = [datum for datum in range(num_tasks)]
    
    def partition_data(self, num_RSU):
        sampleDict = {}
        sample_size = int(self.num_tasks / num_RSU)
        sample_id = 0
        for i in range(0, self.num_tasks - 1, sample_size):
            if i + sample_size * 2 > self.num_tasks:
                sample = set(self.data_list[i:])
            else:
                sample = set(self.data_list[i:i + sample_size])
            sampleDict[sample_id] = RSU_Subtasks(sample_id, self.data_list_id, sample)
            sample_id += 1
        return sampleDict


class RSU_Subtasks:
    def __init__(self, sample_id, parent_id, sample):
        self.sample_id = sample_id
        self.parent_id = parent_id
        self.sample = sample

class SUMO_Dataset:
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

    def rsuList(self, rsu_range, rsu_nums, sample_dict):
        tree = ET.parse(self.NET_file)
        root = tree.getroot()
        RSUList = []
        junctionList = np.random.choice(root.findall('junction'), rsu_nums, replace=False)
        for i in range(rsu_nums):
            id = 'rsu' + str(i)
            sample = sample_dict[i]
            RSUList.append(RSU(id, float(junctionList[i].attrib['x']), float(junctionList[i].attrib['y']), rsu_range, sample))
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
