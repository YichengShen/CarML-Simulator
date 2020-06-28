import simpy
import random
import numpy as np
import xml.etree.ElementTree as ET 


class Simulation(object):
    """
    Simulation object for Car ML Simulator.
    Attributes:
    - env
    - FCD_file
    - vDict
    - rsu_x_range
    - rsu_y_range
    - num_tasks
    - num_tasks_left
    """
    def __init__(self, env, FCD_file, vDict: dict, rsuRange: tuple, num_tasks):
        self.env = env
        self.FCD_file = FCD_file
        self.vDict = vDict
        self.rsu_x_range = rsuRange[0]
        self.rsu_y_range = rsuRange[1]
        self.num_tasks = num_tasks
        self.num_tasks_left = num_tasks
    
    def tasks_completed(self):
        return self.num_tasks_left == 0



class Vehicle:
    """
    Vehicle object for Car ML Simulator.
    Attributes:
    - car_id
    - x
    - y
    - speed
    - comp_power
    - bandwidth
    - task_timer
    """
    def __init__(self, car_id, comp_power, comp_power_std, bandwidth, bandwidth_std):
        self.car_id = car_id
        self.x = 0
        self.y = 0
        self.speed = 0
        self.comp_power = round(np.random.normal(comp_power, comp_power_std))
        self.bandwidth = round(np.random.normal(bandwidth, bandwidth_std))

        self.task_timer = 0 # if == 0, this car can take new tasks
    
    def downloaded(self):
        return self.task_timer == self.bandwidth
    
    def computed(self):
        return self.task_timer == (self.bandwidth + self.comp_power)

    def uploaded(self):
        return self.task_timer == (self.bandwidth * 2 + self.comp_power)
    

class RSU:
    """
    Road Side Unit object for Car ML Simulator.
    Attributes:
    - rsu_id
    - location_x
    - location_y
    - rsu_range
    - rsu_x_range
    - rsu_y_range
    """
    def __init__(self, rsu_id, location_x, location_y, rsu_range):
        self.rsu_id = rsu_id
        self.location_x = location_x
        self.location_y = location_y
        self.rsu_range = rsu_range
        self.rsu_x_range = [location_x - rsu_range, location_x + rsu_range]
        self.rsu_y_range = [location_y - rsu_range, location_y + rsu_range]



class SUMOfile:
    """
    Data read from SUMO XML files.
    Attributes:
    - ROU_file
    - NET_file
    """
    def __init__(self, ROU_file, NET_file):
        self.ROU_file = ROU_file
        self.NET_file = NET_file

    def make_vehicleDict(self, comp_power, comp_power_std, bandwidth, bandwidth_std):
        tree = ET.parse(self.ROU_file)
        root = tree.getroot()
        vehicleDict = {}
        for vehicle in root.findall('trip'):
            vehicleDict[vehicle.attrib['id']] = Vehicle(vehicle.attrib['id'], comp_power, comp_power_std, bandwidth, bandwidth_std)
        return vehicleDict

    def make_RSUList(self, range, num_rsu):
        tree = ET.parse(self.NET_file)
        root = tree.getroot()
        RSUList = []
        junctionList = np.random.choice(root.findall('junction'), num_rsu, replace=False)
        for i in range(num_rsu):
            id = 'rsu' + str(i)
            RSUList.append(RSU(id, junctionList[i].attrib['x'], junctionList[i].attrib['y'], range))
        return RSUList

    def make_RSURangeList(self, range, nums):
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



# class Dataset:  # the training data
#     """
#     Dataset object for Car ML Simulator.
#     Attributes:
#     - name
#     - size
#     """
#     def __init__(self, name, size):
#         self.name = name
#         self.size = size



# class Sample:
#     """
#     Sample object for Car ML Simulator.
#     Attributes:
#     - s_id
#     - dataset
#     - size
#     """
#     def __init__(self, s_id, dataset, size):
#         self.s_id = s_id
#         self.dataset = dataset
#         self.size = size