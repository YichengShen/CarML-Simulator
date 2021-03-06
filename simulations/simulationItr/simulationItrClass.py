import math
import numpy as np
import yaml
import xml.etree.ElementTree as ET


file = open('config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)


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
    - time_left_rsu
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
        self.time_left_rsu = 0

    def set_properties(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed

    def download_from_rsu(self, rsuList):
        rsu = self.in_range_rsu(rsuList)
        if rsu is not None:
            if self.rsu_assigned == None:
                self.rsu_assigned = rsu
                for _ in range(int(self.comp_power * cfg['vehicle']['tasks_per_comp_power'])):
                    if rsu.tasks_unassigned:
                        task = rsu.tasks_unassigned.pop()
                        self.tasks_assigned.append(task)
                        rsu.tasks_assigned.add(task)
                    elif rsu.tasks_assigned:
                        task = rsu.tasks_assigned.pop()
                        self.tasks_assigned.append(task)
                        rsu.tasks_assigned.add(task)
                    else:
                        break
            if self.rsu_assigned == rsu:
                num_downloaded_tasks = len(self.tasks_remaining)
                self.tasks_remaining.extend(self.tasks_assigned[num_downloaded_tasks:num_downloaded_tasks+int(self.bandwidth)])
                if len(self.tasks_remaining) == len(self.tasks_assigned) and len(self.tasks_assigned) > 0:
                    self.tasks_assigned.append(-1)

    def download_complete(self):
        if len(self.tasks_assigned) != int(self.comp_power * 20) + 1:
            return False
        else:
            if self.tasks_assigned[0] in self.rsu_assigned.tasks_assigned:
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

    def upload_to_rsu(self, rsu_list):
        rsu = self.in_range_rsu(rsu_list)
        if rsu is not None:
            rsu.received_results.append(self.computed_array.pop())
    
    def upload_complete(self):
        return not self.computed_array

    def transfer_data(self, simulation, timestep):
        vehicles_in_range = self.in_range_vehicle(timestep)
        for vehicle in vehicles_in_range:
            if vehicle.attrib['id'] not in simulation.vehicle_dict:
                simulation.add_into_vehicle_dict(vehicle)
                vehi = simulation.vehicle_dict[vehicle.attrib['id']]
                vehi.tasks_assigned = self.tasks_assigned
                vehi.tasks_remaining = self.tasks_remaining
                vehi.computed_array = self.computed_array
                break
            elif not simulation.vehicle_dict[vehicle.attrib['id']].tasks_assigned:
                vehi = simulation.vehicle_dict[vehicle.attrib['id']]
                vehi.tasks_assigned = self.tasks_assigned
                vehi.tasks_remaining = self.tasks_remaining
                vehi.computed_array = self.computed_array
                break

    def free_up(self):
        self.tasks_assigned = []
        self.rsu_assigned = None

    def in_range_rsu(self, rsu_list):
        shortest_distance = 99999999 # placeholder (a random large number)
        closest_rsu = None
        for rsu in rsu_list:
            distance = math.sqrt((rsu.rsu_x - self.x) ** 2 + (rsu.rsu_y - self.y) ** 2)
            if distance <= rsu.rsu_range and distance < shortest_distance:
                shortest_distance = distance
                closest_rsu = rsu
        return closest_rsu

    def in_range_vehicle(self, timestep):
        vehicles_in_range = []
        for vehicle in timestep.findall('vehicle'):
            distance = math.sqrt((float(vehicle.attrib['x']) - self.x) ** 2 + (float(vehicle.attrib['y']) - self.y) ** 2)
            if distance <= cfg['comm_range']['v2v']:
                vehicles_in_range.append(vehicle)
        return vehicles_in_range

    def out_of_range(self):
        if self.rsu_assigned != None:
            distance = math.sqrt((self.rsu_assigned.rsu_x - self.x) ** 2 + (self.rsu_assigned.rsu_y - self.y) ** 2)
            if distance > self.rsu_assigned.rsu_range:
                return True
        return False

    def out_of_bounds(self, root, timestep):
        current_timestep = float(timestep.attrib['time'])
        next_timestep = root.find('timestep[@time="{:.2f}"]'.format(current_timestep+1))
        if next_timestep == None:
            return False
        else:
            id_set = set(map(lambda vehicle: vehicle.attrib['id'], next_timestep.findall('vehicle')))
            return not self.car_id in id_set

    def update_time_left_rsu(self):
        self.time_left_rsu += 1
    
    def refresh_time_left_rsu(self):
        self.time_left_rsu = 0

    def remove_tasks_and_rsu(self):
        self.tasks_assigned = []
        self.tasks_remaining = []
        self.rsu_assigned = None
        self.computed_array = [] # ?
        self.time_left_rsu = 0



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
    """
    The dataset used to learn.
    Attributes:
    - data_list_id
    - num_tasks
    - data_list
    """
    def __init__(self, data_list_id, num_tasks):
        self.data_list_id = data_list_id
        self.num_tasks = num_tasks
        self.data_list = [datum for datum in range(num_tasks)]
    
    def partition_data(self, num_RSU):
        sample_dict = {}
        sample_size = int(self.num_tasks / num_RSU)
        sample_id = 0
        for i in range(0, self.num_tasks - 1, sample_size):
            if i + sample_size * 2 > self.num_tasks:
                sample = set(self.data_list[i:])
            else:
                sample = set(self.data_list[i:i + sample_size])
            sample_dict[sample_id] = RSU_Subtasks(sample_id, self.data_list_id, sample)
            sample_id += 1
        return sample_dict


class RSU_Subtasks:
    """
    A partition (sample) of the big dataset. Different partitions are on different RSUs.
    Attributes:
    - sample_id
    - parent_id
    - sample
    """
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
        vehicle_dict = {}
        for vehicle in root.findall('trip'):
            vehicle_dict[vehicle.attrib['id']] = Vehicle(vehicle.attrib['id'], comp_power, comp_power_std, bandwidth, bandwidth_std)
        return vehicle_dict

    def rsuList(self, rsu_range, rsu_nums, sample_dict):
        tree = ET.parse(self.NET_file)
        root = tree.getroot()
        rsu_list = []
        junction_list = np.random.choice(root.findall('junction'), rsu_nums, replace=False)
        for i in range(rsu_nums):
            id = 'rsu' + str(i)
            sample = sample_dict[i]
            rsu_list.append(RSU(id, float(junction_list[i].attrib['x']), float(junction_list[i].attrib['y']), rsu_range, sample))
        return rsu_list


class Simulation:
    """
    Simulation object for Car ML Simulator.
    Attributes:
    - FCD_file
    - vehicle_dict
    - rsu_list
    - num_tasks
    """
    def __init__(self, FCD_file, vehicle_dict: dict, rsu_list: list, num_tasks):
        self.FCD_file = FCD_file
        self.vehicle_dict = vehicle_dict
        self.rsu_list = rsu_list
        self.num_tasks = num_tasks

    def add_into_vehicle_dict(self, vehicle):
        self.vehicle_dict[vehicle.attrib['id']] = Vehicle(vehicle.attrib['id'],
                                                          cfg['simulation']['comp_power'],
                                                          cfg['simulation']['comp_power_std'], 
                                                          cfg['simulation']['bandwidth'],
                                                          cfg['simulation']['bandwidth_std'])
