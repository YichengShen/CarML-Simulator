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
        self.bandwidth = np.random.normal(bandwidth, bandwidth_std)
        self.max_data_rows = 10
        self.data_downloaded = []
        self.target_downloaded = []
        self.parent_rsu = []
        self.num_data_computed = 0

    def set_properties(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed

    # def find_available_rsu_with_data(self, rsu_list):
    #     rsu_available = []
    #     for rsu in rsu_list:
    #         distance = math.sqrt((rsu.rsu_x - self.x) ** 2 + (rsu.rsu_y - self.y) ** 2)
    #         if distance <= rsu.rsu_range:
    #             for each_id in rsu.id_data:
    #                 rsu_available.append(rsu)
    #     return rsu_available

    def find_rsu_in_range(self, rsu_list):
        rsu_in_range = []
        for rsu in rsu_list:
            distance = math.sqrt((rsu.rsu_x - self.x) ** 2 + (rsu.rsu_y - self.y) ** 2)
            if distance <= rsu.rsu_range:
                rsu_in_range.append(rsu)
        return rsu_in_range

    def download_from_rsu(self, rsu_list):
        rsu_in_range = self.find_rsu_in_range(rsu_list)
        num_to_download = self.max_data_rows - len(self.data_downloaded)
        if num_to_download > 0:
            i = min(num_to_download, int(self.bandwidth))
            j = i
            for rsu in rsu_in_range:
                if len(rsu.data) > 0:
                    for data in rsu.data:
                        if i > 0:
                            self.data_downloaded.append(data)
                            rsu.data_downloaded.append(data)
                            rsu.data = np.delete(rsu.data, data, 0)
                            i -= 1
                    for target in rsu.target:
                        if j > 0:
                            self.target_downloaded.append(target)
                            rsu.target_downloaded.append(target)
                            rsu.target = np.delete(rsu.target, [target], 0)
                            j -= 1
        

               

    # def download_from_rsu(self, rsu_list):
    #     rsu_available = self.find_available_rsu_with_data(rsu_list)
    #     if rsu_available:
    #         for idx in range(int(self.bandwidth)):
    #             if len(self.data_id_downloaded) < int(self.comp_power) * cfg['vehicle']['tasks_per_comp_power']:
    #                 if idx < len(rsu_available): # to prevent index out of range
    #                     id_data_row = rsu_available[idx].id_data.pop()
    #                     rsu_available[idx].id_data_downloaded.add(id_data_row)
    #                     self.data_id_downloaded.append((id_data_row, rsu_available[idx]))
    #             else:
    #                 break
    
    def download_completed(self):
        return len(self.data_id_downloaded) >= int(self.comp_power) * cfg['vehicle']['tasks_per_comp_power'] 

    def compute(self):
        self.num_data_computed += int(self.comp_power)
            
    def compute_completed(self):
        return self.num_data_computed >= len(self.data_id_downloaded)

    def upload(self, rsu_list, cloud_server):
        in_rsu_range = False
        for rsu in rsu_list:
            distance = math.sqrt((rsu.rsu_x - self.x) ** 2 + (rsu.rsu_y - self.y) ** 2)
            if distance <= rsu.rsu_range:
                in_rsu_range = True
                break
        if in_rsu_range:
            upload_speed = int(self.bandwidth)
        else:
            upload_speed = cfg['comm_speed']['speed_4g']
        for _ in range(upload_speed):
            tuple_result = self.data_id_downloaded.pop()
            id_result = tuple_result[0]
            cloud_server.data_id_list_finished.append(id_result)

    def upload_completed(self):
        return len(self.data_id_downloaded) == 0

    def free_up(self):
        self.data_id_downloaded = []
        self.num_data_computed = 0

    # def transfer_data(self, simulation, timestep):
    #     vehicles_in_range = self.in_range_vehicle(timestep)
    #     for vehicle in vehicles_in_range:
    #         if vehicle.attrib['id'] not in simulation.vehicle_dict:
    #             simulation.add_into_vehicle_dict(vehicle)
    #             vehi = simulation.vehicle_dict[vehicle.attrib['id']]
    #             vehi.tasks_assigned = self.tasks_assigned
    #             vehi.tasks_remaining = self.tasks_remaining
    #             vehi.computed_array = self.computed_array
    #             break
    #         elif not simulation.vehicle_dict[vehicle.attrib['id']].tasks_assigned:
    #             vehi = simulation.vehicle_dict[vehicle.attrib['id']]
    #             vehi.tasks_assigned = self.tasks_assigned
    #             vehi.tasks_remaining = self.tasks_remaining
    #             vehi.computed_array = self.computed_array
    #             break

    # def in_range_rsu(self, rsu_list):
    #     shortest_distance = 99999999 # placeholder (a random large number)
    #     closest_rsu = None
    #     for rsu in rsu_list:
    #         distance = math.sqrt((rsu.rsu_x - self.x) ** 2 + (rsu.rsu_y - self.y) ** 2)
    #         if distance <= rsu.rsu_range and distance < shortest_distance:
    #             shortest_distance = distance
    #             closest_rsu = rsu
    #     return closest_rsu

    # def in_range_vehicle(self, timestep):
    #     vehicles_in_range = []
    #     for vehicle in timestep.findall('vehicle'):
    #         distance = math.sqrt((float(vehicle.attrib['x']) - self.x) ** 2 + (float(vehicle.attrib['y']) - self.y) ** 2)
    #         if distance <= cfg['comm_range']['v2v']:
    #             vehicles_in_range.append(vehicle)
    #     return vehicles_in_range

    # def out_of_range(self):
    #     if self.rsu_assigned != None:
    #         distance = math.sqrt((self.rsu_assigned.rsu_x - self.x) ** 2 + (self.rsu_assigned.rsu_y - self.y) ** 2)
    #         if distance > self.rsu_assigned.rsu_range:
    #             return True
    #     return False

    def out_of_bounds(self, root, timestep):
        current_timestep = float(timestep.attrib['time'])
        next_timestep = root.find('timestep[@time="{:.2f}"]'.format(current_timestep+1))
        if next_timestep == None:
            return False
        else:
            id_set = set(map(lambda vehicle: vehicle.attrib['id'], next_timestep.findall('vehicle')))
            return not self.car_id in id_set

    def unlock_downloaded_data(self):
        for each_tuple in self.data_id_downloaded:
            rsu = each_tuple[1]
            id_data_row = each_tuple[0]
            rsu.id_data.append(id_data_row)
            rsu.id_data_downloaded.remove(id_data_row)


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
    def __init__(self, rsu_id, rsu_x, rsu_y, rsu_range):
        self.rsu_id = rsu_id
        self.rsu_x = rsu_x
        self.rsu_y = rsu_y
        self.rsu_range = rsu_range
        self.data = [] 
        self.target = []
        self.data_downloaded = []
        self.target_downloaded = []


class Cloud_Server:
    """
    Cloud Server object for Car ML Simulator.
    Attributes:
    """
    def __init__(self, dataset, rsu_list):
        self.dataset = dataset
        self.rsu_list = rsu_list
        # self.data_id_list_finished = []
        

    def distribute_to_rsu(self):
        # this function directly changes RSU instances in rsu_list
        # partition data_id_list and send to rsu
        num_data_rows = len(self.dataset.data)
        num_rsu = len(self.rsu_list)
        overlap_rate = 0.03 / num_rsu # actual overlap rate can be slightly lower than 0.03 due to rounding
        
        minibatch_size = int(num_data_rows / num_rsu)
        minibatch_size_overlap = int(num_data_rows / num_rsu + overlap_rate * num_data_rows)
        
        idx = 0
        for rsu in self.rsu_list:
            if idx + minibatch_size_overlap <= num_data_rows:
                rsu.data = self.dataset.data[idx:idx + minibatch_size_overlap, :]
                rsu.target = self.dataset.target[idx:idx + minibatch_size_overlap]
            # If reached the end, overlap from the beginning
            else:
                diff = idx + minibatch_size_overlap - num_data_rows
                rsu.data = np.vstack((self.dataset.data[idx:, :], self.dataset.data[:diff, :]))
                rsu.target = np.concatenate((self.dataset.target[idx:], self.dataset.target[:diff]))
            idx += minibatch_size


class Training_Dataset:
    """
    The dataset used to learn.
    Attributes:

    """
    def __init__(self, dataset_id, X, y):
        self.dataset_id = dataset_id
        self.data = X
        self.target = y
        self.num_tasks = len(self.data)

    # def data_dict(self):
    #     data_dictionary = {}
    #     for idx in range(self.num_tasks):
    #         data_dictionary[idx] = (self.data[idx], self.target[idx])
    #     return data_dictionary


# class Data_Minibatch:
#     """
#     A partition (sample) of the big dataset. Different partitions are on different RSUs.
#     Attributes:
#     - sample_id
#     - parent_id
#     - sample
#     """
#     def __init__(self, sample_id, parent_id, sample):
#         self.sample_id = sample_id
#         self.parent_id = parent_id
#         self.sample = sample

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

    def rsuList(self, rsu_range, rsu_nums):
        tree = ET.parse(self.NET_file)
        root = tree.getroot()
        rsu_list = []
        junction_list = np.random.choice(root.findall('junction'), rsu_nums, replace=False)
        for i in range(rsu_nums):
            id = 'rsu' + str(i)
            rsu_list.append(RSU(id, float(junction_list[i].attrib['x']), float(junction_list[i].attrib['y']), rsu_range))
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
    def __init__(self, FCD_file, vehicle_dict: dict, rsu_list: list, dataset):
        self.FCD_file = FCD_file
        self.vehicle_dict = vehicle_dict
        self.rsu_list = rsu_list
        self.dataset = dataset
       

    def add_into_vehicle_dict(self, vehicle):
        self.vehicle_dict[vehicle.attrib['id']] = Vehicle(vehicle.attrib['id'],
                                                          cfg['simulation']['comp_power'],
                                                          cfg['simulation']['comp_power_std'], 
                                                          cfg['simulation']['bandwidth'],
                                                          cfg['simulation']['bandwidth_std'])
