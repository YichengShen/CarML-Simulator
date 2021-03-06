import math
import numpy as np
import yaml
import xml.etree.ElementTree as ET
from sklearn.linear_model import LogisticRegression


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
    - bandwidth
    - lock
    - max_data_rows
    - data_tuples_downloaded
    - parent_rsu
    - coef
    - uploaded
    """
    def __init__(self, car_id, comp_power, comp_power_std, bandwidth, bandwidth_std):
        self.car_id = car_id
        self.x = 0
        self.y = 0
        self.speed = 0
        self.comp_power = np.random.normal(comp_power, comp_power_std)
        self.bandwidth = np.random.normal(bandwidth, bandwidth_std)
        self.lock = 0
        self.max_data_rows = cfg['vehicle']['max_data_rows']
        self.data_tuples_downloaded = []
        self.parent_rsu = []
        self.coef = []
        self.uploaded = False

    def set_properties(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed

    def find_rsu_in_range(self, rsu_list):
        rsu_in_range = []
        for rsu in rsu_list:
            distance = math.sqrt((rsu.rsu_x - self.x) ** 2 + (rsu.rsu_y - self.y) ** 2)
            if distance <= rsu.rsu_range:
                rsu_in_range.append(rsu)
        return rsu_in_range

    def download_from_rsu(self, rsu_list):
        rsu_in_range = self.find_rsu_in_range(rsu_list)
        num_to_download = self.max_data_rows - len(self.data_tuples_downloaded)
        if num_to_download > 0:
            i = min(num_to_download, int(self.bandwidth))
            for rsu in rsu_in_range:
                if len(rsu.data_tuples) > 0:
                    if i > 0:
                        if len(rsu.data_tuples) >= i:
                            self.data_tuples_downloaded.extend(rsu.data_tuples[:i])
                            rsu.data_tuples_downloaded.extend(rsu.data_tuples[:i])
                            rsu.data_tuples = rsu.data_tuples[i:]
                            return
                        else:
                            self.data_tuples_downloaded.extend(rsu.data_tuples)
                            rsu.data_tuples_downloaded.extend(rsu.data_tuples)
                            i -= len(rsu.data_tuples)
                            rsu.data_tuples = []
                    
    def download_completed(self):
        return self.max_data_rows <= len(self.data_tuples_downloaded)

    def compute(self):
        X_train = np.array([data_tuple[0] for data_tuple in self.data_tuples_downloaded])
        Y_train = np.array([data_tuple[1] for data_tuple in self.data_tuples_downloaded])
        
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(X_train, Y_train)
        coef = classifier.coef_
        # Y_pred = classifier.predict(X_train)
        self.coef = coef
        # Timer
        lock_time = int(len(self.data_tuples_downloaded) / self.comp_power) - 1
        self.lock += lock_time
            
    def compute_completed(self):
        return self.coef != []

    def upload(self, rsu_list, cloud_server):
        in_rsu_range = False
        for rsu in rsu_list:
            distance = math.sqrt((rsu.rsu_x - self.x) ** 2 + (rsu.rsu_y - self.y) ** 2)
            if distance <= rsu.rsu_range:
                in_rsu_range = True
                break
        cloud_server.results.append(self.coef)
        # Calculate upload time depending on uploading to RSU or via 4G
        if in_rsu_range:
            upload_time = int(len(self.coef) / self.bandwidth)
        else:
            upload_time = int(len(self.coef) / cfg['comm_speed']['speed_4g'])
        if upload_time == 0:
            upload_time = 1
        # Timer
        lock_time = upload_time - 1
        self.lock += lock_time
        self.uploaded = True
            
    def upload_completed(self):
        return self.uploaded

    def free_up(self):
        self.data_tuples_downloaded = []
        self.parent_rsu = []
        self.coef = []
        self.uploaded = False

    def is_not_locked(self):
        return self.lock == 0

    def update_lock(self):
        if self.lock > 0:
            self.lock -= 1

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

    # def out_of_bounds(self, root, timestep):
    #     current_timestep = float(timestep.attrib['time'])
    #     next_timestep = root.find('timestep[@time="{:.2f}"]'.format(current_timestep+1))
    #     if next_timestep == None:
    #         return False
    #     else:
    #         id_set = set(map(lambda vehicle: vehicle.attrib['id'], next_timestep.findall('vehicle')))
    #         return not self.car_id in id_set

    # def unlock_downloaded_data(self):
    #     for each_tuple in self.data_id_downloaded:
    #         rsu = each_tuple[1]
    #         id_data_row = each_tuple[0]
    #         rsu.id_data.append(id_data_row)
    #         rsu.id_data_downloaded.remove(id_data_row)


class RSU:
    """
    Road Side Unit object for Car ML Simulator.
    Attributes:
    - rsu_id
    - rsu_x
    - rsu_y
    - rsu_range
    - data_tuples
    - data_tuples_downloaded
    """
    def __init__(self, rsu_id, rsu_x, rsu_y, rsu_range):
        self.rsu_id = rsu_id
        self.rsu_x = rsu_x
        self.rsu_y = rsu_y
        self.rsu_range = rsu_range
        self.data_tuples = [] 
        self.data_tuples_downloaded = []


class Cloud_Server:
    """
    Cloud Server object for Car ML Simulator.
    Attributes:
    - dataset
    - rsu_list
    - results
    """
    def __init__(self, dataset, rsu_list):
        self.dataset = dataset
        self.rsu_list = rsu_list
        self.results = []
        

    def distribute_to_rsu(self):
        # this function directly changes RSU instances in rsu_list
        # partition data_id_list and send to rsu
        num_data_rows = len(self.dataset.data)
        num_rsu = len(self.rsu_list)
        overlap_rate = 0.03 / num_rsu # actual overlap rate can be slightly lower than 0.03 due to rounding
        
        rsu_batch_size = int(num_data_rows / num_rsu)
        rsu_batch_size_overlap = int(num_data_rows / num_rsu + overlap_rate * num_data_rows)
        
        tuple_list = self.dataset.data_tuples()

        idx = 0
        for rsu in self.rsu_list:
            if idx + rsu_batch_size_overlap <= num_data_rows:
                rsu.data_tuples = tuple_list[idx:idx + rsu_batch_size_overlap]
            # If reached the end, overlap from the beginning
            else:
                diff = idx + rsu_batch_size_overlap - num_data_rows
                rsu.data_tuples = tuple_list[idx:] + tuple_list[:diff]
                rsu.target = np.concatenate((self.dataset.target[idx:], self.dataset.target[:diff]))
            idx += rsu_batch_size


class Training_Dataset:
    """
    The dataset used to learn.
    Attributes:
    - dataset_id
    - data
    - target
    - num_tasks
    """
    def __init__(self, dataset_id, X, y):
        self.dataset_id = dataset_id
        self.data = X
        self.target = y
        self.num_tasks = len(self.data)

    def data_tuples(self):
        tuple_list = []
        for idx in range(len(self.data)):
            tuple_list.append((self.data[idx], self.target[idx]))
        return tuple_list


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
    - dataset
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
