import random


class Vehicle:
    """
    Vehicle object for Car ML Simulator.
    Attributes:
    - car_id
    - comp_power
    - bandwidth
    """
    def __init__(self, car_id, comp_power, bandwidth):
        self.car_id = car_id
        self.comp_power = comp_power
        self.bandwidth = bandwidth

    def download():
        return

    def upload():
        return
    


class RSU:
    """
    Road Side Unit object for Car ML Simulator.
    Attributes:
    - rsu_id
    - location_x
    - location_y
    """
    def __init__(self, rsu_id, location_x, location_y):
        self.rsu_id = rsu_id
        self.location_x = location_x
        self.location_y = location_y



class Dataset:
    """
    Dataset object for Car ML Simulator.
    Attributes:
    - name
    - size
    """
    def __init__(self, name, size):
        self.name = name
        self.size = size



class Sample:
    """
    Sample object for Car ML Simulator.
    Attributes:
    - s_id
    - dataset
    - size
    """
    def __init__(self, s_id, dataset, size):
        self.s_id = s_id
        self.dataset = dataset
        self.size = size

