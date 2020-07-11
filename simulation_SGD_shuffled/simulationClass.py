import math
import heapq
import random
from collections import deque
import numpy as np
import yaml
import xml.etree.ElementTree as ET 
import tensorflow as tf
import tensorflow_datasets as tfds


file = open('config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)
np.random.seed(101)

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
    - rsu_assigned
    - model
    - training_data_assigned
    - num_training_data-downloaded
    - training_label_assigned
    - num_training_label_downloaded
    - data_index
    - data_epoch
    - data_length
    - gradients
    - gradients_index
    - upload_complete
    - lock
    """
    def __init__(self, car_id, comp_power, comp_power_std, bandwidth, bandwidth_std):
        self.car_id = car_id
        self.x = 0
        self.y = 0
        self.speed = 0
        self.comp_power = np.random.normal(comp_power, comp_power_std)
        self.bandwidth = np.random.normal(bandwidth, bandwidth_std)
        self.rsu_assigned = None
        self.model = None
        self.training_data_assigned = []
        self.num_training_data_downloaded = 0
        self.training_label_assigned = []
        self.num_training_label_downloaded = 0
        self.data_index = -1                        # The index of the mini-batch assigned
        self.data_epoch = 0                         # The epoch the mini-batch belongs to
        self.data_length = 0                        # The length of the mini-batch
        self.gradients = None
        self.gradients_index = None                 # The index of the central server model when the car downloads the model
        self.upload_complete = False
        self.lock = 0
        self.malfunction = False

    def set_properties(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed

    # Return the RSU that is cloest to the vehicle
    def closest_rsu(self, rsu_list):
        shortest_distance = 99999999 # placeholder (a random large number)
        closest_rsu = None
        for rsu in rsu_list:
            distance = math.sqrt((rsu.rsu_x - self.x) ** 2 + (rsu.rsu_y - self.y) ** 2)
            if distance <= rsu.rsu_range and distance < shortest_distance:
                shortest_distance = distance
                closest_rsu = rsu
        return closest_rsu

    # Return a list of RSUs that is within the range of the vehicle
    # with each RSU being sorted from the closest to the furtherst
    def in_range_rsus(self, rsu_list):
        in_range_rsus = []
        for rsu in rsu_list:
            distance = math.sqrt((rsu.rsu_x - self.x) ** 2 + (rsu.rsu_y - self.y) ** 2)
            if distance <= rsu.rsu_range:
                heapq.heappush(in_range_rsus, (distance, rsu))
        return [heapq.heappop(in_range_rsus)[1] for i in range(len(in_range_rsus))]

    def download_from_rsu(self, rsu_list, central_server):
        # If a car still holds data from previous epoch
        if self.data_epoch != central_server.num_epoch:
            self.rsu_assigned = None
        if self.rsu_assigned is None:
            for rsu in rsu_list:
                if rsu.dataset:
                    self.rsu_assigned = rsu
                    rsu.vehicle_traffic += 1
                    dataset = rsu.dataset.popleft()
                    self.training_data_assigned = dataset[1][0]
                    self.training_label_assigned = dataset[1][1]
                    self.data_index = dataset[0]
                    self.data_epoch = central_server.num_epoch
                    self.data_length = len(self.training_label_assigned)
                    self.model = rsu.model
                    break
        self.num_training_data_downloaded  += int(self.bandwidth)
        self.num_training_label_downloaded += int(self.bandwidth)

    def download_from_central_server(self):
        self.num_training_data_downloaded  += int(self.bandwidth / 4)
        self.num_training_label_downloaded += int(self.bandwidth / 4)
        
    def download_completed(self):
        if self.rsu_assigned is not None:
            return self.num_training_data_downloaded >= self.data_length
        return False

    def download_model_from(self, central_server):
        self.model = central_server.model

    # If update is True, the vehicle computes the gradients with the model from the central server
    # If update is False, the vehicle computes the gradients with the model from an RSU
    # If bounded is True, the training will be bounded by bounded_staleness
    # If bounded is False, the training will be bounded by max_gradients_difference, which is the gradient difference between when the vehicle download and update the model
    def compute_gradients(self, central_server, update: bool, bounded: bool):
        # If the car still holds data from previous epoch
        if self.data_epoch != central_server.num_epoch:
            self.free_up()
            return
        if update:
            if bounded:
                if central_server.bounded_staleness > 0:
                    neural_network = Neural_Network()
                    self.gradients = neural_network.grad(central_server.model, np.array(self.training_data_assigned), np.array(self.training_label_assigned), central_server)
                    central_server.bounded_staleness -= 1
                    lock_time = int(self.bandwidth / 2) + int(self.data_length / self.comp_power) - 1
                    self.lock += lock_time
            else:
                neural_network = Neural_Network()
                self.gradients = neural_network.grad(central_server.model, np.array(self.training_data_assigned), np.array(self.training_label_assigned), central_server)
                self.gradients_index = central_server.gradients_received
                lock_time = int(self.bandwidth / 2) + int(self.data_length / self.comp_power) - 1
                self.lock += lock_time
        else:
            neural_network = Neural_Network()
            self.gradients = neural_network.grad(self.model, np.array(self.training_data_assigned), np.array(self.training_label_assigned), central_server)
            lock_time = int(self.data_length / self.comp_power) - 1
            self.lock += lock_time

    def compute_completed(self):
        return self.gradients is not None

    # Assume the car directly uploads the gradients to the central server and update the model of central server
    def upload_gradients_to_central_server(self, central_server, update: bool, bounded: bool):
        # If the gradients is still from the last epoch
        if self.data_epoch != central_server.num_epoch:
            central_server.bounded_staleness += 1
            self.free_up()
            return
        neural_network = Neural_Network()
        if self.data_index not in central_server.received_data:
            if update:
                if bounded:
                    neural_network.optimizer.apply_gradients(zip(self.gradients, central_server.model.trainable_variables))
                    central_server.gradients_received += 1
                    lock_time = int(self.bandwidth / 2)
                    self.lock += lock_time
                else:
                    # The current gradients cannot be max_gradients_difference behind the current model
                    if central_server.gradients_received - self.gradients_index <= cfg['simulation']['max_gradients_difference']:
                        neural_network.optimizer.apply_gradients(zip(self.gradients, central_server.model.trainable_variables))
                        central_server.gradients_received += 1
                        lock_time = int(self.bandwidth / 2)
                        self.lock += lock_time
            else:
                neural_network.accumulate_gradients(central_server, self.gradients)
                central_server.gradients_received += 1
                lock_time = int(self.bandwidth / 2)
                self.lock += lock_time
            central_server.assigned_data.remove(self.data_index)
            central_server.received_data.add(self.data_index)
        central_server.bounded_staleness += 1
        self.upload_complete = True

    # Assume the car uploads the gradients to one RSU for the RSU to accumulate the received gradients
    def upload_gradients_to_rsu(self, rsu_list, central_server):
        # If the gradients is still from the last epoch
        if self.data_epoch != central_server.num_epoch:
            self.free_up()
            return
        closest_rsu = self.closest_rsu(rsu_list)
        if closest_rsu:
            closest_rsu.accumulative_gradients[self.data_index] = self.gradients
            closest_rsu.num_accumulative_gradients += 1
            self.upload_complete = True
            lock_time = int(self.bandwidth / 4)
            self.lock += lock_time

    def upload_completed(self):
        return self.upload_complete

    # Check if the vehicle is going out of the simulation in the next timestep
    def out_of_bounds(self, root, timestep):
        current_timestep = float(timestep.attrib['time'])
        next_timestep = root.find('timestep[@time="{:.2f}"]'.format(current_timestep+1))
        if next_timestep == None:
            return False
        else:
            id_set = set(map(lambda vehicle: vehicle.attrib['id'], next_timestep.findall('vehicle')))
            return not self.car_id in id_set

    # Return a list of all the vehicles within the range of the vehicle
    def in_range_vehicle(self, timestep):
        vehicles_in_range = []
        for vehicle in timestep.findall('vehicle'):
            distance = math.sqrt((float(vehicle.attrib['x']) - self.x) ** 2 + (float(vehicle.attrib['y']) - self.y) ** 2)
            if distance <= cfg['comm_range']['v2v']:
                vehicles_in_range.append(vehicle)
        return vehicles_in_range

    # Transfer all training data and labels to one near-by vehicle
    def transfer_data_to_vehicle(self, simulation, timestep):
        vehicles_in_range = self.in_range_vehicle(timestep)
        for vehicle in vehicles_in_range:
            if vehicle.attrib['id'] not in simulation.vehicle_dict:
                simulation.add_into_vehicle_dict(vehicle)
                vehi = simulation.vehicle_dict[vehicle.attrib['id']]
                vehi.training_data_assigned = self.training_data_assigned
                vehi.num_training_data_downloaded = self.num_training_data_downloaded
                vehi.training_label_assigned = self.training_label_assigned
                vehi.num_training_label_downloaded = self.num_training_label_downloaded
                vehi.rsu_assigned = self.rsu_assigned
                vehi.model = self.model
                vehi.data_index = self.data_index
                vehi.data_epoch = self.data_epoch
                vehi.gradients_index = self.gradients_index
                return True
            elif not simulation.vehicle_dict[vehicle.attrib['id']].training_data_assigned:
                vehi = simulation.vehicle_dict[vehicle.attrib['id']]
                vehi = simulation.vehicle_dict[vehicle.attrib['id']]
                vehi.training_data_assigned = self.training_data_assigned
                vehi.num_training_data_downloaded = self.num_training_data_downloaded
                vehi.training_label_assigned = self.training_label_assigned
                vehi.num_training_label_downloaded = self.num_training_label_downloaded
                vehi.rsu_assigned = self.rsu_assigned
                vehi.model = self.model
                vehi.data_index = self.data_index
                vehi.data_epoch = self.data_epoch
                vehi.gradients_index = self.gradients_index
                return True
        return False

    # Give assigned data and label back to RSU if the car is about to exit
    def transfer_data_to_rsu(self, cloest_rsu):
        cloest_rsu.dataset.append((self.data_index, (self.training_data_assigned, self.training_label_assigned)))

    # Give assigned data and label back to central server if the car is about to exit
    def transfer_data_to_central_server(self, central_server):
        central_server.train_dataset.append((self.data_index, (self.training_data_assigned, self.training_label_assigned)))

    # Transfer data either through near-by vehicles, RSUs, or central servers when the car is about to exit
    def transfer_data(self, simulation, timestep):
        # if self.rsu_assigned:
        #     if not self.transfer_data_to_vehicle:
        #         closest_rsu = self.closest_rsu(simulation.rsu_list)
        #         if closest_rsu:
        #             self.transfer_data_to_rsu(closest_rsu)
        #         else:
        #             self.transfer_data_to_central_server(simulation.central_server)
        if self.rsu_assigned:
            if self.compute_completed():
                simulation.central_server.bounded_staleness += 1
            self.transfer_data_to_central_server(simulation.central_server)
        # if self.rsu_assigned:
        #     if not self.transfer_data_to_vehicle(simulation, timestep):
        #         if self.gradients is not None:
        #             simulation.central_server.bounded_staleness += 1


    def locked(self):
        return self.lock > 0

    def update_lock(self):
        self.lock -= 1

    def free_up(self):
        self.training_data_assigned = []
        self.num_training_data_downloaded = 0
        self.training_label_assigned = []
        self.num_training_label_downloaded = 0
        self.rsu_assigned = None
        self.model = None
        self.gradients = None
        self.upload_complete = False
        self.data_index = -1
        self.data_epoch = 0
        self.data_length = 0
        self.gradients = None
        self.gradients_index = None
        

class RSU:
    """
    Road Side Unit object for Car ML Simulator.
    Attributes:
    - rsu_id
    - rsu_x
    - rsu_y
    - rsu_range
    - dataset
    - model
    - accumulative_gradients
    - num_accumulative_gradients
    - vehicle_traffic
    - traffic_proportion
    """
    def __init__(self, rsu_id, rsu_x, rsu_y, rsu_range, traffic_proportion):
        self.rsu_id = rsu_id
        self.rsu_x = rsu_x
        self.rsu_y = rsu_y
        self.rsu_range = rsu_range
        self.dataset = deque()
        self.model = None
        self.accumulative_gradients = {}
        self.num_accumulative_gradients = 0
        self.vehicle_traffic = 0
        self.traffic_proportion = traffic_proportion

    def low_on_data(self):
        return len(self.dataset) < 10

    # Check if the RSU has received the specified amount of gradients
    def max_gradients_accumulated(self):
        return self.num_accumulative_gradients >= cfg['simulation']['maximum_rsu_accumulative_gradients']

    # When the RSU has sent out all of its data but is stilling receiving gradients
    # Usually toward the end of an epoch. Edge case
    def dataset_empty(self):
        return not self.dataset and self.accumulative_gradients

    def verify_redundancy(self, central_server):
        for i in list(self.accumulative_gradients.keys()):
            if i in central_server.assigned_data:
                central_server.assigned_data.remove(i)
                central_server.received_data.add(i)
            else:
                del self.accumulative_gradients[i]
                self.num_accumulative_gradients -= 1

    # The RSU updates the model in the central server with its accumulative gradients and downloads the 
    # latest model from the central server
    def communicate_with_central_server(self, central_server):
        self.verify_redundancy(central_server)
        if self.accumulative_gradients:
            neural_network = Neural_Network()
            num_accumulative_gradients = len(self.accumulative_gradients)
            accumulative_gradients = neural_network.accumulate_gradients_itr(self.accumulative_gradients.values())
            accumulative_gradients = np.true_divide(accumulative_gradients, num_accumulative_gradients)
            gradient_zip = zip(accumulative_gradients, central_server.model.trainable_variables)
            neural_network.optimizer.apply_gradients(gradient_zip)
            central_server.gradients_received += num_accumulative_gradients
            self.model = central_server.model
            self.accumulative_gradients = {}
            self.num_accumulative_gradients = 0


class Central_Server:
    """
    Central Server object for Car ML Simulator.
    Attributes:
    - dataset
    - num_mini_batches
    - train_dataset
    - test_dataset
    - model
    - epoch_loss_avg
    - epoch_accuracy
    - num_epoch
    - rsu_list
    - accumulative_gradients
    - gradients_received
    - bounded_staleness
    - assigned_data
    - received_data
    """
    def __init__(self, rsu_list):
        train, test = tf.keras.datasets.mnist.load_data()

        # Normalize the training data to fit the model
        train_images, train_labels = train
        num_training_data = cfg['simulation']['num_training_data']
        train_images, train_labels = train_images[:num_training_data], train_labels[:num_training_data]
        train_images = train_images.reshape(train_images.shape[0], 784)
        train_images = train_images/255

        # Normalize the testing data to fit the model
        test_images, test_labels = test
        test_images, test_labels = test_images, test_labels
        test_images = test_images.reshape(test_images.shape[0], 784)
        test_images = test_images/255

        batch_size = cfg['neural_network']['batch_size']
        self.dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(int(num_training_data/batch_size)).batch(batch_size)
        self.num_mini_batches = len(list(self.dataset))
        self.train_dataset = []
        self.train_dataset_index = []
        self.test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)
        # The structure of the neural network
        self.model = tf.keras.Sequential([
                    tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape=(784,)),  # input shape required
                    # tf.keras.layers.Dense(10, activation=tf.nn.relu),
                    tf.keras.layers.Dense(10)
        ])
        self.epoch_loss_avg = None
        self.epoch_accuracy = None
        self.num_epoch = 0
        self.rsu_list = rsu_list
        self.accumulative_gradients = None
        self.gradients_received = 0     # Number of gradients received from vehicle in each epoch (Async)
        self.bounded_staleness = cfg['simulation']['bounded_staleness']

        self.assigned_data = set()
        self.received_data = set()

    # Initially distribute 1/4 of the data to each RSU
    def distribute_to_rsu(self):
        for rsu in self.rsu_list:
            initial_distribution = int(0.25 * self.num_mini_batches * rsu.traffic_proportion)+1
            data = self.train_dataset[:initial_distribution]
            indexs = set(map(lambda x: x[0], data))
            rsu.dataset = deque(data)
            self.assigned_data |= indexs
            rsu.model = self.model
            del self.train_dataset[:initial_distribution]
    
    # Redistribute part of the data to RSU when the RSU is running low on data
    def redistribute_to_rsu(self, rsu):
        num_redistributed = int(0.25 * self.num_mini_batches * rsu.traffic_proportion)
        if self.train_dataset:
            data = self.train_dataset[:num_redistributed]
            indexs = set(map(lambda x: x[0], data))
            rsu.dataset.extend(data)
            self.assigned_data |= indexs
            del self.train_dataset[:num_redistributed]
            if len(indexs) >= num_redistributed:
                return
            else:
                num_redistributed -= len(indexs)
        if self.assigned_data:
            for _ in range(num_redistributed):
                rsu.dataset.append(self.train_dataset_index[random.choice(tuple(self.assigned_data))])

    def epoch_completed(self):
        return len(self.received_data) == self.num_mini_batches
        # return self.gradients_received >= self.num_mini_batches

    # Update the model with its accumulative gradients
    # Used for batch gradient descent
    def update_model(self):
        if self.accumulative_gradients is not None:
            neural_network = Neural_Network()
            self.accumulative_gradients = np.true_divide(self.accumulative_gradients, self.gradients_received)
            gradient_zip = zip(self.accumulative_gradients, self.model.trainable_variables)
            neural_network.optimizer.apply_gradients(gradient_zip)

    def print_accuracy(self):
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(self.num_epoch,
                                                                self.epoch_loss_avg.result(),
                                                                self.epoch_accuracy.result()))

    def new_epoch(self):
        # Need to loop through the dataset each epoch to shuffle the data
        self.train_dataset = []
        for i, (x, y) in self.dataset.enumerate().as_numpy_iterator():
            for _ in range(cfg['simulation']['replication_factor']):
                self.train_dataset.append((i,(x.tolist(), y.tolist())))
            self.train_dataset_index.append((i,(x.tolist(), y.tolist())))
        self.epoch_loss_avg = tf.keras.metrics.Mean()
        self.epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.num_epoch += 1
        self.assigned_data = set()
        self.received_data = set()
        self.distribute_to_rsu()
        self.gradients_received = 0


class Neural_Network:
    """
    Neural network functions
    Attributes:
    - optimizer
    """
    def __init__(self):
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=cfg['neural_network']['learning_rate'])

    # The loss function
    def loss(self, model, x, y, training):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        y_ = model(x, training=training)
        return loss_object(y_true=y, y_pred=y_)
    
    # Gradients and loss
    def grad(self, model, inputs, targets, central_server):
        with tf.GradientTape() as tape:
            loss_value = self.loss(model, inputs, targets, training=True)
            central_server.epoch_loss_avg.update_state(loss_value)
            central_server.epoch_accuracy.update_state(targets, central_server.model(inputs, training=True))
        return tape.gradient(loss_value, model.trainable_variables)

    # Function used to aggregate gradient values into one
    def accumulate_gradients(self, dest, step_gradients):
        if dest.accumulative_gradients is None:
            dest.accumulative_gradients = [self.flat_gradients(g) for g in step_gradients]
        else:
            for i, g in enumerate(step_gradients):
                dest.accumulative_gradients[i] += self.flat_gradients(g) 

    # Function used to aggregate gradient values into one
    def accumulate_gradients_itr(self, step_gradients):
        accumulative_gradients = []
        for x in step_gradients:
            if not accumulative_gradients:
                accumulative_gradients = [self.flat_gradients(g) for g in x]
            else:
                for i, g in enumerate(x):
                    accumulative_gradients[i] += self.flat_gradients(g) 
        return accumulative_gradients

    # Helper function for accumulate_gradients()
    def flat_gradients(self, grads_or_idx_slices):
        if type(grads_or_idx_slices) == tf.IndexedSlices:
            return tf.scatter_nd(
                tf.expand_dims(grads_or_idx_slices.indices, 1),
                grads_or_idx_slices.values,
                grads_or_idx_slices.dense_shape
            )
        return grads_or_idx_slices


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

    def rsuList_random(self, rsu_range, rsu_nums):
        tree = ET.parse(self.NET_file)
        root = tree.getroot()
        rsu_list = []
        junction_list = np.random.choice(root.findall('junction'), rsu_nums, replace=False)
        for i in range(rsu_nums):
            id = 'rsu' + str(i)
            rsu_list.append(RSU(id, float(junction_list[i].attrib['x']), float(junction_list[i].attrib['y']), rsu_range, 1/cfg['simulation']['num_rsu']))
        return rsu_list

    def rsuList(self, rsu_range, rsu_nums, junction_list):
        rsu_list = []
        for i in range(rsu_nums):
            id = 'rsu' + str(i)
            rsu_list.append(RSU(id, float(junction_list[i][0].attrib['x']), float(junction_list[i][0].attrib['y']), rsu_range, junction_list[i][1]))
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
    def __init__(self, FCD_file, vehicle_dict: dict, rsu_list: list, central_server):
        self.FCD_file = FCD_file
        self.vehicle_dict = vehicle_dict
        self.rsu_list = rsu_list
        self.central_server = central_server
       
    def add_into_vehicle_dict(self, vehicle):
        self.vehicle_dict[vehicle.attrib['id']] = Vehicle(vehicle.attrib['id'],
                                                          cfg['simulation']['comp_power'],
                                                          cfg['simulation']['comp_power_std'], 
                                                          cfg['simulation']['bandwidth'],
                                                          cfg['simulation']['bandwidth_std'])