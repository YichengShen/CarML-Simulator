import math
import heapq
import numpy as np
import yaml
import xml.etree.ElementTree as ET 
import tensorflow as tf
import tensorflow_datasets as tfds


file = open('config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)

np.random.seed(3)

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
    - data_length
    - gradients
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
        self.data_length = 0
        self.gradients = None
        self.upload_complete = False
        self.lock = 0

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
    # with the first element of the list being the RSU that is cloest to the vehicle
    def in_range_rsus(self, rsu_list):
        in_range_rsus = []
        for rsu in rsu_list:
            distance = math.sqrt((rsu.rsu_x - self.x) ** 2 + (rsu.rsu_y - self.y) ** 2)
            if distance <= rsu.rsu_range:
                heapq.heappush(in_range_rsus, (distance, rsu))
        return list(map(lambda x: x[1], in_range_rsus))

    def download_from_rsu(self, rsu_list):
        if self.rsu_assigned is None:
            for rsu in rsu_list:
                if rsu.dataset:
                    self.rsu_assigned = rsu
                    rsu.vehicle_traffic += 1
                    dataset = rsu.dataset.pop()
                    self.training_data_assigned = dataset[0]
                    self.training_label_assigned = dataset[1]
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

    # Assume the car downloads the model directly from the central server when it is computing
    # After completing, store the gradients
    def compute_from_central_server(self, central_server):
        if central_server.bounded_staleness > 0:
            neural_network = Neural_Network()
            self.gradients = neural_network.grad(central_server.model, np.array(self.training_data_assigned), np.array(self.training_label_assigned), central_server)
            central_server.bounded_staleness -= 1
            lock_time = int(self.bandwidth / 2) + int(self.data_length / self.comp_power) - 1
            self.lock += lock_time

    # Assume the car downloads the model from the RSU and use it to compute the gradients
    def compute_from_rsu(self, central_server):
        neural_network = Neural_Network()
        self.gradients = neural_network.grad(self.model, np.array(self.training_data_assigned), np.array(self.training_label_assigned), central_server)
        lock_time = int(self.data_length / self.comp_power) - 1
        self.lock += lock_time

    def compute_completed(self):
        return self.gradients is not None

    # Assume the car directly uploads the gradients to the central server and update the model of central server
    def upload_gradients_to_central_server(self, central_server):
        neural_network = Neural_Network()
        neural_network.optimizer.apply_gradients(zip(self.gradients, central_server.model.trainable_variables))
        central_server.gradients_received += 1
        central_server.bounded_staleness += 1
        self.upload_complete = True
        lock_time = int(self.bandwidth / 2)
        self.lock += lock_time

    # Assume the car uploads the gradients to one RSU for the RSU to accumulate the received gradients
    def upload_gradients_to_rsu(self, rsu_list):
        closest_rsu = self.closest_rsu(rsu_list)
        if closest_rsu:
            closest_rsu.accumulate_gradients(self.gradients)
            closest_rsu.num_accumulative_gradients += 1
            self.upload_complete = True
            lock_time = int(self.bandwidth / 4)
            self.lock += lock_time

    def upload_completed(self):
        return self.upload_complete

    def out_of_bounds(self, root, timestep):
        current_timestep = float(timestep.attrib['time'])
        next_timestep = root.find('timestep[@time="{:.2f}"]'.format(current_timestep+1))
        if next_timestep == None:
            return False
        else:
            id_set = set(map(lambda vehicle: vehicle.attrib['id'], next_timestep.findall('vehicle')))
            return not self.car_id in id_set

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
                return True
            elif not simulation.vehicle_dict[vehicle.attrib['id']].training_data_assigned:
                vehi = simulation.vehicle_dict[vehicle.attrib['id']]
                vehi = simulation.vehicle_dict[vehicle.attrib['id']]
                vehi.training_data_assigned = self.training_data_assigned
                vehi.num_training_data_downloaded = self.num_training_data_downloaded
                vehi.training_label_assigned = self.training_label_assigned
                vehi.num_training_label_downloaded = self.num_training_label_downloaded
                vehi.rsu_assigned = self.rsu_assigned
                return True
        return False

    # Give assigned data and label back to RSU if the car is about to exit
    def transfer_data_to_rsu(self, cloest_rsu):
        cloest_rsu.dataset.append((self.training_data_assigned, self.training_label_assigned))

    # Give assigned data and label back to central server if the car is about to exit
    def transfer_data_to_central_server(self, central_server):
        central_server.train_dataset.append((self.training_data_assigned, self.training_label_assigned))

    # Transfer data either through near-by vehicles, RSUs, or central servers when the car is about to exit
    def transfer_data(self, simulation):
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
    def __init__(self, rsu_id, rsu_x, rsu_y, rsu_range):
        self.rsu_id = rsu_id
        self.rsu_x = rsu_x
        self.rsu_y = rsu_y
        self.rsu_range = rsu_range
        self.dataset = []
        self.model = None
        self.accumulative_gradients = None
        self.num_accumulative_gradients = 0
        self.vehicle_traffic = 0
        self.traffic_proportion = 1 / cfg['simulation']['num_rsu']

    def low_on_data(self):
        return len(self.dataset) < 10

    # Function used to aggregate gradient values into one
    def accumulate_gradients(self,step_gradients):
        if self.accumulative_gradients is None:
            self.accumulative_gradients = [self.flat_gradients(g) for g in step_gradients]
        else:
            for i, g in enumerate(step_gradients):
                self.accumulative_gradients[i] += self.flat_gradients(g) 

    # Helper function for accumulate_gradients()
    def flat_gradients(self, grads_or_idx_slices):
        if type(grads_or_idx_slices) == tf.IndexedSlices:
            return tf.scatter_nd(
                tf.expand_dims(grads_or_idx_slices.indices, 1),
                grads_or_idx_slices.values,
                grads_or_idx_slices.dense_shape
            )
        return grads_or_idx_slices

    def max_gradients_accumulated(self):
        return self.num_accumulative_gradients >= cfg['simulation']['maximum_rsu_accumulative_gradients']

    def dataset_empty(self):
        return not self.dataset and self.accumulative_gradients

    # The RSU updates the model in the central server with its accumulative gradients and downloads the 
    # latest model from the central server
    def communicate_with_central_server(self, central_server):
        neural_network = Neural_Network()
        gradient_zip = zip(self.accumulative_gradients, central_server.model.trainable_variables)
        neural_network.optimizer.apply_gradients(gradient_zip)
        central_server.gradients_received += self.num_accumulative_gradients
        self.model = central_server.model
        self.accumulative_gradients = None
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
    - num_distributed
    - gradients_received
    - bounded_staleness
    """
    def __init__(self, rsu_list):
        train, test = tf.keras.datasets.mnist.load_data()

        # Normalize the training data to fit the model
        train_images, train_labels = train
        train_images, train_labels = train_images[:10000], train_labels[:10000]
        train_images = train_images.reshape(train_images.shape[0], 784)
        train_images = train_images/255

        # Normalize the testing data to fit the model
        test_images, test_labels = test
        test_images, test_labels = test_images[:10000], test_labels[:10000]
        test_images = test_images.reshape(test_images.shape[0], 784)
        test_images = test_images/255

        self.dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(200).batch(cfg['neural_network']['batch_size'])
        self.num_mini_batches = len(list(self.dataset))
        self.train_dataset = []
        self.test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(cfg['neural_network']['batch_size'])
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
        self.num_distributed = 0        # Amount of mini-batches already distributed out to RSUs
        self.gradients_received = 0     # Number of gradients received from vehicle in each epoch (Async)
        self.bounded_staleness = cfg['simulation']['bounded_staleness']

    # Initially distribute 1/4 of the data to each RSU
    def distribute_to_rsu(self, degree_of_overlap = 0):
        for rsu in self.rsu_list:
            initial_distribution = int(0.25 * self.num_mini_batches * rsu.traffic_proportion)
            initial_overlapping = int(degree_of_overlap * self.num_mini_batches)
            total_distribution = initial_distribution + initial_overlapping
            rsu.dataset.extend(self.train_dataset[:total_distribution])
            rsu.model = self.model
            del self.train_dataset[:total_distribution]
    
    # Redistribute part of the data to RSU when the RSU is running low on data
    def redistribute_to_rsu(self, rsu):
        if self.train_dataset:
            num_redistributed = int(0.25 * self.num_mini_batches * rsu.traffic_proportion)
            rsu.dataset.extend(self.train_dataset[:num_redistributed])
            del self.train_dataset[:num_redistributed]

    def epoch_completed(self):
        return self.gradients_received >= self.num_mini_batches

    def print_accuracy(self):
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(self.num_epoch,
                                                                self.epoch_loss_avg.result(),
                                                                self.epoch_accuracy.result()))

    def new_epoch(self):
        # Need to loop through the dataset each epoch to shuffle the data
        for x in self.dataset:
            self.train_dataset.append(x)
        self.epoch_loss_avg = tf.keras.metrics.Mean()
        self.epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.num_epoch += 1
        self.distribute_to_rsu()
        self.num_distributed = 0
        self.gradients_received = 0


class Neural_Network:
    """
    Neural network functions
    Attributes:
    - optimizer
    """
    def __init__(self):
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

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