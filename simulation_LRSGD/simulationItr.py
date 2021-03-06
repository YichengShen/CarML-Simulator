from simulationClass import *
from locationPicker_v3 import output_junctions
import xml.etree.ElementTree as ET 

def simulate(simulation):
    simulation.central_server.new_epoch()
    tree = ET.parse(simulation.FCD_file)
    root = tree.getroot()

    # For each time step (sec) in the FCD file 
    for timestep in root:
        
        # Maximum training epochs
        if simulation.central_server.num_epoch <= cfg['neural_network']['epoch']:
            # Calculate in-real-time RSU traffic every 10 minutes
            if timestep.attrib['time'] != '0.00' and float(timestep.attrib['time']) % 600 == 0:
                total_traffic = sum(map(lambda x: x.vehicle_traffic, simulation.rsu_list))
                for rsu in simulation.rsu_list:
                    rsu.traffic_proportion = rsu.vehicle_traffic / total_traffic
                    rsu.vehicle_traffic = 0

            for rsu in simulation.rsu_list:
                # Redistribute data from central server to RSU if RSU is about to run out of data
                if rsu.low_on_data():
                    simulation.central_server.redistribute_to_rsu(rsu)
                # Update the central server model and obtain the latest model when an RSU has accumulated
                # a centrain number of gradients
                if rsu.max_gradients_accumulated() or rsu.dataset_empty():
                    rsu.communicate_with_central_server(simulation.central_server)

            # When each epoch is completed, print accuracy of each epoch
            if simulation.central_server.epoch_completed():
                # simulation.central_server.update_model()
                simulation.central_server.print_mse()
                simulation.central_server.new_epoch()

            # For each vehicle on the map at the timestep
            for vehicle in timestep.findall('vehicle'):
                # If vehicle not yet stored in vehicle_dict
                if vehicle.attrib['id'] not in simulation.vehicle_dict:
                    simulation.add_into_vehicle_dict(vehicle)
                # Get the vehicle object from vehicle_dict
                vehi = simulation.vehicle_dict[vehicle.attrib['id']]  
                # Set location and speed
                vehi.set_properties(float(vehicle.attrib['x']),
                                    float(vehicle.attrib['y']),
                                    float(vehicle.attrib['speed']))
                # Download if not finished downloading
                if not vehi.download_completed():
                    rsus_in_range = vehi.in_range_rsus(simulation.rsu_list)
                    # Download from RSU if in range of RSU
                    if vehi.rsu_assigned is None or vehi.rsu_assigned in rsus_in_range:
                        vehi.download_from_rsu(rsus_in_range, simulation.central_server)
                    # Download from the central server if not in range of RSU
                    else:
                        vehi.download_from_central_server()
                # If finished downloading
                else:
                    # If the vehicle is still computing
                    if not vehi.locked():
                        # Compute the gradients using the latest model
                        if not vehi.compute_completed():
                            vehi.compute_gradients(simulation.central_server)
                        # If finished computing
                        else:
                            # Upload the gradients if not finished uploading
                            if not vehi.upload_completed():
                                # One needs to be commented out
                                vehi.upload_gradients_to_central_server(simulation.central_server)
                                # vehi.upload_gradients_to_rsu(simulation.rsu_list, simulation.central_server)
                            else:
                                vehi.free_up()
                    # If locked, lock -1 in every time step
                    else:
                        vehi.update_lock()
                # If the vehicle is about to exit the map(city), transfer its data to either
                # nearby vehicles, RSUs, or the central server
                if vehi.out_of_bounds(root, timestep) and not vehi.upload_completed():
                    vehi.transfer_data(simulation, timestep)
    return simulation.central_server.weight, simulation.central_server.bias


def main():
    ROU_FILE = cfg['simulation']['ROU_FILE']
    NET_FILE = cfg['simulation']['NET_FILE']
    FCD_FILE = cfg['simulation']['FCD_FILE']
    
    RSU_RANGE = cfg['comm_range']['v2rsu']           # range of RSU
    NUM_RSU = cfg['simulation']['num_rsu']           # number of RSU

    sumo_data = SUMO_Dataset(ROU_FILE, NET_FILE)
    vehicle_dict = {}
    rsu_list = sumo_data.rsuList(RSU_RANGE, NUM_RSU, output_junctions)
    # rsu_list = sumo_data.rsuList_random(RSU_RANGE, NUM_RSU)
    central_server = Central_Server(rsu_list)

    simulation = Simulation(FCD_FILE, vehicle_dict, rsu_list, central_server)
    weight, bias = simulate(simulation)

    # Test MSE
    x_test = central_server.test_images
    y_test = central_server.test_labels

    # MSE of prediction by Sklearn SGDRegressor
    from sklearn.linear_model import SGDRegressor
    clf_ = SGDRegressor(max_iter=100)
    clf_.fit(central_server.train_images, central_server.train_labels)
    y_pred_sksgd=clf_.predict(x_test)
    print('Final Mean Squared Error (sklearn) : {:.3f}'.format(mean_squared_error(y_test, y_pred_sksgd)))

    # MSE of prediction by distributed LR via SGD
    y_pred = []
    for i in range(len(x_test)):
        y = np.asscalar(np.dot(weight, x_test[i]) + bias)
        y_pred.append(y)
    y_pred = np.array(y_pred)
    print('Final Mean Squared Error (distributed) : {:.3f}'.format(mean_squared_error(y_test, y_pred)))
    

if __name__ == '__main__':
    main()
