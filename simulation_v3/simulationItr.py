from simulationItrClass import *

import xml.etree.ElementTree as ET
from sklearn.datasets import load_breast_cancer


def simulate(simulation):
    # Preparation - Distribute data from Cloud Server to RSU
    cloud_server = Cloud_Server(simulation.dataset, simulation.rsu_list)
    cloud_server.distribute_to_rsu()

    for rsu in simulation.rsu_list:
        print("# data on", rsu, len(rsu.data))
        print("# target on", rsu, len(rsu.target))
        print()

    tree = ET.parse(simulation.FCD_file)
    root = tree.getroot()
    total_num_data = simulation.dataset.num_tasks

    # For each time step (sec) in the FCD file  
    for timestep in root:
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

            vehi.download_from_rsu(simulation.rsu_list)

    for rsu in simulation.rsu_list:
        print("# downloaded data on", rsu, len(rsu.data_downloaded))
        print("# downloaded target on", rsu, len(rsu.target_downloaded))
        print()



    #         # Tell RSU to allow other vehicles to download its tasks if vehicle is about to go out of bounds
    #         if vehi.out_of_bounds(root, timestep):
    #             vehi.unlock_downloaded_data()

            # Download if not finished downloading
            # if not vehi.download_completed():
            #     vehi.download_from_rsu(simulation.rsu_list)
    #         # If finished downloading
    #         else:
    #             # Compute when there are still tasks left
    #             if not vehi.compute_completed():
    #                 vehi.compute()
    #             # If finished compute
    #             else:
    #                 # Upload if not finished uploading
    #                 if not vehi.upload_completed():
    #                     vehi.upload(simulation.rsu_list, cloud_server)
    #                 # If finished upload
    #                 else:
    #                     vehi.free_up()

    # print(total_num_data - len(list(set(cloud_server.data_id_list_finished))))


   
    #                 # Transfer data if vehicle is about to go out of bounds
    #                 if vehi.out_of_bounds(root, timestep):
    #                     vehi.transfer_data(simulation, timestep)
    #             # If finished computing
    #             else:
    #                 # Transfer data if vehicle is about to go out of bounds
    #                 if vehi.out_of_bounds(root, timestep):
    #                     vehi.transfer_data(simulation, timestep)
    #                 # Upload if not finished uploading
    #                 if not vehi.upload_complete():
    #                     vehi.upload_to_rsu(simulation.rsu_list)
    #                 else:
    #                     simulation.num_tasks -= (len(vehi.tasks_assigned)-1) # Update number of tasks left
    #                     vehi.free_up()
    #         # If all tasks are finished
    #         if simulation.num_tasks <= 0:
    #             print("\nAll {} tasks were completed in {} units of time.\n".format(total_tasks, timestep.attrib['time']))
    #             return
    # # If some tasks aren't finished after running through all the timestpes
    # print("\nAll vehicles left the RSU ranges when {} tasks were left.\n".format(simulation.num_tasks))


def main():
    ROU_FILE = cfg['simulation']['ROU_FILE']
    NET_FILE = cfg['simulation']['NET_FILE']
    FCD_FILE = cfg['simulation']['FCD_FILE']
    
    RSU_RANGE = cfg['comm_range']['v2rsu']           # range of RSU
    NUM_RSU = cfg['simulation']['num_rsu']           # number of RSU

    sumo_data = SUMO_Dataset(ROU_FILE, NET_FILE)
    vehicle_dict = {}

    # Load dataset from sklearn
    data_breast_cancer = load_breast_cancer()
    X = data_breast_cancer.data
    y = data_breast_cancer.target

    dataset_to_learn = Training_Dataset(1, X, y)

    rsu_list = sumo_data.rsuList(RSU_RANGE, NUM_RSU)

    simulation = Simulation(FCD_FILE, vehicle_dict, rsu_list, dataset_to_learn)
    simulate(simulation)

if __name__ == '__main__':
    main()