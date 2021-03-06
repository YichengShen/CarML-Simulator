import xml.etree.ElementTree as ET
from simulationItrClass import *

def simulate(simulation):
    tree = ET.parse(simulation.FCD_file)
    root = tree.getroot()
    total_tasks = simulation.num_tasks

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
            # Download if not finished downloading
            if not vehi.download_complete():
                vehi.download_from_rsu(simulation.rsu_list)
                # If out of range of the assigned rsu
                if vehi.out_of_range():
                    vehi.update_time_left_rsu()
                else:
                    vehi.refresh_time_left_rsu()
                # If time after leaving rsu > N
                if vehi.time_left_rsu > cfg['timer']['time_remove_rsu_assigned']:
                    vehi.remove_tasks_and_rsu()
            # If finished downloading
            else:
                # Compute when there are still tasks left
                if not vehi.compute_complete():
                    vehi.compute()
                    # Transfer data if vehicle is about to go out of bounds
                    if vehi.out_of_bounds(root, timestep):
                        vehi.transfer_data(simulation, timestep)
                # If finished computing
                else:
                    # Transfer data if vehicle is about to go out of bounds
                    if vehi.out_of_bounds(root, timestep):
                        vehi.transfer_data(simulation, timestep)
                    # Upload if not finished uploading
                    if not vehi.upload_complete():
                        vehi.upload_to_rsu(simulation.rsu_list)
                    else:
                        simulation.num_tasks -= (len(vehi.tasks_assigned)-1) # Update number of tasks left
                        vehi.free_up()
            # If all tasks are finished
            if simulation.num_tasks <= 0:
                print("\nAll {} tasks were completed in {} units of time.\n".format(total_tasks, timestep.attrib['time']))
                return
    # If some tasks aren't finished after running through all the timestpes
    print("\nAll vehicles left the RSU ranges when {} tasks were left.\n".format(simulation.num_tasks))


def main():
    ROU_FILE = cfg['simulation']['ROU_FILE']
    NET_FILE = cfg['simulation']['NET_FILE']
    FCD_FILE = cfg['simulation']['FCD_FILE']
    
    NUM_TASKS = cfg['simulation']['num_tasks']       # number of tasks
    RSU_RANGE = cfg['comm_range']['v2rsu']           # range of RSU
    NUM_RSU = cfg['simulation']['num_rsu']           # number of RSU

    sumo_data = SUMO_Dataset(ROU_FILE, NET_FILE)
    # vehicle_dict = data.vehicleDict(COMP_POWER, COMP_POWER_STD, BANDWIDTH, BANDWIDTH_STD)
    vehicle_dict = {}

    data_to_learn = Task_Set(1, NUM_TASKS)
    sample_dict = data_to_learn.partition_data(NUM_RSU)

    rsu_list = sumo_data.rsuList(RSU_RANGE, NUM_RSU, sample_dict)

    simulation = Simulation(FCD_FILE, vehicle_dict, rsu_list, NUM_TASKS)
    simulate(simulation)

if __name__ == '__main__':
    main()
