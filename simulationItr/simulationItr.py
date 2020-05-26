import numpy as np 
import xml.etree.ElementTree as ET 
from simulationItrClass import Dataset 
from simulationItrClass import Simulation

def simulate(simulation):
    tree = ET.parse(simulation.FCD_file)
    root = tree.getroot()
    totalTasks = simulation.num_tasks

    # For each time step (sec) in the FCD file  
    for timestep in root:
        # For each vehicle on the map at the timestep
        for vehicle in timestep.findall('vehicle'):
            vehi = simulation.vehicleDict[vehicle.attrib['id']]
            vehi.x = float(vehicle.attrib['x'])
            vehi.y = float(vehicle.attrib['y'])
            vehi.speed = float(vehicle.attrib['speed'])
            # Range Check
            inRangeX = False
            inRangeY = False
            for x_min, x_max in simulation.rsu_x_range:
                if x_min <= vehi.x <= x_max:
                    inRangeX = True
            for y_min, y_max in simulation.rsu_y_range:
                if y_min <= vehi.y <= y_max:
                    inRangeY = True
            # If in range
            if inRangeX and inRangeY:
                # If finish downloading
                if vehi.downloaded():
                    vehi.upload_time = vehi.comm_time
                    if vehi.tasks_remaining > 0:
                        vehi.tasks_remaining -= vehi.comp_power
                    # If finish assigned tasks
                    if vehi.tasks_remaining <= 0:
                        # If finish uploading
                        if vehi.uploaded():
                            vehi.tasks_remaining = vehi.tasks_distributed
                            vehi.upload_time = vehi.comm_time
                            simulation.num_tasks -= vehi.tasks_distributed
                            # If all tasks are finished
                            if simulation.num_tasks <= 0:
                                print("\nAll {} tasks were completed in {} units of time.\n".format(totalTasks, timestep.attrib['time']))
                                return
    # If some tasks aren't finished after running through all the timestpes
    print("\nAll vehicles left the RSU ranges when {} tasks were left.\n".format(simulation.num_tasks))


def main():
    ROU_FILE = 'osm_boston_common/osm.passenger.trips.xml'
    NET_FILE = 'osm_boston_common/osm.net.xml'
    FCD_FILE = 'osm_boston_common/osm_fcd.xml'
    NUM_TASKS = 300000    # number of tasks
    COMP_POWER = 5        # computation power of cars
    COMP_POWER_STD = 1    # standard deviation
    BANDWIDTH = 10        # bandwidth of cars
    BANDWIDTH_STD = 2     # standard deviation
    RSU_RANGE = 300       # range of RSU
    NUM_RSU = 2           # number of RSU

    data = Dataset(ROU_FILE, NET_FILE)
    vehicleDict = data.vehicleDict(COMP_POWER, COMP_POWER_STD, BANDWIDTH, BANDWIDTH_STD)
    rsuRange = data.RSURangeList(RSU_RANGE, NUM_RSU)
    simulation = Simulation(FCD_FILE, vehicleDict, rsuRange, NUM_TASKS)
    simulate(simulation)

if __name__ == '__main__':
    main()
