import xml.etree.ElementTree as ET 
import simpy

from simulationClass import Simulation, SUMOfile

def run_simulation(s):
    start_time = s.env.now

    tree = ET.parse(s.FCD_file)
    root = tree.getroot()

    # VARIABLES to keep track of time
    previous_max_time = 0
    max_time = 0

    #BIG LOOP
    for timestep in root:
        time = 0
        for each_vehicle in timestep.findall('vehicle'):
            vehicle = s.vDict[each_vehicle.attrib['id']]
            vehicle.x = float(each_vehicle.attrib['x'])
            vehicle.y = float(each_vehicle.attrib['y'])
            vehicle.speed = float(each_vehicle.attrib['speed'])
            # RANGE CHECK
            inRangeX = False
            inRangeY = False
            for x_min, x_max in s.rsu_x_range:
                if x_min <= vehicle.x <= x_max:
                    inRangeX = True
            for y_min, y_max in s.rsu_y_range:
                if y_min <= vehicle.y <= y_max:
                    inRangeY = True

            # if in range and tasks not finished
            if inRangeX and inRangeY and not s.tasks_completed():
                # UPDATE total time
                if vehicle.downloaded():
                    time = max(time, vehicle.bandwidth)
                elif vehicle.computed():
                    time = max(time, vehicle.comp_power)
                elif vehicle.uploaded():
                    time = max(time, vehicle.bandwidth)
                # UPDATE task timer for each vehicle and UPDATE tasks left
                if vehicle.uploaded():
                    s.num_tasks_left -= 1
                    vehicle.task_timer = 0
                else:
                    vehicle.task_timer += 1

        previous_max_time = max_time
        max_time = max(int(timestep.attrib['time'][:-3]) + time, max_time)
        diff = max_time - previous_max_time

        # UPDATE total time
        yield s.env.timeout(diff)

        if s.tasks_completed():
            break

    total_time = s.env.now - start_time
    print("\nTotal Time: {} units of time".format(total_time))
    print("Tasks Left: {} tasks\n".format(s.num_tasks_left))
    if s.tasks_completed():
        print("All {} tasks were completed in {} units of time.".format(s.num_tasks, total_time))
    else:
        print("All vehicles left the RSU ranges when {} tasks were left.\n".format(s.num_tasks_left))


def main():
    # SET UP
    ROU_FILE = 'test_data_boston_common/osm.passenger.trips.xml'
    NET_FILE = 'test_data_boston_common/osm.net.xml'
    FCD_FILE = 'test_data_boston_common/osm_fcd.xml'
    NUM_TASKS = 1000    # number of tasks
    COMP_TIME = 5       # units of time to compute one task
    COMP_TIME_STD = 1   # standard deviation
    COMM_TIME = 10      # units of time to download/upload one task
    COMM_TIME_STD = 2   # standard deviation
    RSU_RANGE = 300     # range of RSU
    NUM_RSU = 2         # number of RSU

    # PROCESS SUMO DATA
    sumo_data = SUMOfile(ROU_FILE, NET_FILE)
    vehicle_dict = sumo_data.make_vehicleDict(COMP_TIME, COMP_TIME_STD, COMM_TIME, COMM_TIME_STD)
    rsu_range = sumo_data.make_RSURangeList(RSU_RANGE, NUM_RSU)

    # RUN THE SIMULATION
    env = simpy.Environment()
    simulation = Simulation(env, FCD_FILE, vehicle_dict, rsu_range, NUM_TASKS)
    env.process(run_simulation(simulation))
    env.run()


if __name__ == '__main__':
    main()
