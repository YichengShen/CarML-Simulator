# Car ML Simulator

## Sumo Output
In Terminal:  
1. cd into the directory
2. ```sumo -c osm.sumocfg --fcd-output 'filename'.xml```

## Classes:
- Simulation
  - env
  - FCD_file
  - vDict
  - rsu_x_range
  - rsu_y_range
  - num_tasks
  - num_tasks_left
- Vehicle
  - car_id
  - x
  - y
  - speed
  - comp_power
  - bandwidth
  - task_timer
- RSU (Road Side Unit)
  - rsu_id
  - location_x
  - location_y
  - rsu_range
  - rsu_x_range
  - rsu_y_range
- SUMOfile
  - ROU_file
  - NET_file
- Dataset
  - name
  - size
- Sample
  - s_id
  - dataset (parent)
  - size
