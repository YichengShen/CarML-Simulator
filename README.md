# Car ML Simulator

## Sumo Output
In Terminal:  
1. cd into the directory
2. ```sumo -c osm.sumocfg --fcd-output 'filename'.xml```

## Classes:
- Vehicle
  - car_id
  - comp_power
  - bandwidth
- RSU (Road Side Unit)
  - rsu_id
  - location_x
  - location_y
- Dataset
  - name
  - size
- Sample
  - s_id
  - dataset (parent)
  - size
