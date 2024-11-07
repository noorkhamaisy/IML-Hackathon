# Machine Learning Hackathon
This project focuses on improving public transportation in Israel using machine learning. 

## HU.BER transportation
HU.BER transportation improvements: Provide insights and practical solutions to improve the public transportation system using machine learning techniques 
The dataset includes over 226,000 bus stops, capturing various features like passenger boarding and trip durations. 
The tasks involve predicting the number of passengers boarding at specific bus stops and estimating the total trip duration for buses. Using these predictions, the project aims to provide practical insights and suggestions to optimize the transportation system, such as adjusting bus frequencies or proposing new routes. 

## Code Files:
### File 1 - main_subtask1: Predicting Passenger Boardings at Bus Stops

The goal of this task is to predict the number of passengers boarding a bus at a given stop. 

**Input:**  
A CSV file where each row contains information about a specific bus stop along a route, excluding the column `passengers_up` which represents the number of passengers boarding.

**Output:**  
A CSV file named `passengers_up_predictions.csv`, containing two columns:  
1. `trip_id_unique_station`  
2. `passengers_up`

This output will provide the predicted number of passengers boarding for each bus stop.

### File 2 - main_subtask2: Predicting Trip Duration

In this task, the goal is to predict the total duration of a bus trip, from its first station to the last. Each bus trip, identified by a unique `trip_unique_id`, is treated as a single sample. Based on the information from all bus stops within the trip, the objective is to predict the arrival time at the final stop.

**Input:**  
A CSV file where each row represents a single bus stop within a specific trip. The test set excludes the arrival times at the stops, except for the first station, which provides the departure time.

**Output:**  
A CSV file named `trip_duration_predictions.csv`, containing two columns: 
- `trip_id_unique` 
- `trip_duration_in_minutes` 

This output will predict the total trip duration in minutes.

## Excuting
To run the project, you can follow these general steps:

1. **Install Dependencies:**
   Ensure that all necessary libraries and dependencies are installed by running:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Script:**
   Execute the main script:
   ```bash
   python main.py
   ```

3. **Specify Input Files:**
   Ensure you have the appropriate dataset files (e.g., `train_bus_schedule.csv`, `X_passengers_up.csv`, etc.), which should be placed in the correct directories. Follow the dataset section in the project documentation.

4. **Output:**
   The output will be saved in CSV format, such as `passengers_up_predictions.csv` or `trip_duration_predictions.csv`.

