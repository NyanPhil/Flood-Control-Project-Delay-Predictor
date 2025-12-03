# Feature Engineering Architecture

## Overview
The app has been updated to separate **raw user inputs** from **engineered features**. Users only provide base project information, and the app automatically calculates derived features for prediction.

## Raw Features (User Input Only)
These are the base features that users enter directly:

### Numerical Raw Features
1. **ApprovedBudgetForContract** - Total approved budget (PHP)
2. **ContractCost** - Actual contract cost (PHP)
3. **ContractorCount** - Number of contractors (1-3)
4. **FundingYear** - Year the project was funded (2018-2025)
5. **ProjectLatitude** - Geographic latitude of project
6. **ProjectLongitude** - Geographic longitude of project
7. **ProvincialCapitalLatitude** - Latitude of provincial capital
8. **ProvincialCapitalLongitude** - Longitude of provincial capital
9. **StartDate** - Project start date (dates only)
10. **ActualCompletionDate** - Project completion date (dates only)

### Categorical Raw Features
1. **MainIsland** - Main island of project
2. **Region** - Administrative region
3. **Province** - Province name
4. **LegislativeDistrict** - Legislative district
5. **Municipality** - Municipality/City name
6. **DistrictEngineeringOffice** - Responsible engineering office
7. **TypeOfWork** - Type of flood control work
8. **ProvincialCapital** - Nearest provincial capital

## Engineered Features (Auto-Calculated)
These are derived from raw inputs during prediction:

### Engineered Numerical Features
1. **ProjectDuration** 
   - Calculation: `ActualCompletionDate - StartDate` (in days)
   - Purpose: Duration of the project
   
2. **AverageDuration_Region**
   - Calculation: Mean project duration for the selected region
   - Source: Historical data from dpwh_flood_control_projects.csv
   - Purpose: Regional baseline for duration comparison
   
3. **AverageDuration_TypeOfWork**
   - Calculation: Mean project duration for the selected work type
   - Source: Historical data from dpwh_flood_control_projects.csv
   - Purpose: Work-type baseline for duration comparison

## Engineering Process

### During Single Project Prediction
1. User enters raw features through the UI
2. `engineer_features()` function is called with:
   - Raw user inputs
   - Pre-computed averages by region
   - Pre-computed averages by work type
3. Function returns complete feature dictionary with engineered features
4. Features are preprocessed (imputation, encoding) and sent to model

### During Batch Prediction (CSV Upload)
1. CSV file is loaded with raw features
2. Date columns are converted to datetime format
3. For each project:
   - Raw data extracted from CSV row
   - `engineer_features()` function calculates derived features
   - Features are preprocessed and predicted
4. Results table is generated and can be downloaded

## Implementation Details

### Feature Engineering Function
```python
def engineer_features(user_data: dict, avg_by_region: dict, 
                     avg_by_work_type: dict, global_avg: float = 250.0) -> dict
```

**Parameters:**
- `user_data`: Dictionary with raw user inputs
- `avg_by_region`: Pre-computed average durations by region
- `avg_by_work_type`: Pre-computed average durations by work type
- `global_avg`: Fallback average (default: 250 days)

**Returns:**
- Complete feature dictionary with all engineered features

### Pre-computation of Averages
The `load_and_process_training_data()` function:
1. Loads historical data from `dpwh_flood_control_projects.csv`
2. Calculates ProjectDuration for all historical projects
3. Computes means grouped by Region
4. Computes means grouped by TypeOfWork
5. Caches results for reuse

## Benefits

✅ **Simpler User Interface**: Users only input 10 raw numerical + 8 categorical features instead of 13 numerical features

✅ **Data Quality**: Engineered features are calculated consistently from raw data

✅ **Historical Context**: Duration averages are based on actual project data for the specific region/work type

✅ **Batch Processing**: CSV files only need raw features; engineering happens automatically

✅ **Reproducibility**: Same engineering logic applied to single and batch predictions

## CSV Format for Batch Upload

```
ProjectName,ApprovedBudgetForContract,ContractCost,StartDate,ActualCompletionDate,
ContractorCount,FundingYear,ProjectLatitude,ProjectLongitude,ProvincialCapitalLatitude,
ProvincialCapitalLongitude,MainIsland,Region,Province,LegislativeDistrict,
Municipality,DistrictEngineeringOffice,TypeOfWork,ProvincialCapital
```

Note: Dates must be in YYYY-MM-DD format.

## Example Calculation

For a project:
- Region: "Cordillera Administrative Region"
- TypeOfWork: "Construction of Flood Mitigation Structure"
- StartDate: 2022-01-15
- ActualCompletionDate: 2022-06-15

Engineering produces:
- ProjectDuration: 152 days (calculated)
- AverageDuration_Region: 187 days (from historical data)
- AverageDuration_TypeOfWork: 201 days (from historical data)

These engineered features are then used alongside the raw features for prediction.
