# DPWH Flood Control Project On-Time Prediction System

## Features

### 1. Single Project Prediction
Enter details for a single project to get an individual on-time/delayed prediction with confidence scores.

The app automatically engineers derived features from your raw inputs:
- **ProjectDuration** = calculated from Start Date and Completion Date
- **AverageDuration_Region** = calculated from historical data for the selected region
- **AverageDuration_TypeOfWork** = calculated from historical data for the selected work type

**Input Sections:**
- **Budget & Cost**: Approved Budget, Contract Cost
- **Project Timeline**: Start Date, Completion Date
- **Project Team**: Number of Contractors, Funding Year
- **Project Coordinates**: Latitude, Longitude
- **Provincial Capital**: Capital Latitude, Longitude
- **Geographic**: Main Island, Region, Province
- **Administrative**: Legislative District, Municipality, District Engineering Office
- **Project Type**: Type of Work, Provincial Capital

**Output:**
- Prediction result (On Time / Delayed)
- Confidence percentage
- Visual indicator cards

### 2. Batch Upload (NEW)
Upload a CSV file containing multiple projects for bulk predictions.

**CSV Format:**
Your CSV file must include all of these columns (raw features only - derived features are calculated automatically):

```
ProjectName, ApprovedBudgetForContract, ContractCost, StartDate, ActualCompletionDate,
ContractorCount, FundingYear, ProjectLatitude, ProjectLongitude, ProvincialCapitalLatitude,
ProvincialCapitalLongitude, MainIsland, Region, Province, LegislativeDistrict,
Municipality, DistrictEngineeringOffice, TypeOfWork, ProvincialCapital
```

**Column Descriptions:**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| ProjectName | String | Name of the project | "Project Alpha" |
| ApprovedBudgetForContract | Float | Approved budget in PHP | 50000000 |
| ContractCost | Float | Actual contract cost in PHP | 48000000 |
| StartDate | Date | Project start date (YYYY-MM-DD format) | 2022-01-15 |
| ActualCompletionDate | Date | Project completion date (YYYY-MM-DD format) | 2022-06-15 |
| ContractorCount | Float | Number of contractors | 1, 2, or 3 |
| FundingYear | Float | Year project was funded | 2018-2025 |
| ProjectLatitude | Float | Geographic latitude | 18.25 |
| ProjectLongitude | Float | Geographic longitude | 121.05 |
| ProvincialCapitalLatitude | Float | Capital latitude | 18.02 |
| ProvincialCapitalLongitude | Float | Capital longitude | 121.17 |
| MainIsland | String | Main island name | "Luzon", "Visayas", "Mindanao" |
| Region | String | Region name | "Cordillera Administrative Region" |
| Province | String | Province name | "Apayao", "Mindoro Occidental" |
| LegislativeDistrict | String | Legislative district | "1st District", "APAYAO (LEGISLATIVE DISTRICT)" |
| Municipality | String | Municipality/City name | "Calanasan", "San Jose" |
| DistrictEngineeringOffice | String | Engineering office | "Apayao 2nd District Engineering Office" |
| TypeOfWork | String | Type of flood control work | "Construction of Flood Mitigation Structure" |
| ProvincialCapital | String | Nearest provincial capital | "Kabugao" |

**Automatic Feature Engineering:**
The system automatically calculates:
- **ProjectDuration** from StartDate and ActualCompletionDate
- **AverageDuration_Region** from historical data for the specified region
- **AverageDuration_TypeOfWork** from historical data for the specified work type

**Batch Results:**
The system generates a table showing:
- **Project Name**: Name of the project
- **Top Important Features**: The 2 most important features for the prediction
- **Status**: On Time / Delayed
- **Confidence**: Prediction confidence percentage

Results can be downloaded as a CSV file for further analysis.

**Sample File:**
A sample CSV file `sample_projects.csv` is provided in the workspace with 5 example projects.

## Running the App

```bash
cd "c:\Users\Ian\Documents\Classes\Elective 1\PIT\PIT STREAMLIT"
python -m streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Model Details

- **Algorithm**: Random Forest Classifier
- **Training Data**: DPWH flood control projects
- **Target**: On-Time (≤ median duration) vs Delayed (> median duration)
- **Features**: 11 numerical + 8 categorical = 19 total features

## Dark Mode Theme

The app uses a professional dark-mode theme inspired by GitHub and VS Code:
- Dark background for eye comfort
- Purple accent color (#a855f7)
- Custom Inter and JetBrainsMono fonts
- Color-coded results (green for on-time, red for delayed)

## File Structure

```
PIT STREAMLIT/
├── app.py                    # Main Streamlit application
├── .streamlit/
│   └── config.toml          # Theme and configuration
├── static/
│   ├── Inter_18pt-*.ttf     # Font files
│   └── JetBrainsMono-*.ttf  # Font files
├── random_forest_model.joblib    # Trained model
├── imputer_numeric.joblib        # Numerical feature imputer
├── encoder.joblib                # Categorical feature encoder
├── sample_projects.csv           # Sample batch upload file
└── requirements.txt              # Python dependencies
```

## Tips for Best Results

1. **Budget Data**: Ensure budget values are realistic for Philippine projects
2. **Duration Data**: Use historical average durations for similar work types and regions
3. **Geographic Data**: Verify latitude/longitude coordinates are within Philippines bounds
4. **Categorical Data**: Use exact values from the dropdown menus to ensure accurate encoding
5. **Batch Processing**: For large CSV files, the app will process all rows and generate results

## Troubleshooting

**File Upload Error**: Ensure your CSV has all required columns with exact names (case-sensitive)

**Missing Values**: The imputer will handle missing numerical values, but categorical fields cannot be empty

**Memory Issues**: For very large batch files (1000+ rows), processing may take longer

For support or issues, contact the project team.
