import pandas as pd
import numpy as np

df = pd.read_csv('data/dpwh_flood_control_projects.csv')
np.random.seed(123)
samples = df.sample(n=2)

for idx, (i, row) in enumerate(samples.iterrows(), 1):
    print(f'\n{"="*70}')
    print(f'PROJECT {idx}: {row["ProjectName"]}')
    print(f'{"="*70}')
    print(f'ApprovedBudgetForContract: {row["ApprovedBudgetForContract"]}')
    print(f'ContractCost: {row["ContractCost"]}')
    print(f'StartDate: {row["StartDate"]}')
    print(f'ActualCompletionDate: {row["ActualCompletionDate"]}')
    print(f'ContractorCount: {int(row["ContractorCount"])}')
    print(f'FundingYear: {int(row["FundingYear"])}')
    print(f'ProjectLatitude: {row["ProjectLatitude"]}')
    print(f'ProjectLongitude: {row["ProjectLongitude"]}')
    print(f'ProvincialCapitalLatitude: {row["ProvincialCapitalLatitude"]}')
    print(f'ProvincialCapitalLongitude: {row["ProvincialCapitalLongitude"]}')
    print(f'MainIsland: {row["MainIsland"]}')
    print(f'Region: {row["Region"]}')
    print(f'Province: {row["Province"]}')
    print(f'LegislativeDistrict: {row["LegislativeDistrict"]}')
    print(f'Municipality: {row["Municipality"]}')
    print(f'DistrictEngineeringOffice: {row["DistrictEngineeringOffice"]}')
    print(f'TypeOfWork: {row["TypeOfWork"]}')
    print(f'ProvincialCapital: {row["ProvincialCapital"]}')
