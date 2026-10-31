import pandas as pd
import numpy as np

p_hist = r"C:\Users\Janhvi Suryavanshi\Downloads\maharashtra_epiclim_filtered.csv"
p_cont = r"C:\Users\Janhvi Suryavanshi\.gemini\antigravity\scratch\JEEVAN-DRISHTI\maharashtra_outbreaks_2023_2025.csv"

df_h = pd.read_csv(p_hist)
df_c = pd.read_csv(p_cont)

official_36 = [
    'Ahmednagar', 'Akola', 'Amravati', 'Beed', 'Bhandara', 'Buldhana',
    'Chandrapur', 'Chhatrapati Sambhajinagar', 'Dharashiv', 'Dhule',
    'Gadchiroli', 'Gondia', 'Hingoli', 'Jalgaon', 'Jalna', 'Kolhapur',
    'Latur', 'Mumbai City', 'Mumbai Suburban', 'Nagpur', 'Nanded',
    'Nandurbar', 'Nashik', 'Palghar', 'Parbhani', 'Pune', 'Raigad',
    'Ratnagiri', 'Sangli', 'Satara', 'Sindhudurg', 'Solapur', 'Thane',
    'Wardha', 'Washim', 'Yavatmal'
]

# Standardize historical
h_map = {
    'Ahmadnagar': 'Ahmednagar',
    'Aurangabad': 'Chhatrapati Sambhajinagar',
    'Bid': 'Beed',
    'Buldana': 'Buldhana',
    'Gondiya': 'Gondia',
    'Mumbai': 'Mumbai City',
    'Nasik': 'Nashik',
    'Osmanabad': 'Dharashiv'
}
df_h['std_district'] = df_h['district'].replace(h_map)

# Standardize contemporary
c_map = {
    'Aurangabad': 'Chhatrapati Sambhajinagar',
    'Osmanabad': 'Dharashiv',
    'Palaghar': 'Palghar'
}
df_c['std_district'] = df_c['district'].replace(c_map)

# 1. District Availability Table
district_rows = []
for dist in sorted(official_36):
    sub_h = df_h[df_h['std_district'] == dist]
    sub_c = df_c[df_c['std_district'] == dist]
    
    h_count = len(sub_h)
    h_cases = int(sub_h['Cases'].sum()) if h_count > 0 else 0
    h_deaths = int(sub_h['Deaths'].sum()) if h_count > 0 else 0
    h_span = f"{sub_h['year'].min()}-{sub_h['year'].max()}" if h_count > 0 else "None"
    
    c_count = len(sub_c)
    c_cases = int(sub_c['reported_cases'].sum()) if c_count > 0 else 0
    c_deaths = int(sub_c['reported_deaths'].sum()) if c_count > 0 else 0
    c_span = f"{sub_c['report_year'].min()}-{sub_c['report_year'].max()}" if c_count > 0 else "None"
    
    tot_count = h_count + c_count
    tot_cases = h_cases + c_cases
    tot_deaths = h_deaths + c_deaths
    
    # Climate gridded coverage
    imd_coverage = "Yes (0.25 deg rain, 1.0 deg temp)"
    cwprs_rh = "Pune Station Ground-truth" if dist == 'Pune' else "Not Station-Representative"
    
    status = 'Sufficient for ML' if c_count >= 5 else ('Low Contemporary Alerts' if c_count > 0 else 'Historical Only')
    
    district_rows.append({
        'district_name': dist,
        'historical_outbreak_records_2009_2022': h_count,
        'historical_cases': h_cases,
        'historical_deaths': h_deaths,
        'historical_span': h_span,
        'contemporary_outbreak_records_2023_2025': c_count,
        'contemporary_cases': c_cases,
        'contemporary_deaths': c_deaths,
        'contemporary_span': c_span,
        'total_combined_records': tot_count,
        'total_combined_cases': tot_cases,
        'total_combined_deaths': tot_deaths,
        'imd_gridded_climate_2023_2025': imd_coverage,
        'cwprs_humidity_representativeness': cwprs_rh,
        'feasibility_status_with_2023_2025_data': status
    })

df_dist = pd.DataFrame(district_rows)
out_dist = r"C:\Users\Janhvi Suryavanshi\.gemini\antigravity\scratch\JEEVAN-DRISHTI\district_data_availability.csv"
df_dist.to_csv(out_dist, index=False)
print("Saved updated district_data_availability.csv")

# 2. Disease Availability Table
# Group diseases into standard epidemiological categories
def categorize_disease(name):
    n = str(name).lower()
    if any(k in n for k in ['dengue', 'malaria', 'chikungunya', 'zika', 'japanese encephalitis', 'je']):
        return 'Vector-Borne Diseases'
    elif any(k in n for k in ['diarrh', 'cholera', 'gastroenter', 'dysentery', 'food poisoning', 'typhoid', 'hepatitis a', 'hepatitis e', 'jaundice']):
        return 'Water-Borne & Food-Borne Diseases'
    elif any(k in n for k in ['leptospirosis', 'scrub typhus', 'rabies', 'anthrax']):
        return 'Zoonotic Diseases'
    elif any(k in n for k in ['measles', 'mumps', 'rubella', 'chickenpox', 'pertussis', 'influenza', 'fever', 'encephalit']):
        return 'Airborne & Other Viral/Infectious'
    return 'Other / Syndromic'

# Collect canonical diseases in contemporary
print("\nContemporary Disease counts:")
print(df_c['disease'].value_counts())

# Generate comprehensive disease table
# Merge disease names across both datasets
disease_rows = []

# Map raw disease to standard disease names
disease_std_map = {
    'Acute Diarrheal Disease': 'Acute Diarrheal Disease',
    'Acute Diarrhoeal Disease': 'Acute Diarrheal Disease',
    'Acute Diarrheal Diseases': 'Acute Diarrheal Disease',
    'Acute Gastroenteritis': 'Acute Diarrheal Disease',
    'Acute Gastroenterit is': 'Acute Diarrheal Disease',
    'Acute Gastroenter itis': 'Acute Diarrheal Disease',
    'Dysentery': 'Dysentery',
    'Food Poisoning': 'Food Poisoning',
    'Suspected Food Poisoning': 'Food Poisoning',
    'Cholera': 'Cholera',
    'Typhoid': 'Typhoid',
    'Hepatitis A': 'Viral Hepatitis A',
    'Hepatitis-A': 'Viral Hepatitis A',
    'Acute Hepatitis A': 'Viral Hepatitis A',
    'Hepatitis A&E': 'Viral Hepatitis A & E',
    'Suspected Acute Hepatitis': 'Viral Hepatitis',
    'Jaundice': 'Viral Hepatitis',
    'Dengue': 'Dengue',
    'Suspected Dengue Fever': 'Dengue',
    'Chikungunya': 'Chikungunya',
    'Dengue & Chikungunya': 'Dengue & Chikungunya Co-infection',
    'Dengue and Chikungunya': 'Dengue & Chikungunya Co-infection',
    'Chikungunya & Dengue': 'Dengue & Chikungunya Co-infection',
    'Malaria': 'Malaria',
    'Malaria (P.vivax)': 'Malaria (P.vivax)',
    'Malaria (P.falciparum)': 'Malaria (P.falciparum)',
    'Malaria (P.vivax & P.falciparum)': 'Malaria (Mixed)',
    'Mixed Malaria (Plasmodiu m falciparum & Plasmodiu m vivax)': 'Malaria (Mixed)',
    'Complicate d Malaria (Plasmodiu m falciparum)': 'Malaria (Severe P.falciparum)',
    'Dengue & Malaria (P.vivax)': 'Dengue & Malaria Co-infection',
    'Zika Virus': 'Zika Virus',
    'Leptospirosis': 'Leptospirosis',
    'Scrub Typhus': 'Scrub Typhus',
    'Human Rabies': 'Rabies',
    'Measles': 'Measles',
    'Measles & Rubella': 'Measles & Rubella',
    'Mumps': 'Mumps',
    'Chickenpox': 'Chickenpox',
    'Pertussis': 'Pertussis',
    'Japanese Encephalitis': 'Japanese Encephalitis',
    'Acute Encephalitis Syndrome': 'Acute Encephalitis Syndrome (AES)',
    'Acute Encephalitic Syndrome': 'Acute Encephalitis Syndrome (AES)',
    'Fever': 'Fever of Unknown Origin',
    'Fever of Unknown Cause': 'Fever of Unknown Origin',
    'Fever with Rash': 'Fever with Rash'
}

df_c['std_disease'] = df_c['disease'].map(lambda x: disease_std_map.get(x, x))

all_diseases = sorted(list(set(df_h['Disease'].unique()).union(set(df_c['std_disease'].unique()))))

for dis in all_diseases:
    sub_h = df_h[df_h['Disease'] == dis]
    sub_c = df_c[df_c['std_disease'] == dis]
    
    h_rec = len(sub_h)
    h_cases = int(sub_h['Cases'].sum()) if h_rec > 0 else 0
    h_deaths = int(sub_h['Deaths'].sum()) if h_rec > 0 else 0
    h_dist = sub_h['std_district'].nunique()
    
    c_rec = len(sub_c)
    c_cases = int(sub_c['reported_cases'].sum()) if c_rec > 0 else 0
    c_deaths = int(sub_c['reported_deaths'].sum()) if c_rec > 0 else 0
    c_dist = sub_c['std_district'].nunique()
    
    cat = categorize_disease(dis)
    
    tot_rec = h_rec + c_rec
    tot_cases = h_cases + c_cases
    tot_deaths = h_deaths + c_deaths
    
    # Climate driver
    if 'Vector' in cat:
        climate_driver = 'Temperature, Rainfall, Humidity (Mosquito breeding & EIP)'
    elif 'Water' in cat:
        climate_driver = 'Heavy Rainfall, Flooding, Temperature (Water contamination)'
    elif 'Zoonotic' in cat:
        climate_driver = 'Monsoon Flooding, Waterlogging (Rodent/vector exposure)'
    else:
        climate_driver = 'Seasonal Temperature, Humidity variation'
        
    suitability = 'High ML Priority' if (tot_rec >= 100 or c_rec >= 20) else ('Moderate ML Priority' if tot_rec >= 20 else 'Low Alert Volume')
    
    disease_rows.append({
        'disease_name': dis,
        'epidemiological_category': cat,
        'historical_outbreak_records': h_rec,
        'historical_cases': h_cases,
        'historical_deaths': h_deaths,
        'historical_districts_affected': h_dist,
        'contemporary_outbreak_records_2023_2025': c_rec,
        'contemporary_cases': c_cases,
        'contemporary_deaths': c_deaths,
        'contemporary_districts_affected': c_dist,
        'total_combined_records': tot_rec,
        'total_combined_cases': tot_cases,
        'total_combined_deaths': tot_deaths,
        'primary_climate_drivers': climate_driver,
        'ml_modeling_suitability': suitability
    })

df_dis_out = pd.DataFrame(disease_rows).sort_values(by=['total_combined_records', 'contemporary_outbreak_records_2023_2025'], ascending=False)
out_dis = r"C:\Users\Janhvi Suryavanshi\.gemini\antigravity\scratch\JEEVAN-DRISHTI\disease_data_availability.csv"
df_dis_out.to_csv(out_dis, index=False)
print("Saved updated disease_data_availability.csv")
print(df_dis_out.head(15))
