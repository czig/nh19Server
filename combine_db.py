import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('sqlite:///./surveys.db')
engine2 = create_engine('sqlite:///./surveys2.db')
out_engine = create_engine('sqlite:///./combined_surveys.db')

grade_dict = {
'E1': 'E1-E4', 
'E2': 'E1-E4',
'E3': 'E1-E4',
'E4': 'E1-E4',
'E5': 'E5-E6',
'E6': 'E5-E6',
'E7': 'E7-E9',
'E8': 'E7-E9',
'E9': 'E7-E9',
'O1': 'O1-O3',
'O2': 'O1-O3',
'O3': 'O1-O3',
'O4': 'O4-O6',
'O5': 'O4-O6',
'O6': 'O4-O6',
}

entry_df = pd.read_sql("""select * from entry_surveys""", engine)
camp_df = pd.read_sql("""select * from camp_surveys""",engine)
exit_df = pd.read_sql("""select * from exit_surveys""",engine)
print('db1 entry survey shape: ',entry_df.shape)
print('db1 camp survey shape: ',camp_df.shape)
print('db1 exit survey shape: ',exit_df.shape)

entry_df2 = pd.read_sql("""select * from entry_surveys""", engine2)
camp_df2 = pd.read_sql("""select * from camp_surveys""",engine2)
exit_df2 = pd.read_sql("""select * from exit_surveys""",engine2)
print('db2 entry survey shape: ', entry_df2.shape)
print('db2 camp survey shape: ', camp_df2.shape)
print('db2 exit survey shape: ', exit_df2.shape)

#combine databases into one
all_entry = pd.concat([entry_df, entry_df2])
all_camp = pd.concat([camp_df, camp_df2])
all_exit = pd.concat([exit_df, exit_df2])

#use grade_dict above to standardize grades
all_entry['grade'].replace(grade_dict, inplace=True)
all_camp['grade'].replace(grade_dict, inplace=True)
all_exit['grade'].replace(grade_dict, inplace=True)

print('combined entry survey shape: ', all_entry.shape)
print('combined camp survey shape: ', all_camp.shape)
print('combined exit survey shape: ', all_exit.shape)
all_entry.sort_values(by='submitDate',inplace=True)
all_camp.sort_values(by='submitDate',inplace=True)
all_exit.sort_values(by='submitDate',inplace=True)

all_entry.to_sql('entry_surveys',con=out_engine,if_exists="replace")
all_camp.to_sql('camp_surveys',con=out_engine,if_exists="replace")
all_exit.to_sql('exit_surveys',con=out_engine,if_exists="replace")
