import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('sqlite:///./surveys.db')
engine2 = create_engine('sqlite:///./surveys2.db')
out_engine = create_engine('sqlite:///./combined_surveys.db')

entry_df = pd.read_sql("""select * from entry_surveys""", engine)
camp_df = pd.read_sql("""select * from camp_surveys""",engine)
exit_df = pd.read_sql("""select * from exit_surveys""",engine)
print('first entry surveys')
print(entry_df)

entry_df2 = pd.read_sql("""select * from entry_surveys""", engine2)
camp_df2 = pd.read_sql("""select * from camp_surveys""",engine2)
exit_df2 = pd.read_sql("""select * from exit_surveys""",engine2)

print('second entry surveys')
print(entry_df2)

all_entry = pd.concat([entry_df, entry_df2])
all_camp = pd.concat([camp_df, camp_df2])
all_exit = pd.concat([exit_df, exit_df2])

print('combined entry surveys')
print(all_entry)

all_entry.to_sql('entry_surveys',con=out_engine,if_exists="replace")
all_camp.to_sql('camp_surveys',con=out_engine,if_exists="replace")
all_exit.to_sql('exit_surveys',con=out_engine,if_exists="replace")
