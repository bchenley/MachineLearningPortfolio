import pandas as pd
import generate_dataset

load_path = input('Enter load path: ')
save_path = input('Enter save path: ')
interval = input('Interval (day/hour): ')

df_master = pd.read_csv(load_path)

ts = generate_dataset(df_master, interval = interval)

ts.to_csv(save_path, index = True)
