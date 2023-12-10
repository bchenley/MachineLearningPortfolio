print("Initializing AirBnB package...")

# Import the entire modules if needed
from . import DatasetCreator
from . import metrics
from . import moving_average
from . import fft
from . import periodogram

# import importlib

# __all__ = ['generate_timeseries', # 'create_time_series',                      
#            'fft',
#            'moving_average',           
#            'periodogram',
#            'metrics']

# for module_name in __all__:
#     module = importlib.import_module(f'.{module_name}', __name__)
#     globals()[module_name] = getattr(module, module_name)


           
print("Done")
