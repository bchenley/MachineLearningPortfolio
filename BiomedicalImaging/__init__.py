print("Initializing BiomedicalImaging package...")

import importlib

__all__ = ['access_images',
           'get_images',
           'get_train_test', 
           'register_image',
           'register_subject_images']

for module_name in __all__:
    module = importlib.import_module(f'.{module_name}', __name__)
    globals()[module_name] = getattr(module, module_name)


           
print("Done")
