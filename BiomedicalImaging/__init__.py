print("Initializing BiomedicalImaging package...")

import importlib

__all__ = ['create_train_test',
           'access_images',
           'get_images',
           'register_image',
           'register_subject_images']

for module_name in __all__:
    module = importlib.import_module(f'.{module_name}', __name__)
    globals()[module_name] = getattr(module, module_name)


           
print("Done")
