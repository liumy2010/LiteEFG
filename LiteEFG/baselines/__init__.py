def _import_all_modules():
    import os
    import importlib
    package_dir = os.path.dirname(__file__)

    for filename in os.listdir(package_dir):
        if filename.endswith('.py') and filename != '__init__.py' and filename != 'utils.py':
            module_name = f'.{filename[:-3]}'
            importlib.import_module(module_name, package=__name__)

_import_all_modules()