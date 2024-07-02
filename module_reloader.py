
def reload_module(module_name):
    import importlib
    import sys

    if module_name in sys.modules:
        imported_module = sys.modules[module_name]
        importlib.reload(imported_module)
    else:
        imported_module = importlib.import_module(module_name)

    globals().update(vars(imported_module))

