from mock import MagicMock


def mock_import_modules(modules):
    '''Given a list of modules,it will create a dictionary of mocked modules,
    including all submodules.
    Examples:

        _mock_import_modules(['tensorflow']) => mock, {'tensorflow': mock}
        _mock_import_modules(['a', 'b.c.d') =>
            mock, {'a': mock, 'b': mock, 'c': mock, 'd': mock}
    Args:
        modules: list (str) list of modules to be imported

    Returns:
        mock: containing all imports
        imports: dictionary containing all imports
    '''
    imports = {}
    _mock = MagicMock()
    for _module in modules:
        namespaces = _module.split('.')
        full_module_name = namespaces.pop(0)
        imports[full_module_name] = _mock

        for namespace in namespaces:
            full_module_name = '{}.{}'.format(full_module_name, namespace)

            imports[full_module_name] = _mock
    return _mock, imports
