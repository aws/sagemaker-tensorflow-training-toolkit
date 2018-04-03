#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#  
#      http://www.apache.org/licenses/LICENSE-2.0
#  
#  or in the "license" file accompanying this file. This file is distributed 
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either 
#  express or implied. See the License for the specific language governing 
#  permissions and limitations under the License.

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
