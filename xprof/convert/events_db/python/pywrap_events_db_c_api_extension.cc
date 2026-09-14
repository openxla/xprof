/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <Python.h>

static struct PyModuleDef pywrap_events_db_c_api_module = {
    PyModuleDef_HEAD_INIT, "pywrap_events_db_c_api_extension", nullptr, -1,
    nullptr};

PyMODINIT_FUNC PyInit_pywrap_events_db_c_api_extension(void) {
  PyObject* m = PyModule_Create(&pywrap_events_db_c_api_module);
  if (!m) return nullptr;
#ifdef Py_GIL_DISABLED
  if (PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED) != 0) {
    Py_DECREF(m);
    return nullptr;
  }
#endif
  return m;
}
