////////////////////////////////////////////////////////////////////////////////

// Copyright 2014 Kensuke Arai, Scott 
// Copyright 2012 Nick Polson, James Scott, and Jesse Windle.

// This file is part of BayesLogit.

// BayesLogit is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.

// BayesLogit is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

// You should have received a copy of the GNU General Public License along with
// BayesLogit.  If not, see <http://www.gnu.org/licenses/>.

////////////////////////////////////////////////////////////////////////////////

#include <Python.h>
#include "numpy/arrayobject.h"
#include "PLogitWrapper.h"
#include "RNG.hpp"
#include "PolyaGamma.h"
#include "PolyaGammaAlt.h"
#include "PolyaGammaSP.h"
#include <exception>
#include <cstdio>
//#include <gsl/gsl_randist.h>
#include <omp.h>
#include <stdio.h>

////////////////////////////////////////////////////////////////////////////////
				// PolyaGamma //
////////////////////////////////////////////////////////////////////////////////

static PyObject *SpamError;

//gsl_rng *rng;   // This doesn't have to be static.  Once loaded
vector<RNG*> vr;
vector<PolyaGamma*> vpg;   // trunc == 1, for devroye
vector<PolyaGammaAlt*> vpa;
vector<PolyaGammaSP*> vps;

//  PLogitWrapper.cpp   -- This file is Python specific

static PyObject *rpg_devroye(PyObject *self, PyObject *args)
{
  PyArrayObject *n, *z, *x;
  int num, i, xst, nst, zst;

  //if (!PyArg_ParseTuple(args, "OOOiiii", &x, &n, &z, &xst, &nst, &zst, &num))    return NULL;
  if (!PyArg_ParseTuple(args, "OOOi", &x, &n, &z, &num))    return NULL;
  double *dx    = (double*)x->data;
  int    *dn    = (int*)n->data;
  double *dz    = (double*)z->data;

  //rpg_devroye(dx, dn, dz, &xst, &nst, &zst, &num);
  rpg_devroye(dx, dn, dz, &num);

  Py_RETURN_NONE;
}

static PyObject *rpg_gamma(PyObject *self, PyObject *args)
{
  PyArrayObject *h, *z, *x;
  int num, mxth, tid, i, trunc;

  if (!PyArg_ParseTuple(args, "OOOii", &x, &h, &z, &num, &trunc))
    return NULL;
  double *dx    = (double*)x->data;
  int    *dh    = (int*)h->data;
  double *dz    = (double*)z->data;

  Py_RETURN_NONE;
}

static PyMethodDef SpamMethods[] = {
    {"rpg_devroye",  rpg_devroye, METH_VARARGS,
     "Polya-Gamma, multiple random draws, integer"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC PyInit_pyPG_cpp_module(void)
{
  //  This gets called when "import spam" called.
  PyObject *m;
  int i;

  static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "pyPG_cpp_module",     /* m_name */
    "This is a module",  /* m_doc */
    -1,                  /* m_size */
    SpamMethods,    /* m_methods */
    NULL,                /* m_reload */
    NULL,                /* m_traverse */
    NULL,                /* m_clear */
    NULL,                /* m_free */
  };
  //m = Py_InitModule("pyPG_cpp_module", SpamMethods);
  m = PyModule_Create(&moduledef);
  if (m == NULL) {
    //return;
    Py_RETURN_NONE;
  }

  SpamError = PyErr_NewException("spam.error", NULL, NULL);
  Py_INCREF(SpamError);
  PyModule_AddObject(m, "error", SpamError);

  //  added a return value, needed for 3.7
  //  http://python3porting.com/cextensions.html
  return m;
}
