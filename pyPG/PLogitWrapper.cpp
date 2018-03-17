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

/*
static PyObject *rpg_gamma(PyObject *self, PyObject *args)
{
  RNG r;
  PyArrayObject *x, *n, *z;
  int num, trunc;

  if (!PyArg_ParseTuple(args, "OOOii", &x, &n, &z, &num, &trunc))
    return NULL;

  double *dx = (double*)x->data;
  double *dn = (double*)n->data;
  double *dz = (double*)z->data;

  PolyaGamma pg(trunc);

  for(int i=0; i < num; ++i){
      if (dn[i]!=0.0)
	dx[i] = pg.draw_sum_of_gammas(dn[i], dz[i], r);
      else
  	dx[i] = 0.0;
  }

  return PyLong_FromLong(5);  // not correct behavior
} // rpg


// Draw PG(h, z) where h is \geq 1.
static PyObject *rpg_alt(PyObject *self, PyObject *args)
{
  RNG r;
  PolyaGammaAlt pg;
  PyArrayObject *h, *z, *x;
  int num;

  if (!PyArg_ParseTuple(args, "OOOi", &x, &h, &z, &num))
    return NULL;

  double *dx = (double*)x->data;
  int    *dh = (int*)h->data;
  double *dz = (double*)z->data;

  for(int i=0; i < num; ++i){
    if (dh[i]!=0)
      dx[i] = pg.draw(dh[i], dz[i], r);
    else
      dx[i] = 0.0;
  }

  return PyLong_FromLong(5);  // not correct behavior
}

//void rpg_sp(double *x, double *h, double *z, int* num, int *iter)
static PyObject *rpg_sp(PyObject *self, PyObject *args)
{
  RNG r;
  PolyaGammaSP pg;

  PyArrayObject *h, *z, *x, *iter;
  int num;

  if (!PyArg_ParseTuple(args, "OOOiO", &x, &h, &z, &num, &iter))
    return NULL;

  double *dx    = (double*)x->data;
  int    *dh    = (int*)h->data;
  double *dz    = (double*)z->data;
  int    *diter = (int*)iter->data;

  for(int i=0; i < num; ++i){
    if (dh[i]!=0)
      diter[i] = pg.draw(dx[i], dh[i], dz[i], r);
    else
      dx[i] = 0.0;
  }

  return PyLong_FromLong(5);  // not correct behavior
}


//void rpg_hybrid(double *x, double *h, double *z, int* num)
static PyObject *rpg_hybrid(PyObject *self, PyObject *args)
{
  RNG r;
  PolyaGamma dv;
  PolyaGammaAlt alt;
  PolyaGammaSP sp;

  PyArrayObject *h, *z, *x;
  int num;

  if (!PyArg_ParseTuple(args, "OOOi", &x, &h, &z, &num))
    return NULL;

  double *dx    = (double*)x->data;
  int    *dh    = (int*)h->data;
  double *dz    = (double*)z->data;

  for(int i=0; i < num; ++i){
    double b = dh[i];
    if (b > 170) {
      double m = dv.pg_m1(b,dz[i]);
      double v = dv.pg_m2(b,dz[i]) - m*m;
      dx[i] = r.norm(m, sqrt(v));
    }
    else if (b > 13) {
      sp.draw(dx[i], b, dz[i], r);
    }
    else if (b==1 || b==2) {
      dx[i] = dv.draw((int)b, dz[i], r);
    }
    else if (b > 0) {
      dx[i] = alt.draw(b, dz[i], r);
    }
    else {
      dx[i] = 0.0;
    }
  }

  return PyLong_FromLong(5);  // not correct behavior
}


double *x,
double *h,
double *z,
int* num,
int* nthreads,
vector<RNG> *r,
vector<PolyaGamma> *dv,
vector<PolyaGammaAlt> *alt,
vector<PolyaGammaSP> *sp)


    {"rpg_devroyeS",  rpg_devroyeS, METH_VARARGS,
     "Polya-Gamma, single random draw, integer"},
    {"rpg_gamma",  rpg_gamma, METH_VARARGS,
     "Polya-Gamma, multiple random draws, real"},
    {"rpg_alt",  rpg_alt, METH_VARARGS,
     "Polya-Gamma, multiple random draws, real > 1"},
*/

static PyMethodDef SpamMethods[] = {
    {"rpg_devroye",  rpg_devroye, METH_VARARGS,
     "Polya-Gamma, multiple random draws, integer"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC initpyPG_cpp_module(void)
{
  //  This gets called when "import spam" called.
  PyObject *m;
  int i;

  m = Py_InitModule("pyPG_cpp_module", SpamMethods);
  if (m == NULL)
    return;

  SpamError = PyErr_NewException("spam.error", NULL, NULL);
  Py_INCREF(SpamError);
  PyModule_AddObject(m, "error", SpamError);
}
