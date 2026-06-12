#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Mathlib.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mathlib, m) {
    m.doc() = "Python bindings for mathlib";

    m.def("round", &round);
    m.def("randomNumber", &randomNumber);
    m.def("clip", &clip);
    m.def("clipAboveZeroBelowOne", &clipAboveZeroBelowOne);
    m.def("dot2Vectors", &dot2Vectors);
    m.def("dotVectorMatrix", &dotVectorMatrix);
    m.def("addTwoVectors", &addTwoVectors);
    m.def("addTwoMatrices", &addTwoMatrices);
    m.def("elementWiseMult", &elementWiseMult);
    m.def("scale", &scale);
    m.def("normalize", &normalize);

    m.def("hilbertIndexToXy", &hilbertIndexToXy);
    m.def("getOrder", &getOrder);
    m.def("hilbertFlatten", &hilbertFlatten);
    m.def("hilbertUnflatten", &hilbertUnflatten);

    m.def("sum", py::overload_cast<const std::vector<int>&>(&sum));
    m.def("sum", py::overload_cast<const std::vector<double>&>(&sum));

    m.def("mean", py::overload_cast<const std::vector<int>&>(&mean));
    m.def("mean", py::overload_cast<const std::vector<double>&>(&mean));

    m.def("argmax", py::overload_cast<const std::vector<int>&>(&argmax));
    m.def("argmax", py::overload_cast<const std::vector<double>&>(&argmax));
    m.def("argmin", py::overload_cast<const std::vector<int>&>(&argmin));
    m.def("argmin", py::overload_cast<const std::vector<double>&>(&argmin));

    m.def("zeroes", py::overload_cast<int>(&zeroes));
    m.def("zeroes", py::overload_cast<int, int>(&zeroes));

    m.def("vectorScalerMult",
          py::overload_cast<const std::vector<double>&, double>(&vectorScalerMult));

    m.def("vectorScalerMult",
          py::overload_cast<const std::vector<std::vector<double>>&, double>(&vectorScalerMult));
}
