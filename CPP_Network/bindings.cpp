#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Layer.hpp"
#include "Types.hpp"

namespace py = pybind11;

PYBIND11_MODULE(NeuralNetwork_CPP, m) {
    m.doc() = "Neural Network module";

    py::enum_<ActivationType>(m, "ActivationType")
        .value("RELU", ActivationType::RELU)
        .value("LEAKY_RELU", ActivationType::LEAKY_RELU)
        .value("PASS", ActivationType::PASS)
        .export_values();

    py::class_<Layer>(m, "Layer")
        .def(py::init<int, int, ActivationType>())
        .def("forward", &Layer::forward)
        .def("backward", &Layer::backward)
        .def("get_weights", &Layer::getWeights)
        .def("get_biases", &Layer::getBiases)
        .def("get_input_gradient", &Layer::getInputGradient);
}
