#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Layer.hpp"
#include "Batch.hpp"
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
        .def("backward", py::overload_cast<const std::vector<double>&>(&Layer::backward), py::arg("d_values"))
        .def("backward", py::overload_cast<const std::vector<double>&, const std::vector<double>&, const std::vector<double>&>(&Layer::backward), 
                py::arg("d_values"), py::arg("inputs"), py::arg("preActivationOut"))

        .def("getWeights", &Layer::getWeights)
        .def("setWeights", &Layer::setWeights)
        .def("addToWeight", &Layer::addToWeight)
        .def("getBiases", &Layer::getBiases)
        .def("setBiases", &Layer::setBiases)
        .def("addToBias", &Layer::addToBias)
        .def("getInputGradient", &Layer::getInputGradient)
        .def("getOutputs", &Layer::getOutputs)
        .def("getPreActivationOutputs", &Layer::getPreActivationOutputs)
        .def("getWeightsGradient", &Layer::getWeightsGradient)
        .def("getBiasesGradient", &Layer::getBiasesGradient);

    py::class_<Batch>(m, "Batch")
        .def(py::init<int, int, int, ActivationType>())

        .def("forward", &Batch::forward)
        .def("backward", &Batch::backward)

        .def("getWeights", &Batch::getWeights)
        .def("setWeights", &Batch::setWeights)
        .def("addToWeight", &Batch::addToWeight)
        .def("getBiases", &Batch::getBiases)
        .def("setBiases", &Batch::setBiases)
        .def("addToBias", &Batch::addToBias)
        .def("getInputGradient", &Batch::getInputGradient)
        .def("getOutputs", &Batch::getOutputs)
        .def("getPreActivationOutputs", &Batch::getPreActivationOutputs)
        .def("getWeightsGradient", &Batch::getWeightsGradient)
        .def("getBiasesGradient", &Batch::getBiasesGradient);
}
