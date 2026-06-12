from setuptools import setup, Extension
import pybind11

extra_compile_args = ["/std:c++17", "/O2"]

# If you were strictly on Windows/MSVC, you would use ["/std:c++17", "/O2"]
# This setup handles the C++ source files
ext = Extension(
    "NeuralNetwork_CPP", # This must match the name in PYBIND11_MODULE
    sources=["bindings.cpp", "Layer.cpp", "Mathlib.cpp"],
    include_dirs=[pybind11.get_include()],
    language="c++",
    extra_compile_args=extra_compile_args,
)

setup(
    name="NeuralNetwork_CPP",
    version="0.4",
    ext_modules=[ext],
    # This ensures pybind11 is installed if missing
    install_requires=["pybind11>=2.6"],
)