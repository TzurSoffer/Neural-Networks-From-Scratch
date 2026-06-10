import os
from setuptools import setup, Extension
import pybind11

# Force MinGW
os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

extra_compile_args = ["/std:c++17", "/O2"]

ext = Extension(
    "mathlib",
    ["bindings.cpp", "mathlib.cpp"],
    include_dirs=[pybind11.get_include()],
    language="c++",
    extra_compile_args=extra_compile_args,
)

setup(
    name="Mathlib",
    version="0.4",
    ext_modules=[ext],
)
