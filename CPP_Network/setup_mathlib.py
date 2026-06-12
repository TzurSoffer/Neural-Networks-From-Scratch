from setuptools import setup, Extension
import pybind11

extra_compile_args = ["/std:c++17", "/O2"]

ext = Extension(
    "mathlib",
    ["bindings_mathlib.cpp", "Mathlib.cpp"],
    include_dirs=[pybind11.get_include()],
    language="c++",
    extra_compile_args=extra_compile_args,
)

setup(
    name="Mathlib",
    version="0.4",
    ext_modules=[ext],
)
