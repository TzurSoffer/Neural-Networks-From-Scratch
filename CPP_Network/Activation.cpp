#include <cmath>
#include <vector>

#include "Activation.hpp"
#include "Mathlib.hpp"


double Pass::forward(double val) {
    return val;
}

double Pass::backward(double val) {
    /* derivative of x with respect to x */
    return 1.0;
}


double ReLU::forward(double val) {
    return std::max(0.0, val);
}

double ReLU::backward(double val) {
    if (val > 0.0) {
        return 1.0;
    }
    return 0.0;
}


double LeakyReLU::forward(double val) {
    return std::max(LeakyReLU::alpha * val, val);
}

double LeakyReLU::backward(double val) {
    /* partial derivative of LeakyReLU with respect to x */
    if (val > 0) {
        return 1.0;
    }
    return LeakyReLU::alpha;
}


std::vector<double> Softmax::forward(std::vector<double> vals) {
    for (double& val : vals) {
        val = std::exp(val);
    }
    return normalize(vals);
}

std::vector<std::vector<double>> Softmax::backward(const std::vector<double>& vals) {
    /* derivative of e^x/(sum(e^x) for x in vals) with respect to x
    For proof, see image above or in 'proofs_math/ActivationFuncs'
    */
    std::vector<std::vector<double>> jacobian = std::vector<std::vector<double>>(vals.size(), std::vector<double>(vals.size(), 0.0));
    for (int i=0; i<vals.size(); i++) {
        double val = vals[i];
        for (int j=0; j<vals.size(); j++) {
            if (j == i) {
                jacobian[i][j] = val*(1-val);
            }
            else {
                jacobian[i][j] = -val*vals[j];
            }
        }
    }
    return jacobian;
}


std::vector<std::vector<double>> Softmax_batch::forward(const std::vector<std::vector<double>>& vals) {
    /* same as regular forward, but for a batch */
    std::vector<std::vector<double>> result = std::vector<std::vector<double>>(vals.size(), std::vector<double>(vals[0].size(), 0.0));

    for (int i=0; i<vals.size(); i++) {
        result[i] = Softmax::forward(vals[i]);
    }
    return result;
}

std::vector<std::vector<std::vector<double>>> Softmax_batch::backward(const std::vector<std::vector<double>>& vals) {
    std::vector<std::vector<std::vector<double>>> result =
        std::vector<std::vector<std::vector<double>>>(
            vals.size(),
            std::vector<std::vector<double>>(
                vals.size(),
                std::vector<double>(vals[0].size(), 0.0)
            )
        );
    for (int i=0; i<vals.size(); i++) {
        result[i] = Softmax::backward(vals[i]);
    }
    return result;
}


std::vector<double> ProtectedSoftmax::forward(std::vector<double> vals) {
    /* softmax, but between 0 and 1 */
    double maxVal = max(vals);

    for (double& val : vals) {
        val -= maxVal;
    }
    return Softmax::forward(vals);
}

std::vector<std::vector<double>> ProtectedSoftmax::backward(
    const std::vector<double>& vals) {

    return Softmax::backward(vals);
}


std::vector<std::vector<double>> ProtectedSoftmax_batch::forward(const std::vector<std::vector<double>>& vals) {
    std::vector<std::vector<double>> result = std::vector<std::vector<double>>(vals.size(), std::vector<double>(vals[0].size(), 0.0));
    for (int i=0; i<vals.size(); i++) {
        result[i] = ProtectedSoftmax::forward(vals[i]);
        }
    return result;
}