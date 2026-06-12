#include <cmath>
#include <vector>

#include "Mathlib.hpp"

class Pass {
public:
    /* returns x */
    static double forward(double val) {
        return val;
    }

    static double backward(double val) {
        /* derivative of x with respect to x */
        return 1.0;
    }
};

class ReLU {
public:
    /* Returns 0 if values and x if its not */
    static double forward(double val) {
        return std::max(0.0, val);
    }

    static double backward(double val) {
        if (val > 0.0) {
            return 1.0;
        }
        return 0.0;
    }
};

class LeakyReLU {
public:
    /*
    ReLU with small negative slope to prevent dying ReLU problem. Gradient flows even when val < 0
    */
    static constexpr double alpha = 0.01;  // small negative slope

    static double forward(double val) {
        return std::max(LeakyReLU::alpha * val, val);
    }

    static double backward(double val) {
        /* partial derivative of LeakyReLU with respect to x */
        if (val > 0) {
            return 1.0;
        }
        return LeakyReLU::alpha;
    }
};

class Softmax{
public:
    /*
    converts any output to be squashed from 0 to 1 and also had a nice derivative when paired with  Cross-Entropy
    */

    static std::vector<double> forward(std::vector<double> vals) {
        for (double& val : vals) {
            val = std::exp(val);
        }
        return normalize(vals);
    }

    static std::vector<std::vector<double>> backward(const std::vector<double>& vals) {
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
};

class Softmax_batch {
public:
    static std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& vals) {
        /* same as regular forward, but for a batch */
        std::vector<std::vector<double>> result = std::vector<std::vector<double>>(vals.size(), std::vector<double>(vals.size(), 0.0));

        for (int i=0; i<vals.size(); i++) {
            result[i] = Softmax::forward(vals[i]);
        }
    }
    static std::vector<std::vector<std::vector<double>>> backward(const std::vector<std::vector<double>>& vals) {
        std::vector<std::vector<std::vector<double>>> result =
            std::vector<std::vector<std::vector<double>>>(
                vals.size(),
                std::vector<std::vector<double>>(
                    vals.size(),
                    std::vector<double>(vals.size(), 0.0)
                )
            );
        for (int i=0; i<vals.size(); i++) {
            result[i] = Softmax::backward(vals[i]);
        }
        return result;
    }
};

class ProtectedSoftmax {
public:
    /* Softmax but without any overflow from e^x being too large */
    static std::vector<double> forward(std::vector<double> vals) {
        /* softmax, but between 0 and 1 */
        double maxVal = max(vals);

        for (double& val : vals) {
            val -= maxVal;
        }
        return Softmax::forward(vals);
    }

    static std::vector<std::vector<double>> backward(
        const std::vector<double>& vals) {

        return Softmax::backward(vals);
    }
};

class ProtectedSoftmax_batch {
public:
    static std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& vals) {
        std::vector<std::vector<double>> result = std::vector<std::vector<double>>(vals.size(), std::vector<double>(vals.size(), 0.0));
        for (int i=0; i<vals.size(); i++) {
            result[i] = Softmax::forward(vals[i]);
            }
        return result;
    }
};