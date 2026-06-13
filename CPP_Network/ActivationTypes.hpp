#pragma once

enum class ActivationType {
    RELU,
    LEAKY_RELU,
    PASS
};

typedef double (*ActivationForwardFunc)(double);
typedef double (*ActivationBackwardFunc)(double);