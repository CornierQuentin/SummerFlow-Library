#include "Neuron.h"
#include "SummerFlow.h"
#include <math.h>
#include <iostream>

Neuron::Neuron(unsigned numOutput, unsigned neuronIndex, std::string activationFunc)
{
    for (unsigned numConnection{}; numConnection < numOutput; numConnection++)
    {
        m_outputWeight.push_back(Connection());
        m_outputWeight.back().weight = randomWeight();
    }

    index = neuronIndex;

    m_activation = activationFunc;
}

double Neuron::randomWeight(void)
{
    return rand() / double(RAND_MAX);
}

void Neuron::feedForward(std::vector<Neuron>& prevLayer)
{
    double sum{ 0.0 };

    for (unsigned prevNeuron{}; prevNeuron < prevLayer.size(); prevNeuron++)
    {
        sum += prevLayer[prevNeuron].getOutputWeight()[index].weight * prevLayer[prevNeuron].getOutputValue();
    }

    if (m_activation == "relu")
    {
        this->m_outputValue = Neuron::reluActivationFunction(sum);
    }
    else if (m_activation == "sigmoid")
    {
        this->m_outputValue = Neuron::sigmoidActivationFunction(sum);
    }
    else if (m_activation == "tanH")
    {
        this->m_outputValue = Neuron::tangentHActivationFunction(sum);
    }
    else if (m_activation == "softplus")
    {
        this->m_outputValue = Neuron::softplusActivationFunction(sum);
    }
    else
    {
        std::cout << "Wrong activation function" << std::endl;
        exit(-1);
    }
}

double Neuron::sigmoidActivationFunction(double value)
{
    return 1 / (1 + exp(-value));
}

double Neuron::sigmoidActivationFunctionDerivative(double value)
{
    return Neuron::sigmoidActivationFunction(value) * (1 - Neuron::sigmoidActivationFunction(value));
}

double Neuron::tangentHActivationFunction(double value)
{
    return tanh(value);
}

double Neuron::tangentHActivationFunctionDerivative(double value)
{
    return 1.0 - value * value;
}

double Neuron::reluActivationFunction(double value)
{
    if (value < 0)
        return 0;
    else
        return value;
}

double Neuron::reluActivationFunctionDerivative(double value)
{
    if (value < 0)
        return 0;
    else
        return 1;
}

double Neuron::softplusActivationFunction(double value)
{
    return log(1 + exp(value));
}

double Neuron::softplusActivationFunctionDerivative(double value)
{
    return 1 / (1 + exp(-value));
}

void Neuron::calculateOutputGradient(double targetVal)
{
    double delta{ targetVal };

    if (m_activation == "relu")
    {
        m_gradient = delta * Neuron::reluActivationFunctionDerivative(m_outputValue);
    }
    else if (m_activation == "sigmoid")
    {
        m_gradient = delta * Neuron::sigmoidActivationFunctionDerivative(m_outputValue);
    }
    else if (m_activation == "tanH")
    {
        m_gradient = delta * Neuron::tangentHActivationFunctionDerivative(m_outputValue);
    }
    else if (m_activation == "softplus")
    {
        m_gradient = delta * Neuron::softplusActivationFunctionDerivative(m_outputValue);
    }
    else
    {
        std::cout << "Wrong activation function" << std::endl;
        exit(-1);
    }
}

double Neuron::sumDOW(const std::vector<Neuron>& nextLayer) const
{
    double sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed.

    for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
        sum += m_outputWeight[n].weight * nextLayer[n].m_gradient;
    }

    return sum;
}

void Neuron::calculateHiddenGradient(std::vector<Neuron>& nextLayer)
{
    double dow = sumDOW(nextLayer);

    if (m_activation == "relu")
    {
        m_gradient = dow * Neuron::reluActivationFunctionDerivative(m_outputValue);
    }
    else if (m_activation == "sigmoid")
    {
        m_gradient = dow * Neuron::sigmoidActivationFunctionDerivative(m_outputValue);
    }
    else if (m_activation == "tanH")
    {
        m_gradient = dow * Neuron::tangentHActivationFunctionDerivative(m_outputValue);
    }
    else if (m_activation == "softplus")
    {
        m_gradient = dow * Neuron::softplusActivationFunctionDerivative(m_outputValue);
    }
    else
    {
        std::cout << "Wrong activation function" << std::endl;
        exit(-1);
    }
}

void Neuron::updateWeight(std::vector<Neuron>& prevLayer)
{
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer

    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        Neuron& neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeight[index].deltaWeight;

        double newDeltaWeight = eta * m_gradient;

        neuron.m_outputWeight[index].deltaWeight = newDeltaWeight;
        neuron.m_outputWeight[index].weight -= newDeltaWeight;
    }
}