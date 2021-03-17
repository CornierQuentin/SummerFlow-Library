#if !defined(__NEURON__)
#define __NEURON__

#include <vector>
#include <string>

struct Connection
{
    double weight;
    double deltaWeight;
};


class Neuron
{
public:
    Neuron(unsigned numOutput, unsigned neuronIndex, std::string activationFunc);
    double randomWeight(void);
    void feedForward(std::vector<Neuron>& prevLayer);
    void calculateOutputGradient(double targetVal);
    void calculateHiddenGradient(std::vector<Neuron>& nextLayer);
    void updateWeight(std::vector<Neuron>& prevLayer);
    double sumDOW(const std::vector<Neuron>& nextLayer) const;

    void setOutputValue(double newValue) { this->m_outputValue = newValue; };
    double getOutputValue(void) { return this->m_outputValue; };
    void setEta(double newEta) { this->eta = newEta; }
    void setAlpha(double newAlpha) { this->alpha = newAlpha; }
    std::vector<Connection> getOutputWeight(void) { return m_outputWeight; };

private:
    double m_outputValue;
    std::vector<Connection> m_outputWeight;
    std::string m_activation;
    unsigned index;
    double m_gradient;

    double eta{ 0.001 };
    double alpha{ 0.3 };

    double sigmoidActivationFunction(double value);
    double sigmoidActivationFunctionDerivative(double value);

    double tangentHActivationFunction(double value);
    double tangentHActivationFunctionDerivative(double value);

    double reluActivationFunction(double value);
    double reluActivationFunctionDerivative(double value);

    double softplusActivationFunction(double value);
    double softplusActivationFunctionDerivative(double value);
};

#endif