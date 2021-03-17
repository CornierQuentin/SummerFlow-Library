#if !defined(__SUMMERFLOW__)
#define __SUMMERFLOW

#include "Neuron.h"
#include <fstream>
#include <utility>
#include <stdexcept>
#include <sstream>

typedef std::vector<Neuron> Layer;

class Network
{
public:
    Network();
    void feedForward(const std::vector<double>& inputValue);

    void backProp(const std::vector<double>& targetValue, std::string loss);

    void getResult(std::vector<double>& resultValue);
    void add(unsigned numNeuron, unsigned numOutput, std::string activationFunc);
    void fit(std::vector<std::pair<std::string, std::vector<double>>> inputVals, std::vector<std::vector<double>> targetVals, unsigned numEpisode, std::string loss, std::string type);
    std::vector<double> predict(std::vector<double> inputVals);
    std::string displayProgressionBar(unsigned numEpisode, unsigned actualEpisode);
    void changeHyperParameters(double learningRate, double momentum);

private:
    std::vector<Layer> m_layers; // m_layer[layerNum][neuroNum]
    double m_error;

    void MSELossFunction(const std::vector<double>& targetVal);
    void MAELossFunction(const std::vector<double>& targetVal);
    void MBELossFunction(const std::vector<double>& targetVal);

    void binaryCrossEntropyLossFunction(const std::vector<double>& targetVal);
};

class CSVReader
{
public:
    CSVReader(std::string filename);

    std::vector<std::pair<std::string, std::vector<double>>> getInputData();
    std::vector<double> getTargetData();

    void extractTargetColumn(unsigned columnNumber);
    double normalizeValue(double val, double max, double min) { return (val - min) / (max - min); }
    void normalizeColumns(std::vector<unsigned> columnToNormalize);
    std::vector<std::vector<double>> targetToVector();

private:
    std::string m_fileName;
    std::vector<double> m_targetData;
    std::vector<std::pair<std::string, std::vector<double>>> m_inputData;
};

#endif