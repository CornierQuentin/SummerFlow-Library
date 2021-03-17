#include "SummerFlow.h"
#include <iostream>
#include <cassert>
#include <math.h>

Network::Network() { }

void Network::add(unsigned numNeuron, unsigned numOutput, std::string activationFunc)
{
    std::vector<Neuron> layerCreation;
    for (unsigned neuron{}; neuron <= numNeuron; neuron++)
    {
        layerCreation.push_back(Neuron(numOutput, neuron, activationFunc));
    }

    m_layers.push_back(layerCreation);

    // Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
    m_layers.back().back().setOutputValue(1.0);
}

void Network::feedForward(const std::vector<double>& inputValue)
{
    assert(inputValue.size() == m_layers[0].size() - 1);

    for (unsigned input{}; input < inputValue.size(); input++)
    {
        m_layers[0][input].setOutputValue(inputValue[input]);
    }

    for (unsigned layer{ 1 }; layer < m_layers.size(); layer++)
    {
        for (unsigned neuron{}; neuron < m_layers[layer].size() - 1; neuron++)
        {
            m_layers[layer][neuron].feedForward(m_layers[layer - 1]);
        }

    }
}

void Network::MSELossFunction(const std::vector<double>& targetVal)
{
    m_error = 0.0;

    for (unsigned output{}; output < targetVal.size(); output++)
    {
        double delta{ targetVal[output] - m_layers.back()[output].getOutputValue() };
        m_error += delta * delta;
    }
    m_error = m_error / targetVal.size();
}

void Network::MAELossFunction(const std::vector<double>& targetVal)
{
    m_error = 0.0;

    for (unsigned output{}; output < targetVal.size(); output++)
    {
        double delta{ targetVal[output] - m_layers.back()[output].getOutputValue() };
        m_error += std::abs(delta);
    }
    m_error = m_error / targetVal.size();
}

void Network::MBELossFunction(const std::vector<double>& targetVal)
{
    m_error = 0.0;

    for (unsigned output{}; output < targetVal.size(); output++)
    {
        double delta{ targetVal[output] - m_layers.back()[output].getOutputValue() };
        m_error += delta;
    }
    m_error = m_error / targetVal.size();
}

void Network::binaryCrossEntropyLossFunction(const std::vector<double>& targetVal)
{
    m_error = -(targetVal[0] * log(m_layers.back()[0].getOutputValue()) + (1 - targetVal[0]) * log(1 - m_layers.back()[0].getOutputValue()));
}

void Network::backProp(const std::vector<double> &targetVal, std::string loss)
{
    // Calculate the loss
    if (loss == "MSE")
    {
        this->MSELossFunction(targetVal);
    }
    else if (loss == "MAE")
    {
        this->MAELossFunction(targetVal);
    }
    else if (loss == "MBE")
    {
        this->MBELossFunction(targetVal);
    }
    else if (loss == "binaryCrossEntropy")
    {
        this->binaryCrossEntropyLossFunction(targetVal);
    }

    // Calculate the output layer gradients

    for (unsigned output{}; output < targetVal.size(); output++)
    {
        m_layers.back()[output].calculateOutputGradient(m_error);
    }

    // Calculate the hidden layer gradients

    for (auto layer{ m_layers.size() - 2 }; layer > 0; layer--)
    {
        for (unsigned neuron{}; neuron < m_layers[layer].size(); neuron++)
        {
            m_layers[layer][neuron].calculateHiddenGradient(m_layers[layer + 1]);
        }
    }

    // Update layer weight

    for (auto layer{ m_layers.size() - 1 }; layer > 0; layer--)
    {
        Layer& actualLayer = m_layers[layer];
        Layer& prevLayer = m_layers[layer - 1];
        for (unsigned neuron{}; neuron < actualLayer.size() - 1; neuron++)
        {
            actualLayer[neuron].updateWeight(prevLayer);
        }
    }
}

void Network::getResult(std::vector<double> &resultVals)
{
    resultVals.clear();

    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
        resultVals.push_back(m_layers.back()[n].getOutputValue());
    }
}

void Network::fit(std::vector<std::pair<std::string, std::vector<double>>> inputVals, std::vector<std::vector<double>> targetVals, unsigned numEpisode, std::string loss, std::string type)
{
    unsigned endTrainingSet = (80 * inputVals[0].second.size()) / 100;

    std::cout << "\nStarting training phase :\n" << std::endl;

    for (unsigned episode{}; episode < numEpisode; episode++)
    {
        std::string progressBar = this->displayProgressionBar(numEpisode, episode + 1);

        std::cout << progressBar << "\r" << std::flush;

        for (unsigned row{}; row < endTrainingSet; row++)
        {
            std::vector<double> input;
            for (unsigned categories{}; categories < inputVals.size(); categories++)
            {
                input.push_back(inputVals[categories].second[row]);
            }

            this->feedForward(input);

            std::vector<double> resultVals;

            this->getResult(resultVals);

            this->backProp(targetVals[row], loss);
        }
    }

    if (type == "classifier")
    {
        std::cout << std::endl << "\nStarting evaluating phase :" << std::endl;

        float correctPredictions{};
        float totalPredictions{};

        for (unsigned validationRow{ endTrainingSet }; validationRow < inputVals[0].second.size(); validationRow++)
        {
            std::vector<double> input;
            for (unsigned categories{}; categories < inputVals.size(); categories++)
            {
                input.push_back(inputVals[categories].second[validationRow]);
            }

            std::vector<double> prediction = this->predict(input);

            unsigned correctOutput{};

            for (unsigned predictionNumber{}; predictionNumber < targetVals[validationRow].size(); predictionNumber++)
            {
                unsigned result{};

                if (prediction[predictionNumber] >= 0.5)
                {
                    result = 1;
                }
                else {
                    result = 0;
                }

                if (result == targetVals[validationRow][predictionNumber])
                {
                    correctOutput++;
                }
            }

            if (correctOutput == targetVals[validationRow].size())
            {
                correctPredictions++;
            }

            totalPredictions++;
        }

        if (totalPredictions == 0)
            totalPredictions++;

        std::cout << "\nFor " << totalPredictions << " predictions :" << "\n\nAccuracy : " << (static_cast<double>(correctPredictions) / static_cast<double>(totalPredictions)) * 100 << "%" << std::endl;
    }

    if (type == "regression")
    {
        std::cout << std::endl << "\nStarting evaluating phase :" << std::endl;

        std::vector<double> lossList;

        for (unsigned validationRow{ endTrainingSet }; validationRow < inputVals[0].second.size(); validationRow++)
        {
            std::vector<double> input;
            for (unsigned categories{}; categories < inputVals.size(); categories++)
            {
                input.push_back(inputVals[categories].second[validationRow]);
            }

            std::vector<double> prediction = this->predict(input);

            for (unsigned predictionNumber{}; predictionNumber < targetVals[validationRow].size(); predictionNumber++)
            {
                if (loss == "MSE")
                {
                    this->MSELossFunction(targetVals[validationRow]);
                }
                else if (loss == "MAE")
                {
                    this->MAELossFunction(targetVals[validationRow]);
                }
                else if (loss == "MBE")
                {
                    this->MBELossFunction(targetVals[validationRow]);
                }
                else if (loss == "binaryCrossEntropy")
                {
                    this->binaryCrossEntropyLossFunction(targetVals[validationRow]);
                }

                lossList.push_back(m_error);
            }
        }

        double lossSum{};

        for (unsigned loss{}; loss < lossList.size(); loss++)
        {
            lossSum += lossList[loss];
        }

        std::cout << "\nFor " << lossList.size() << " predictions :" << "\n\nMean loss : " << static_cast<double>(lossSum) / static_cast<double>(lossList.size()) << std::endl;
    }
}

std::vector<double> Network::predict(std::vector<double> inputVals)
{
    this->feedForward(inputVals);

    std::vector<double> results;

    this->getResult(results);

    return results;
}

std::string Network::displayProgressionBar(unsigned numEpisode, unsigned actualEpisode)
{
    float progress = static_cast<double>(actualEpisode) / static_cast<double>(numEpisode);

    int barWidth = 70;

    std::string progressBar{};

    progressBar += "Episode " + std::to_string(actualEpisode) + " : [";
    int pos = barWidth * progress;
    for (int barFilling = 0; barFilling < barWidth; barFilling++) {
        if (barFilling < pos)
        {
            progressBar += "=";
        }
        else if (barFilling == pos)
        {
            progressBar += ">";
        }
        else
        {
            progressBar += " ";
        }
    }
    progressBar += "] " + std::to_string(int(progress * 100.0)) + " %";

    return progressBar;
}

void Network::changeHyperParameters(double learningRate, double momentum)
{
    for (unsigned layer{}; layer < m_layers.size(); layer++)
    {
        for (unsigned neuron{}; neuron < m_layers[layer].size(); neuron++)
        {
            m_layers[layer][neuron].setEta(learningRate);
            m_layers[layer][neuron].setAlpha(momentum);
        }
    }
}