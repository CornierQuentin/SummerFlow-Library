#include "SummerFlow.h"
#include <fstream>
#include <utility>
#include <stdexcept>
#include <sstream>
#include <algorithm>


CSVReader::CSVReader(std::string filename)
{
    m_fileName = filename;

    // Reads a CSV file into a vector of <string, vector<int>> pairs where
    // each pair represents <column name, column values>

    // Create a vector of <string, int vector> pairs to store the result
    std::vector<std::pair<std::string, std::vector<double>>> result;

    // Create an input filestream
    std::ifstream myFile(m_fileName);

    // Make sure the file is open
    if (!myFile.is_open()) throw std::runtime_error("Could not open file");

    // Helper vars
    std::string line, colname;
    std::string val;

    // Read the column names
    if (myFile.good())
    {
        // Extract the first line in the file
        std::getline(myFile, line);

        // Create a stringstream from line
        std::stringstream ss(line);

        // Extract each column name
        while (std::getline(ss, colname, ',')) {

            // Initialize and add <colname, int vector> pairs to result
            result.push_back({ colname, std::vector<double> {} });
        }
    }

    // Read data, line by line
    while (std::getline(myFile, line))
    {
        // Create a stringstream of the current line
        std::stringstream ss(line);

        // Keep track of the current column index
        int colIdx = 0;

        // Extract each integer
        while (std::getline(ss, val, ',')) {

            double doubleVal = std::stod(val);
            // Add the current integer to the 'colIdx' column's values vector
            result.at(colIdx).second.push_back(doubleVal);

            // If the next token is a comma, ignore it and move on
            if (ss.peek() == ',') ss.ignore();

            // Increment the column index
            colIdx++;
        }
    }

    // Close file
    myFile.close();

    m_inputData = result;


}

std::vector<std::pair<std::string, std::vector<double>>> CSVReader::getInputData()
{
    return m_inputData;
}

std::vector<double> CSVReader::getTargetData()
{
    return m_targetData;
}

void CSVReader::extractTargetColumn(unsigned columnNumber)
{
    m_targetData = m_inputData[columnNumber].second;

    std::vector<std::pair<std::string, std::vector<double>>> newInputData;

    for (unsigned column{}; column < m_inputData.size(); column++)
    {
        if (column != columnNumber)
        {
            newInputData.push_back(m_inputData[column]);
        }
    }

    m_inputData = newInputData;
}

void CSVReader::normalizeColumns(std::vector<unsigned> columnToNormalize)
{
    for (unsigned columnNumber{}; columnNumber < columnToNormalize.size(); columnNumber++)
    {
        for (unsigned elements{}; elements < m_inputData[columnToNormalize[columnNumber]].second.size(); elements++)
        {
            auto minmax = std::minmax_element(m_inputData[columnToNormalize[columnNumber]].second.begin(), m_inputData[columnToNormalize[columnNumber]].second.end());
            m_inputData[columnToNormalize[columnNumber]].second[elements] = normalizeValue(m_inputData[columnToNormalize[columnNumber]].second[elements], *minmax.second, *minmax.first);
        }
    }
}

std::vector<std::vector<double>> CSVReader::targetToVector()
{
    std::vector<std::vector<double>> newTargetData;

    for (unsigned target{}; target < m_targetData.size(); target++)
    {
        std::vector<double> newTarget = { m_targetData[target] };
        newTargetData.push_back(newTarget);
    }

    return newTargetData;
}