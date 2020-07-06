// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <map>

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

#include "single_layer_tests/eltwise.hpp"

namespace LayerTestsDefinitions {
std::ostream& operator<<(std::ostream & os, EltwiseParams::OpType type) {
    switch (type) {
        case EltwiseParams::OpType::SCALAR:
            os << "SCALAR";
            break;
        case EltwiseParams::OpType::VECTOR:
            os << "VECTOR";
            break;
        default:
            THROW_IE_EXCEPTION << "NOT_SUPPORTED_OP_TYPE";
    }
    return os;
}

std::string EltwiseLayerTest::getTestCaseName(testing::TestParamInfo<EltwiseTestParams> obj) {
    std::vector<std::vector<size_t>> inputShapes;
    InferenceEngine::Precision netPrecision;
    ngraph::helpers::InputLayerType secondaryInputType;
    EltwiseParams::OpType opType;
    ngraph::helpers::EltwiseTypes eltwiseOpType;
    std::string targetName;
    std::map<std::string, std::string> additional_config;
    std::tie(inputShapes, eltwiseOpType, secondaryInputType, opType, netPrecision, targetName, additional_config) = obj.param;
    std::ostringstream results;

    results << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    results << "eltwiseOpType=" << eltwiseOpType << "_";
    results << "secondaryInputType=" << secondaryInputType << "_";
    results << "opType=" << opType << "_";
    results << "netPRC=" << netPrecision.name() << "_";
    results << "targetDevice=" << targetName;
    return results.str();
}

void EltwiseLayerTest::SetUp() {
    std::vector<std::vector<size_t>> inputShapes;
    InferenceEngine::Precision netPrecision;
    ngraph::helpers::InputLayerType secondaryInputType;
    EltwiseParams::OpType opType;
    ngraph::helpers::EltwiseTypes eltwiseType;
    std::map<std::string, std::string> additional_config;
    std::tie(inputShapes, eltwiseType, secondaryInputType, opType, netPrecision, targetDevice, additional_config) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    std::vector<size_t> inputShape1, inputShape2;
    if (inputShapes.size() == 1) {
        inputShape1 = inputShape2 = inputShapes.front();
    } else if (inputShapes.size() == 2) {
        inputShape1 = inputShapes.front();
        inputShape2 = inputShapes.back();
    } else {
        THROW_IE_EXCEPTION << "Incorrect number of input shapes";
    }

    configuration.insert(additional_config.begin(), additional_config.end());
    auto input = ngraph::builder::makeParams(ngPrc, {inputShape1});

    std::vector<size_t> shape_input_secondary;
    switch (opType) {
        case EltwiseParams::OpType::SCALAR: {
            shape_input_secondary = std::vector<size_t>({1});
            break;
        }
        case EltwiseParams::OpType::VECTOR:
            shape_input_secondary = inputShape2;
            break;
        default:
            FAIL() << "Unsupported Secondary operation type";
    }

    auto secondaryInput = ngraph::builder::makeInputLayer(ngPrc, secondaryInputType, shape_input_secondary);
    if (secondaryInputType == ngraph::helpers::InputLayerType::PARAMETER) {
        input.push_back(std::dynamic_pointer_cast<ngraph::opset3::Parameter>(secondaryInput));
    }
    auto eltwise = ngraph::builder::makeEltwise(input[0], secondaryInput, eltwiseType);
    function = std::make_shared<ngraph::Function>(eltwise, input, "Eltwise");
}


TEST_P(EltwiseLayerTest, EltwiseTests) {
    Run();
}
} // namespace LayerTestsDefinitions