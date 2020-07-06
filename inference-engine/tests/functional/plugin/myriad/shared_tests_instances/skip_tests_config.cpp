// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    return {
        // Not supported activation types
        ".*ActivationLayerTest\\.CompareWithRefs/Tanh.*netPRC=FP32.*",
        ".*ActivationLayerTest\\.CompareWithRefs/Exp.*netPRC=FP32.*",
        ".*ActivationLayerTest\\.CompareWithRefs/Log.*netPRC=FP32.*",
        ".*ActivationLayerTest\\.CompareWithRefs/Sigmoid.*netPRC=FP32.*",
        ".*ActivationLayerTest\\.CompareWithRefs/Relu.*netPRC=FP32.*",
        ".*Behavior.*ExecGraphTests.*",
        // TODO: Issue: 26268
        ".*ConcatLayerTest.*axis=0.*",
        // TODO: Issue 31197
        R"(.*(IEClassBasicTestP).*smoke_registerPluginsXMLUnicodePath.*)",
        // TODO: Issue: 34206
        R"(.*(IEClassLoadNetwork).*(QueryNetworkMULTIWithHETERONoThrow_V10|QueryNetworkHETEROWithMULTINoThrow_V10).*)",
        // TODO: Issue: 34348
        R"(.*IEClassGetAvailableDevices.*)",
    };
}
