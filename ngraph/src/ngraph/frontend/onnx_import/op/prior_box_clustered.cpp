//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <memory>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/node.hpp"
#include "ngraph/validation_util.hpp"
#include "prior_box_clustered.hpp"
#include "ngraph/op/prior_box_clustered.hpp"
//#include "ngraph/opsets/opset3.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector PriorBoxClustered(const Node& node)
                {
                    std::cout << "In prior box layer" << std::endl;
                    const Output<ngraph::Node> input1 = node.get_ng_inputs().at(0);
                    const Output<ngraph::Node> input2 = node.get_ng_inputs().at(1);
                   
                    ngraph::Output<ngraph::Node> layer_shapes = ngraph::opset3::Constant::create(element::i64, Shape{2}, {input1.get_shape()[2], input1.get_shape()[3]});
                    ngraph::Output<ngraph::Node> image_shapes = ngraph::opset3::Constant::create(element::i64, Shape{2}, {input2.get_shape()[2], input2.get_shape()[3]});
                   
                    ngraph::op::PriorBoxClusteredAttrs attrs;
                    attrs.widths = node.get_attribute_value<std::vector<float>>("width", {}); 
                    attrs.heights = node.get_attribute_value<std::vector<float>>("height", {}); 
                    attrs.clip = false;
                    attrs.step_widths = node.get_attribute_value<float>("step_w",1);
                    attrs.step_heights = node.get_attribute_value<float>("step_h",1);
                    attrs.offset = node.get_attribute_value<float>("offset",0);
                    attrs.variances = node.get_attribute_value<std::vector<float>>("variance", {}); 

                    const auto prior_box_data = std::make_shared<default_opset::PriorBoxClustered>(layer_shapes, image_shapes, attrs);

                    //auto axes_node = std::make_shared<default_opset::Constant>(
                    //    element::i64, Shape{1}, {1});
                    ngraph::Output<ngraph::Node> unsqueeze_axes = ngraph::opset3::Constant::create(element::i64, Shape{1}, {0});
                    return {std::make_shared<default_opset::Unsqueeze>(prior_box_data, unsqueeze_axes)};
               
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
