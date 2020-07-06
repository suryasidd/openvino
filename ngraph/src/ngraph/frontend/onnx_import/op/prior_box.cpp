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
#include "prior_box.hpp"
#include "ngraph/op/prior_box.hpp"
//#include "ngraph/opsets/opset3.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector PriorBox(const Node& node)
                {
                    std::cout << "In prior box layer" << std::endl;
                    const Output<ngraph::Node> input1 = node.get_ng_inputs().at(0);
                    const Output<ngraph::Node> input2 = node.get_ng_inputs().at(1);
                   
                    ngraph::Output<ngraph::Node> layer_shapes = ngraph::opset3::Constant::create(element::i64, Shape{2}, {input1.get_shape()[2], input1.get_shape()[3]});
                    ngraph::Output<ngraph::Node> image_shapes = ngraph::opset3::Constant::create(element::i64, Shape{2}, {input2.get_shape()[2], input2.get_shape()[3]});
                   
                    ngraph::op::PriorBoxAttrs attrs;
                    attrs.min_size = node.get_attribute_value<std::vector<float>>("min_size", {}); 
                    attrs.max_size = node.get_attribute_value<std::vector<float>>("max_size", {}); 
                    attrs.aspect_ratio = node.get_attribute_value<std::vector<float>>("aspect_ratio", {}); 
                    attrs.density = node.get_attribute_value<std::vector<float>>("density", {}); 
                    attrs.fixed_ratio = node.get_attribute_value<std::vector<float>>("fixed_ratio", {}); 
                    attrs.fixed_size = node.get_attribute_value<std::vector<float>>("fixed_size", {});  
                    attrs.clip = false;
                    attrs.flip = false;
                    attrs.step = node.get_attribute_value<float>("step",1.0f);
                    attrs.offset = node.get_attribute_value<float>("offset",0);
                    attrs.variance = node.get_attribute_value<std::vector<float>>("variance", {}); 
                    attrs.scale_all_sizes = false;

                    const auto prior_box_data = std::make_shared<default_opset::PriorBox>(layer_shapes, image_shapes, attrs);

                    ngraph::Output<ngraph::Node> unsqueeze_axes = ngraph::opset3::Constant::create(element::i64, Shape{1}, {0});
                    return {std::make_shared<default_opset::Unsqueeze>(prior_box_data, unsqueeze_axes)};

               
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
