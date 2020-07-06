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
#include "detection_output.hpp"
#include "ngraph/op/detection_output.hpp"
//#include "ngraph/opsets/opset3.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector DetectionOutput(const Node& node)
                {
                    const Output<ngraph::Node> box_logits = node.get_ng_inputs().at(0);
                    const Output<ngraph::Node> class_preds = node.get_ng_inputs().at(1);
                    const Output<ngraph::Node> proposals = node.get_ng_inputs().at(2);
                    //Output<ngraph::Node> layer_shapes=input2;
                    //Output<ngraph::Node> image_shapes=input1;
                    
                    //image_shapes = input2;//.get_shape();//.size();
                    //layer_shapes = input1;//.get_shape();//.size();
                    //std::cout << "Input 1 box_logits shape: " << proposals.get_shape()[0] << " " << proposals.get_shape()[1] << " " << proposals.get_shape()[2] << " " << proposals.get_shape()[3] << std::endl;//" " << input1.get_shape()[3] << " " << input2.get_shape()[2] << " " << input2.get_shape()[3] << std::endl; 

                    ngraph::op::DetectionOutputAttrs attrs;
                    attrs.num_classes = node.get_attribute_value<int64_t>("num_classes", 2); 
                    attrs.background_label_id = node.get_attribute_value<int64_t>("background_label_id", 0); 
                    attrs.top_k = node.get_attribute_value<int64_t>("top_k", 200);
                    attrs.variance_encoded_in_target = node.get_attribute_value<int64_t>("variance_encoded_in_target", 0);
                    attrs.keep_top_k.push_back(node.get_attribute_value<int64_t>("keep_top_k", 200));
                    //attrs.keep_top_k = node.get_attribute_value<int64_t>("keep_top_k", 200);
                    attrs.code_type = std::string{"caffe.PriorBoxParameter."} + node.get_attribute_value<std::string>("code_type", std::string{"CENTER_SIZE"});
                    attrs.share_location = node.get_attribute_value<int64_t>("share_location", 1);
                    attrs.normalized = node.get_attribute_value<int64_t>("normalized", 1);
                    attrs.nms_threshold = node.get_attribute_value<float>("nms_threshold"); 
                    attrs.confidence_threshold = node.get_attribute_value<float>("confidence_threshold", std::numeric_limits<float>::min()); 

                                   
                    return {std::make_shared<default_opset::DetectionOutput>(box_logits, class_preds, proposals, attrs)};  

          
              }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
