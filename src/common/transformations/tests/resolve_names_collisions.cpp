// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/resolve_names_collisions.hpp"

#include <gmock/gmock.h>

#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/util/node_util.hpp"
#include "openvino/opsets/opset8_decl.hpp"
#include "openvino/pass/manager.hpp"

namespace ov::test {
using namespace ov::opset8;

TEST(ResolveNameCollisionsTest, FixGeneratedNames) {
    auto arg0 = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{1, 3, 3, 3});
    const auto gen_friendly_name = arg0->get_friendly_name();

    std::string name = "Parameter_";
    EXPECT_NE(std::string::npos, gen_friendly_name.find("Parameter_"));
    unsigned long long index = std::stoull(gen_friendly_name.substr(name.length()));
    name += std::to_string(++index);

    arg0->set_friendly_name(name);

    auto arg1 = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3, 3});

    auto concat = std::make_shared<Concat>(ov::NodeVector{arg0, arg1}, 1);
    auto result1 = std::make_shared<Result>(concat);

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result1}, ov::ParameterVector{arg0, arg1});

    EXPECT_EQ(name, arg0->get_friendly_name());
    EXPECT_EQ(arg1->get_friendly_name(), arg0->get_friendly_name());
    EXPECT_NE(arg1->get_friendly_name(), arg0->get_friendly_name() + "_2");

    ov::pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::ResolveNameCollisions>();
    pass_manager.run_passes(model);
    EXPECT_EQ(name, arg0->get_friendly_name());
    EXPECT_NE(arg1->get_friendly_name(), arg0->get_friendly_name());
    EXPECT_EQ(arg1->get_friendly_name(), arg0->get_friendly_name() + "_2");
}

TEST(ResolveNameCollisionsTest, FixFriendlyNamesForAutogeneratedNames) {
    auto arg0 = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{1, 3, 3, 3});
    const auto gen_friendly_name = arg0->get_friendly_name();

    auto arg1 = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3, 3});
    // set the same name as for the first Parameter
    arg1->set_friendly_name(gen_friendly_name);

    auto concat1 = std::make_shared<Concat>(ov::NodeVector{arg0, arg1}, 1);
    concat1->set_friendly_name("concat");
    auto concat = std::make_shared<Concat>(ov::NodeVector{concat1, arg1}, 1);
    concat->set_friendly_name("concat");

    auto result1 = std::make_shared<Result>(concat);

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result1}, ov::ParameterVector{arg0, arg1});

    EXPECT_EQ(concat->get_friendly_name(), concat1->get_friendly_name());

    ov::pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::ResolveNameCollisions>();
    pass_manager.run_passes(model);

    // these names weren't set automatically, and have to remain the same.
    EXPECT_EQ(concat->get_friendly_name(), concat1->get_friendly_name());
    // arg0's name was set automatically and matches with another name in the graph,
    // so it have to be changed.
    EXPECT_NE(arg0->get_friendly_name(), arg1->get_friendly_name());
    EXPECT_EQ(arg0->get_friendly_name(), arg1->get_friendly_name() + "_2");
}

TEST(ResolveNameCollisionsTest, FixFriendlyNamesForAutogeneratedNamesMultiSubgraphOp) {
    // external params
    auto X = std::make_shared<Parameter>(element::f32, Shape{4});
    auto Y = std::make_shared<Parameter>(element::f32, Shape{4});
    auto Z = std::make_shared<Parameter>(element::f32, Shape{8});

    auto axis = std::make_shared<Constant>(element::i32, Shape{}, 0);
    auto external_split = std::make_shared<Split>(X, axis, 2);

    // internal params
    auto Xt = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    Xt->set_friendly_name(X->get_friendly_name());
    auto Yt = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    Yt->set_friendly_name(Y->get_friendly_name());
    auto Ze = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    // then body
    auto cond = std::make_shared<Constant>(element::boolean, Shape{1}, true);
    auto axis_then = std::make_shared<Constant>(element::i32, Shape{}, 0);
    auto split_y = std::make_shared<Split>(Yt, axis_then, 2);
    split_y->set_friendly_name(external_split->get_friendly_name());
    auto then_op = std::make_shared<Subtract>(Xt, split_y->output(0));
    auto res0 = std::make_shared<Result>(then_op);

    // else body
    auto axis_else = std::make_shared<Constant>(element::i32, Shape{}, 0);
    auto split_z = std::make_shared<Split>(Ze, axis_else, 4);
    split_z->set_friendly_name(external_split->get_friendly_name());
    auto else_op = std::make_shared<Relu>(split_z);
    else_op->set_friendly_name(then_op->get_friendly_name());
    auto res1 = std::make_shared<Result>(else_op);

    // If set up
    auto then_body = std::make_shared<ov::Model>(OutputVector{res0}, ParameterVector{Yt, Xt}, "then_body");
    auto else_body = std::make_shared<ov::Model>(OutputVector{res1}, ParameterVector{Ze}, "else_body");
    auto if_op = std::make_shared<op::v8::If>(cond);
    if_op->set_then_body(then_body);
    if_op->set_else_body(else_body);
    if_op->set_input(external_split->output(0), Xt, nullptr);
    if_op->set_input(Y, Yt, nullptr);
    if_op->set_input(Z, nullptr, Ze);
    auto result = if_op->set_output(res0, res1);

    auto res = std::make_shared<Result>(result);
    auto model = std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Y, Z});

    EXPECT_EQ(external_split->get_friendly_name(), split_y->get_friendly_name());
    EXPECT_EQ(external_split->get_friendly_name(), split_z->get_friendly_name());

    EXPECT_EQ(X->get_friendly_name(), Xt->get_friendly_name());
    EXPECT_EQ(Y->get_friendly_name(), Yt->get_friendly_name());

    EXPECT_EQ(then_op->get_friendly_name(), else_op->get_friendly_name());

    ov::pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::ResolveNameCollisions>();
    pass_manager.run_passes(model);

    EXPECT_EQ(external_split->get_friendly_name(), split_y->get_friendly_name() + "_2");

    EXPECT_EQ(X->get_friendly_name(), Xt->get_friendly_name() + "_2");
    EXPECT_EQ(Y->get_friendly_name(), Yt->get_friendly_name() + "_2");

    EXPECT_EQ(then_op->get_friendly_name(), else_op->get_friendly_name() + "_2");
    // remain the same, because they were set via "set_friendly_name" method
    // and are not autogenerated.
    EXPECT_EQ(split_y->get_friendly_name(), split_z->get_friendly_name());
}

TEST(ResolveNameCollisionsTest, FixAllFriendlyNamesMultiSubgraphOp) {
    // external params
    auto X = std::make_shared<Parameter>(element::f32, Shape{4});
    auto Y = std::make_shared<Parameter>(element::f32, Shape{4});
    auto Z = std::make_shared<Parameter>(element::f32, Shape{8});

    auto axis = std::make_shared<Constant>(element::i32, Shape{}, 0);
    auto external_split = std::make_shared<Split>(X, axis, 2);

    // internal params
    auto Xt = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    Xt->set_friendly_name(X->get_friendly_name());
    auto Yt = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    Yt->set_friendly_name(Y->get_friendly_name());
    auto Ze = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    // then body
    auto cond = std::make_shared<Constant>(element::boolean, Shape{1}, true);
    auto axis_then = std::make_shared<Constant>(element::i32, Shape{}, 0);
    auto split_y = std::make_shared<Split>(Yt, axis_then, 2);
    split_y->set_friendly_name(external_split->get_friendly_name());
    auto then_op = std::make_shared<Subtract>(Xt, split_y->output(0));
    auto res0 = std::make_shared<Result>(then_op);

    // else body
    auto axis_else = std::make_shared<Constant>(element::i32, Shape{}, 0);
    auto split_z = std::make_shared<Split>(Ze, axis_else, 4);
    split_z->set_friendly_name(external_split->get_friendly_name());
    auto else_op = std::make_shared<Relu>(split_z);
    else_op->set_friendly_name(then_op->get_friendly_name());
    auto res1 = std::make_shared<Result>(else_op);

    // If set up
    auto then_body = std::make_shared<ov::Model>(OutputVector{res0}, ParameterVector{Yt, Xt}, "then_body");
    auto else_body = std::make_shared<ov::Model>(OutputVector{res1}, ParameterVector{Ze}, "else_body");
    auto if_op = std::make_shared<op::v8::If>(cond);
    if_op->set_then_body(then_body);
    if_op->set_else_body(else_body);
    if_op->set_input(external_split->output(0), Xt, nullptr);
    if_op->set_input(Y, Yt, nullptr);
    if_op->set_input(Z, nullptr, Ze);
    auto result = if_op->set_output(res0, res1);

    auto res = std::make_shared<Result>(result);
    auto model = std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Y, Z});

    EXPECT_EQ(external_split->get_friendly_name(), split_y->get_friendly_name());
    EXPECT_EQ(external_split->get_friendly_name(), split_z->get_friendly_name());

    EXPECT_EQ(X->get_friendly_name(), Xt->get_friendly_name());
    EXPECT_EQ(Y->get_friendly_name(), Yt->get_friendly_name());

    EXPECT_EQ(then_op->get_friendly_name(), else_op->get_friendly_name());

    ov::pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::ResolveNameCollisions>(true);
    pass_manager.run_passes(model);

    EXPECT_EQ(external_split->get_friendly_name() + "_1", split_y->get_friendly_name());
    EXPECT_EQ(external_split->get_friendly_name() + "_2", split_z->get_friendly_name());

    EXPECT_EQ(X->get_friendly_name() + "_1", Xt->get_friendly_name());
    EXPECT_EQ(Y->get_friendly_name() + "_1", Yt->get_friendly_name());

    EXPECT_EQ(then_op->get_friendly_name() + "_1", else_op->get_friendly_name());
}

using testing::UnorderedElementsAre;
using ResolveTensorNamesTest = testing::Test;

TEST_F(ResolveTensorNamesTest, param_result_model_no_name_collision) {
    auto input_1 = std::make_shared<Parameter>(element::f32, Shape{1, 3});
    auto input_2 = std::make_shared<Parameter>(element::f32, Shape{1, 3});
    auto result_1 = std::make_shared<Result>(input_1);
    auto result_2 = std::make_shared<Result>(input_2);

    input_1->output(0).set_names({"input_1", "name"});
    input_2->output(0).set_names({"input_2", "test"});

    auto model = std::make_shared<Model>(ResultVector{result_1, result_2}, ParameterVector{input_1, input_2});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ResolveNameCollisions>();
    pass_manager.run_passes(model);

    EXPECT_THAT(input_1->output(0).get_names(), UnorderedElementsAre("input_1", "name"));
    EXPECT_THAT(input_2->output(0).get_names(), UnorderedElementsAre("input_2", "test"));
}

TEST_F(ResolveTensorNamesTest, collision_on_inputs) {
    auto input_1 = std::make_shared<Parameter>(element::f32, Shape{1, 3});
    auto input_2 = std::make_shared<Parameter>(element::f32, Shape{1, 3});
    auto result = std::make_shared<Result>(input_1);
    auto result_2 = std::make_shared<Result>(input_2);

    input_1->output(0).set_names({"input_2", "name", "test", "input:0", "input:1"});
    input_2->output(0).set_names({"input_2", "test", "input:1", "input:2"});

    auto model = std::make_shared<Model>(ResultVector{result_2, result}, ParameterVector{input_1, input_2});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ResolveNameCollisions>();
    pass_manager.run_passes(model);

    EXPECT_THAT(input_1->output(0).get_names(), UnorderedElementsAre("input_2", "name", "test", "input:0", "input:1"));
    EXPECT_THAT(input_2->output(0).get_names(), UnorderedElementsAre("input_2_1", "test_1", "input_1:1", "input:2"));
    EXPECT_THAT(result_2->output(0).get_names(), UnorderedElementsAre("input_2_1", "test_1", "input_1:1", "input:2"));
}

TEST_F(ResolveTensorNamesTest, collision_on_outputsinputs) {
    auto input_1 = std::make_shared<Parameter>(element::f32, Shape{1, 3});
    auto input_2 = std::make_shared<Parameter>(element::f32, Shape{1, 3});
    auto result_1 = std::make_shared<Result>(input_1);
    auto result_2 = std::make_shared<Result>(input_2);

    input_1->output(0).set_names({"input_1", "input:0"});
    input_2->output(0).set_names({"input_2", "input:2"});

    auto model = std::make_shared<Model>(ResultVector{result_2, result_1}, ParameterVector{input_2, input_1});
    result_1->output(0).set_names({"result_1", "input_1"});
    result_2->output(0).set_names({"result_1"});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ResolveNameCollisions>();
    pass_manager.run_passes(model);
    model->validate_nodes_and_infer_types();

    EXPECT_THAT(input_1->output(0).get_names(), UnorderedElementsAre("input_1", "input:0", "result_1_1"));
    EXPECT_THAT(input_2->output(0).get_names(), UnorderedElementsAre("input_2", "input:2", "result_1"));
    EXPECT_THAT(result_1->output(0).get_names(), UnorderedElementsAre("result_1_1", "input_1"));
    EXPECT_THAT(result_2->output(0).get_names(), UnorderedElementsAre("result_1"));
}

TEST(ResolveNameCollisionsTest, FixTensorNamesMultiSubgraphOp) {
    // external params
    auto X = std::make_shared<Parameter>(element::f32, Shape{4});
    auto Y = std::make_shared<Parameter>(element::f32, Shape{4});
    auto Z = std::make_shared<Parameter>(element::f32, Shape{8});

    auto axis = std::make_shared<Constant>(element::i32, Shape{}, 0);
    auto external_split = std::make_shared<Split>(X, axis, 2);
    external_split->output(0).set_names({"split_1"});

    // internal params
    auto Xt = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    auto Yt = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    auto Ze = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    // then body
    auto cond = std::make_shared<Constant>(element::boolean, Shape{1}, true);
    auto axis_then = std::make_shared<Constant>(element::i32, Shape{}, 0);
    auto split_y = std::make_shared<Split>(Yt, axis_then, 2);
    split_y->output(0).set_names({"split:0"});
    auto then_op = std::make_shared<Subtract>(Xt, split_y->output(0));
    auto res0 = std::make_shared<Result>(then_op);

    // else body
    auto axis_else = std::make_shared<Constant>(element::i32, Shape{}, 0);
    auto split_z = std::make_shared<Split>(Ze, axis_else, 4);
    split_z->output(0).set_names({"split:0"});
    auto else_op = std::make_shared<Relu>(split_z);
    auto res1 = std::make_shared<Result>(else_op);

    // If set up
    auto then_body = std::make_shared<ov::Model>(OutputVector{res0}, ParameterVector{Yt, Xt}, "then_body");
    auto else_body = std::make_shared<ov::Model>(OutputVector{res1}, ParameterVector{Ze}, "else_body");
    auto if_op = std::make_shared<op::v8::If>(cond);
    if_op->set_then_body(then_body);
    if_op->set_else_body(else_body);
    if_op->set_input(external_split->output(0), Xt, nullptr);
    if_op->set_input(Y, Yt, nullptr);
    if_op->set_input(Z, nullptr, Ze);
    auto result = if_op->set_output(res0, res1);

    auto res = std::make_shared<Result>(result);
    auto model = std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Y, Z});

    EXPECT_THAT(external_split->output(0).get_names(), UnorderedElementsAre("split_1"));
    EXPECT_THAT(split_y->output(0).get_names(), UnorderedElementsAre("split:0"));
    EXPECT_THAT(split_z->output(0).get_names(), UnorderedElementsAre("split:0"));

    ov::pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::ResolveNameCollisions>();
    pass_manager.run_passes(model);

    EXPECT_THAT(external_split->output(0).get_names(), UnorderedElementsAre("split_1"));
    EXPECT_THAT(split_y->output(0).get_names(), UnorderedElementsAre("split:0"));
    EXPECT_THAT(split_z->output(0).get_names(), UnorderedElementsAre("split_1:0"));
}

}  // namespace ov::test
