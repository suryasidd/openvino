// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <memory>
#include <queue>
#include <assert.h>

#include <ngraph/function.hpp>
#include <ngraph/pass/visualize_tree.hpp>

std::pair<bool, std::string> compare_functions(const std::shared_ptr<ngraph::Function> & f1, const std::shared_ptr<ngraph::Function> & f2) {
    /*
     * This function compares two nGraph functions and requires them to have exactly one output
     * + Check nodes types
     * + Check number of inputs
     * + Check shapes
     * - Do not check nodes attributes (requires visitor mechanism to be completed)
     */
    auto f1_results = f1->get_results();
    auto f2_results = f2->get_results();

    assert(f1_results.size() == 1);
    assert(f2_results.size() == 1);

    auto typeInfoToStr = [](const ngraph::Node::type_info_t & typeInfo) {
        return std::string(typeInfo.name) + "/" + std::to_string(typeInfo.version);
    };

    std::ostringstream err_log;

    std::queue<std::pair<std::shared_ptr<ngraph::Node>, std::shared_ptr<ngraph::Node> > > q;
    q.push({f1_results[0], f2_results[0]});
    while (!q.empty()) {
        auto node1 = q.front().first;
        auto node2 = q.front().second;
        q.pop();

        if (node1->get_type_info() != node2->get_type_info()) {
            return {false, typeInfoToStr(node1->get_type_info()) + " != " + typeInfoToStr(node2->get_type_info())};
        }

        if (node1->inputs().size() != node2->inputs().size()) {
            return {false, "Number of inputs is different: " + std::to_string(node1->inputs().size()) + " and " + std::to_string(node2->inputs().size())};
        }

        if (node1->outputs().size() != node2->outputs().size()) {
            return {false, "Number of outputs is different: " + std::to_string(node1->outputs().size()) + " and " + std::to_string(node2->outputs().size())};
        }

        for (int i = 0; i < node1->inputs().size(); ++i) {
            if (!node1->input(i).get_partial_shape().same_scheme(node2->input(i).get_partial_shape())) {
                err_log << "Different shape detected" << std::endl
                        << node1->description() << " Input(" << i << ") " << node1->input(i).get_partial_shape() << " and "
                        << node2->description() << " Input(" << i << ") " << node2->input(i).get_partial_shape() << std::endl;
            }

            q.push({node1->input_value(i).get_node_shared_ptr(), node2->input_value(i).get_node_shared_ptr()});
        }

        for (int i = 0; i < node1->outputs().size(); ++i) {
            if (!node1->output(i).get_partial_shape().same_scheme(node2->output(i).get_partial_shape())) {
                err_log << "Different shape detected" << std::endl
                        << node1->description() << " Output(" << i << ") " << node1->output(i).get_partial_shape() << " and "
                        << node2->description() << " Output(" << i << ") " << node2->output(i).get_partial_shape() << std::endl;
            }
        }
    }
    return {err_log.str().empty(), err_log.str()};
}

void check_rt_info(const std::shared_ptr<ngraph::Function> & f) {
    static const std::vector<std::string> attrs_to_check{"Variant::RuntimeAttribute::FusedNames"};

    std::ostringstream err_log;
    for (auto & op : f->get_ops()) {
        if (op->is_constant()) continue;

        const auto & rt_info = op->get_rt_info();
        for (const auto & attr_name : attrs_to_check) {
            if (!rt_info.count(attr_name)) {
                err_log << "Node: " << op->get_friendly_name() << " has no attribute: " << attr_name << std::endl;
            }
        }
    }

    auto err_msg = err_log.str();
    if (!err_msg.empty()) {
        throw ngraph::ngraph_error(err_msg);
    }
}

void visualize_function(std::shared_ptr<ngraph::Function> f, const std::string & file_name) {
    std::vector<std::shared_ptr<ngraph::Function> > g{f};
    ngraph::pass::VisualizeTree(file_name).run_on_module(g);
}