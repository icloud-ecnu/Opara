from functools import reduce
from operator import mul
from Opara import ModelProfiler

import os
path = os.path.abspath(os.path.dirname(__file__))
output_file_path = path + '/profile_result/output.txt'
output_file = open(output_file_path, "w")


def launch(nodes, result, in_degree, sharedMemPerBlock, regsPerBlock, maxThreadsPerBlock):
    def pop_from_queue(q):
        ret_node_name = q[0]

        min_metric = 2
        for node_name in q:
            if len(nodes[node_name].info) > 0:
                achieved_occupancy = nodes[node_name].info[0]["args"]["est. achieved occupancy %"]
                blocksPerSM = nodes[node_name].info[0]["args"]["blocks per SM"]
                shared_memory = nodes[node_name].info[0]["args"]["shared memory"] / sharedMemPerBlock
                thread_num = reduce(mul, nodes[node_name].info[0]["args"]["block"]) / maxThreadsPerBlock
                registers_num = thread_num * nodes[node_name].info[0]["args"]["registers per thread"] / regsPerBlock
                request = [shared_memory, thread_num, registers_num]
                metric = shared_memory
                if metric < min_metric:
                    min_metric = metric
                    ret_node_name = node_name
        q.remove(ret_node_name)

        return ret_node_name
    
    q = []
    for node_name, degree in in_degree.items():
        if degree == 0:
            q.append(node_name)
    while q:
        cur_node_name = pop_from_queue(q)
        result.append(cur_node_name)

        for succ_node in nodes.get(cur_node_name).users:
            in_degree[succ_node.name] -= 1
            if in_degree[succ_node.name] == 0:
                q.append(succ_node.name)
    return result


import json
import os

def get_resource_from_json(path):
    with open(path) as f:
        data = json.load(f)

    step_num = 0
    for event in data["traceEvents"]:
        if "run" in event["name"] and "run_node" not in event["name"]:
            step_num += 1
    # print("step_num", step_num)
   
    # 获取run_node事件、kernel_launch事件、kernel事件
    run_node_events = []
    kernel_launch_events = []
    kernel_events = []
    for event in data["traceEvents"]:
        if "run_node" in event["name"]:
            run_node_events.append(event)

        if event["name"] == "cudaLaunchKernel":
            kernel_launch_events.append(event)

        if event.get("cat", "None") == "kernel":
            kernel_events.append(event)


    # 计算获取一个step中的run_node事件、kernel_launch事件、kernel事件
    one_step_range_of_node = len(run_node_events) // step_num
    one_step_range_of_kernel_launch = len(kernel_launch_events) // step_num
    one_step_range_of_kernel = len(kernel_events) // step_num
    start = step_num - 1
    end = step_num
    run_node_events = run_node_events[start*one_step_range_of_node:end*one_step_range_of_node]
    kernel_launch_events = kernel_launch_events[start*one_step_range_of_kernel_launch:end*one_step_range_of_kernel_launch]
    kernel_events = kernel_events[start*one_step_range_of_kernel:end*one_step_range_of_kernel]


    # 根据时间轴范围获取由node事件触发的kernel_launch事件
    node2kernels = []
    kernel_num = 0
    for i, node_event in enumerate(run_node_events):
        node2kernels.append([])
        for j, kernel_launch_event in enumerate(kernel_launch_events):
            if node_event["ts"] <= kernel_launch_event["ts"] and node_event["ts"] + node_event["dur"] >= kernel_launch_event["ts"]:
                node2kernels[i].append(kernel_events[j])
                kernel_num += 1

    # print("kernel_num", kernel_num)

    max_block_nums = []
    sum_time = 0
    for i, kernel_events in enumerate(node2kernels):
        
        max_block_size = 4096
        for kernel_event in kernel_events:
            sum_time += kernel_event["dur"]
            cur_block_size = kernel_event["args"]["block"][0] * kernel_event["args"]["block"][1] * kernel_event["args"]["block"][2]
            max_block_size = min(max_block_size, cur_block_size)
        if max_block_size == 4096:
            max_block_size = 0
        max_block_nums.append(max_block_size)

    est_achieved_occupancy = 0
    for i, kernel_events in enumerate(node2kernels):
        for kernel_event in kernel_events:
            dur = kernel_event["dur"]
            est_achieved_occupancy += kernel_event["args"]["est. achieved occupancy %"] * dur
    est_achieved_occupancy = est_achieved_occupancy / sum_time
    sharedMemPerBlock = data['deviceProperties'][0]['sharedMemPerBlock']
    regsPerBlock = data['deviceProperties'][0]['regsPerBlock']
    maxThreadsPerBlock = data['deviceProperties'][0]['maxThreadsPerBlock']


    return node2kernels, sharedMemPerBlock, regsPerBlock, maxThreadsPerBlock


def get_topo(fx_nodes, sharedMemPerBlock, regsPerBlock, maxThreadsPerBlock):
    nodes = {node.name: node for node in fx_nodes}
    in_degree = {node.name: 0 for node in nodes.values()}
    for node in nodes.values():
        for input_node in node.all_input_nodes:
            in_degree[node.name] += 1
    visited = set()
    result = []
    # print("fx_nodes", nodes.keys(), file=output_file)
    result = launch(nodes, result, in_degree, sharedMemPerBlock, regsPerBlock, maxThreadsPerBlock)

    return result, nodes

def recompile(model_class_name, graph_module, inputs):
    
    path = os.path.abspath(os.path.dirname(__file__))
    # model_class_name = graph_module.__class__.__name__
    for i in inputs:
        model_class_name += "_" + str(i.shape)
    path += "/profile_result/" + model_class_name + ".pt.trace.json"
    if os.path.exists(path) is False:
        ModelProfiler.profile(graph_module, inputs, path)
    node2kernels, sharedMemPerBlock, regsPerBlock, maxThreadsPerBlock = get_resource_from_json(path)

    for i, node in enumerate(graph_module.graph.nodes):
        if not hasattr(node, 'info'):
            setattr(node, 'info', node2kernels[i])

    pre = None
    result, torch_nodes = get_topo(graph_module.graph.nodes, sharedMemPerBlock, regsPerBlock, maxThreadsPerBlock)

    for name in result:
        if pre == None:
            pre = torch_nodes[name]
        else:
            pre._next = torch_nodes[name]
            torch_nodes[name]._prev = pre
            pre = torch_nodes[name]
    # print(graph_module.graph, file=output_file)
    graph_module.graph.lint()
    graph_module.recompile()
