from torch.cuda.streams import Stream, Event
import os
path = os.path.abspath(os.path.dirname(__file__))
output_file_path = path + '/profile_result/output.txt'
output_file = open(output_file_path, "w")

def assign_stream(graph):
    for node in graph.nodes:
        setattr(node, 'stream', None)
        setattr(node, 'event', None)
        setattr(node, 'event_to_wait', [])

    streams, events = Opara(graph)
    
    for node in graph.nodes:
        for input_node in node.all_input_nodes:
            if node.stream != input_node.stream:
                if input_node.event not in node.event_to_wait:
                    node.event_to_wait.append(input_node.event)
    return streams, events


def Opara(graph):
    streams, events = [], []
    for node in graph.nodes:
        node.event = Event()
        events.append(node.event)
        for input_node in node.all_input_nodes:
            if node == list(input_node.users.keys())[0]:
                node.stream = input_node.stream
                break
        if node.stream is None:
            node.stream = Stream()
            streams.append(node.stream)
    return streams, events

from torch.cuda.streams import Stream, Event
import os

graph = [[]]
bigraph = [[]]

def _convert_graph_to_adjacency_list(graph):
    node_to_id = {}
    for node_id, node in enumerate(graph.nodes):
        node_to_id[node] = node_id

    adjacency_list = [[] for _ in range(len(graph.nodes))]
    for node in graph.nodes:
        for user in node.users:
            adjacency_list[node_to_id[node]].append(node_to_id[user])
    return adjacency_list

def dfs(start, adjacency_list, visited):
    visited[start] = True
    for node in adjacency_list[start]:
        if not visited[node]:
            dfs(node, adjacency_list, visited)
            
def get_transitive_closure(adjacency_list):
    transitive_closure = []
    num_nodes = len(adjacency_list)
    for i in range(num_nodes):
        reachable = [False] * num_nodes
        dfs(i, adjacency_list, reachable)
        reachable[i] = False
        transitive_closure.append(reachable)
    return transitive_closure

def get_MEG(adjacency_list):
    transitive_closure = get_transitive_closure(adjacency_list)
    meg = adjacency_list
    for i in range(len(adjacency_list)):
        meg_child_nodes = meg[i]
        child_nodes = adjacency_list[i]
        for child in child_nodes:
            if child not in meg_child_nodes:
                continue
            for another_child in child_nodes:
                if transitive_closure[child][another_child] and another_child in meg_child_nodes:
                    meg_child_nodes.remove(another_child)
    return meg

def meg_to_bigraph(meg):
    bigraph = []
    for i in range(len(meg)):
        adjacency = [False] * len(meg)
        for child in meg[i]:
            adjacency[child] = True
        bigraph.append(adjacency)
    return bigraph

def dag_to_bigraph(adjacency_list):
    closure = []
    num_vertices = len(adjacency_list)
    for i in range(num_vertices):
        reachable = [False] * num_vertices
        dfs(i, adjacency_list, reachable)
        reachable[i] = False
        closure.append(reachable)
    return closure

def find_matching(start, adjacency_list, seen, match_status):
    num_b = len(adjacency_list[0])
    for i in range(num_b):
        if adjacency_list[start][i] and not seen[i]:
            seen[i] = True
            curr_match = match_status[i]
            if match_status[i] == -1 or find_matching(curr_match, adjacency_list, seen, match_status):
                match_status[i] = start
                return True  
    return False

def maximum_matching(adjacency_list):
    num_b = len(adjacency_list[0])
    match_result = [-1] * num_b
    num_a = len(adjacency_list)
    for i in range(num_a):
        seen = [False] * num_b
        find_matching(i, adjacency_list, seen, match_result)
    return match_result


def get_mapping(matching):
    num_nodes = len(matching)
    chains = []
    for i in range(num_nodes):
        if i not in matching:
            chains.append([i, i])
    group_num = 0
    mapping = [-1] * num_nodes
    for chain in chains:
        group_id = group_num
        group_num += 1
        curr = chain[1]
        while True:
            mapping[curr] = group_id
            if matching[curr] == -1:
                chain[0] = curr
                break
            else:
                curr = matching[curr]
    # print("chains:", chains, file=output_file)
    return (mapping, chains, group_num)        

def build_stream_dag(nn_dag, stream_chains):
    transitive_closure = get_transitive_closure(nn_dag)
    stream_dag = []
    for i in range(len(stream_chains)):
        ensuing_streams = []
        chain_end = stream_chains[i][1]
        for j in range(len(stream_chains)):
            chain_begin = stream_chains[j][0]
            if transitive_closure[chain_end][chain_begin]:
                ensuing_streams.append(j)
        stream_dag.append(ensuing_streams)
    return stream_dag





    
