import torch
from torch.fx import Interpreter
import torch._dynamo as dynamo
from Opara import OperatorLauncher
from Opara import StreamAllocator
from torch._functorch.partitioners import draw_graph
import os
path = os.path.abspath(os.path.dirname(__file__))
output_file_path = path + '/profile_result/output.txt'
output_file = open(output_file_path, "w")

class Scheduler(Interpreter):
    def run_node(self, n):
        """
        Run a specific node ``n`` and return the result.
        Calls into placeholder, get_attr, call_function,
        call_method, call_module, or output depending
        on ``node.op``

        Args:
            n (Node): The Node to execute

        Returns:
            Any: The result of executing ``n``
        """
        # print(n)
        for event in n.event_to_wait:
            # print(n.name, n.stream)
            n.stream.wait_event(event)
        torch.cuda.set_stream(stream=n.stream)

        args, kwargs = self.fetch_args_kwargs_from_env(n)
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)
        
        self.env[n] = getattr(self, n.op)(n.target, args, kwargs)
        
        # n.event.record(n.stream)

        is_record = False
        for user in n.users:
            if n.stream != user.stream:
                if is_record is False:
                    n.event.record(n.stream)
                    is_record = True
        return self.env[n]
    
    def run(self, *args):
        """
        Run `module` via interpretation and return the result.

        Args:
            *args: The arguments to the Module to run, in positional order
            initial_env (Optional[Dict[Node, Any]]): An optional starting environment for execution.
                This is a dict mapping `Node` to any value. This can be used, for example, to
                pre-populate results for certain `Nodes` so as to do only partial evaluation within
                the interpreter.
            enable_io_processing (bool): If true, we process the inputs and outputs with graph's process_inputs and
                process_outputs function first before using them.

        Returns:
            Any: The value returned from executing the Module
        """
        self.env = {}
        self.args_iter = iter(args)
        # Positional function args are consumed left-to-right byp
        # `placeholder` nodes. Use an iterator to keep track of
        # position and extract those values.
        # print("run_node->len(graph.nodes):", len(self.module.graph.nodes))
        for node in self.module.graph.nodes:
            # print("run_node->node:", node)
            self.env[node] = self.run_node(node)

            if node.op == 'output':
                output_val = self.env[node]
                return output_val
            


def capturer(inputs, model, copy_outputs: bool = False):
    assert isinstance(inputs, (list, tuple)), f"inputs is of type {type(inputs)} instead of list"
    static_inputs = [torch.zeros_like(x, device='cuda') for x in inputs]

    dynamo.reset()
    explanation, out_guards, graphs, ops_per_graph, break_reasons, explanation_verbose = dynamo.explain(model, *inputs)
    fx_module = graphs[0]
    # print(fx_module.graph, file=output_file)
    fx_module.cuda()
    model_class_name = model.__class__.__name__
    OperatorLauncher.recompile(model_class_name, fx_module, inputs)

    all_streams, all_events = StreamAllocator.assign_stream(fx_module.graph)

    all_events = [torch.cuda.Event() for _ in range(len(all_streams))]
    first_stream = all_streams[0]
    first_event = all_events[0]
    interpreter = Scheduler(fx_module)

    # with torch.autocast(device_type='cuda', dtype=torch.float16):

    with torch.no_grad():
        for i in range(3):
            interpreter.run(*inputs)
    with torch.no_grad():
        # capture
        g = torch.cuda.CUDAGraph()

        with torch.cuda.graph(g, stream=first_stream):
            first_event.record(first_stream)

            for i, stream in enumerate(all_streams):
                if i > 0:
                    stream.wait_event(first_event)
            
            static_outputs = interpreter.run(*static_inputs)
            
            torch.cuda.set_stream(first_stream)
            for i, event in enumerate(all_events):
                if i > 0:
                    event.record(all_streams[i])
            for i, event in enumerate(all_events):
                if i > 0:
                    first_stream.wait_event(event)

        torch.cuda.synchronize()

        if not isinstance(static_outputs, (list, tuple)):
            static_outputs = (static_outputs,)

    def run(*new_inputs):
        assert isinstance(new_inputs, (list, tuple)), f"inputs is of type {type(new_inputs)} instead of list"
        assert len(static_inputs) == len(new_inputs), f"{len(static_inputs)} == {len(new_inputs)}"
        for dst, src in zip(static_inputs, new_inputs):
            dst.copy_(src)  # cuda graph can only read data from the same address
        g.replay()
        if copy_outputs:
            return [x.clone() for x in static_outputs]
        else:
            return static_outputs

    return run
