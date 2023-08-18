from Model.functions import *
from Model.network import *

def eval_greedy_sol_center(graph):
    """
    Computes the mean distance to the center of the graph.

    Args:
        graph: Graph object containing information about nodes, edges, and distances.

    Returns:
        float: Mean distance to the center of the graph.
    """
    wd=[graph.distance[i] for i in graph.distance if i <graph.N]
    return np.mean(wd)

def global_degree(graph, node):
    """
    Computes the degree of a specified node in the global layer of the graph.

    Args:
        graph: Graph object containing information about nodes and edges.
        node: The specific node for which the degree is calculated.

    Returns:
        int: Degree of the specified node in the global layer.
    """
    return len([ngb for ngb in graph.edges[node] if graph.layer[ngb]=='global' and graph.edges[node][ngb]<1e8])


def remove_leaf(graph, global_edges, global_nodes, global_degrees):
    """
    Identifies a leaf edge in the global layer of the graph and returns it.

    Args:
        graph: Graph object containing information about nodes and edges.
        global_edges: List of global edges.
        global_nodes: List of global nodes.
        global_degrees: Dictionary mapping nodes to their degrees in the global layer.

    Returns:
        edge: The edge identified as a leaf edge.
    """
    leaf_edges=[]
    for e in global_edges:
        if (global_degrees[e[0]]==1 or global_degrees[e[1]]==1) :
            if (e[0]!=15050 and e[1]!=15050) or global_degrees[15050]>1:
                leaf_edges.append(e)
                
    rem_edge=random.choice(leaf_edges)
    return rem_edge

def removal_update(graph, global_edges, global_nodes, rem_edge, global_degrees):
    """
    Updates the graph based on the removal of a specified edge.
    Modifies the global degrees, nodes, and edges accordingly.

    Args:
        graph: Graph object containing information about nodes and edges.
        global_edges: List of global edges.
        global_nodes: List of global nodes.
        rem_edge: The edge to be removed.
        global_degrees: Dictionary mapping nodes to their degrees in the global layer.
    """
    graph.edges[rem_edge[0]][rem_edge[1]],graph.edges[rem_edge[1]][rem_edge[0]]=1e10,1e10
    global_degrees[rem_edge[0]]-=1
    global_degrees[rem_edge[1]]-=1

    if global_degrees[rem_edge[0]]==0:
        global_nodes.remove(rem_edge[0])
    else:
        global_nodes.remove(rem_edge[1])
    global_edges.remove(rem_edge)

def add_leaf(graph, global_edges, global_nodes, global_degrees, eta=0.1):
    """
    Identifies a new leaf edge in the global layer of the graph and returns it.

    Args:
        graph: Graph object containing information about nodes and edges.
        global_edges: List of global edges.
        global_nodes: List of global nodes.
        global_degrees: Dictionary mapping nodes to their degrees in the global layer.
        eta: Optional; Edge weight for the new leaf edge.

    Returns:
        edge: The edge identified as a new leaf edge.
    """
    leaf_edges=[[n,ngb] for n in global_nodes for ngb in graph.edges[n] if graph.layer[ngb]=='global' if global_degrees[ngb]==0]
    new_edge=random.choice(leaf_edges)
    return new_edge





def addition_update(graph, global_edges, global_nodes, new_edge, global_degrees, eta=0.1):
    """
    Updates the graph based on the addition of a specified edge.
    Modifies the global degrees, nodes, and edges accordingly.

    Args:
        graph: Graph object containing information about nodes and edges.
        global_edges: List of global edges.
        global_nodes: List of global nodes.
        new_edge: The edge to be added.
        global_degrees: Dictionary mapping nodes to their degrees in the global layer.
        eta: Optional; Edge weight for the new leaf edge.
    """
    graph.edges[new_edge[0]][new_edge[1]],graph.edges[new_edge[1]][new_edge[0]]=eta,eta
    global_degrees[new_edge[0]]+=1
    global_degrees[new_edge[1]]+=1
    if global_degrees[new_edge[1]]==1:
        global_nodes.append(new_edge[1])
    else:
        global_nodes.append(new_edge[0])
    global_edges.append(new_edge)

def move(graph, global_edges, global_nodes, global_degrees, eta=0.1):
    """
    Performs a move in the simulated annealing process by removing a leaf edge and adding a new one.

    Args:
        graph: Graph object containing information about nodes and edges.
        global_edges: List of global edges.
        global_nodes: List of global nodes.
        global_degrees: Dictionary mapping nodes to their degrees in the global layer.
        eta: Optional; Edge weight for the new leaf edge.

    Returns:
        tuple: The removed and added edges.
    """
    rem_edge=remove_leaf(graph,global_edges,global_nodes,global_degrees)
    removal_update(graph,global_edges,global_nodes,rem_edge,global_degrees)
    add_edge=add_leaf(graph,global_edges,global_nodes,global_degrees,eta=eta)
    addition_update(graph,global_edges,global_nodes,add_edge,global_degrees,eta=eta)
    return rem_edge,add_edge



def unmove(graph, global_edges, global_nodes, added_edge, removed_edge, global_degrees, eta=0.1):
    """
    Reverses a move made in the simulated annealing process.

    Args:
        graph: Graph object containing information about nodes and edges.
        global_edges: List of global edges.
        global_nodes: List of global nodes.
        added_edge: The edge that was added during the move.
        removed_edge: The edge that was removed during the move.
        global_degrees: Dictionary mapping nodes to their degrees in the global layer.
        eta: Optional; Edge weight for the new leaf edge.
    """
    removal_update(graph,global_edges,global_nodes,added_edge,global_degrees)
    addition_update(graph,global_edges,global_nodes,removed_edge,global_degrees,eta=eta)

def SA(graph, global_nodes, global_edges, initialTemp=10, finalTemp=1e-3, iterationPerTemp=1000, alpha=0.99):
    """
    Simulated annealing algorithm to optimize the graph structure.

    Args:
        graph: Graph object containing information about nodes and edges.
        global_nodes: List of global nodes.
        global_edges: List of global edges.
        initialTemp: Optional; Initial temperature for the annealing process.
        finalTemp: Optional; Final temperature for the annealing process.
        iterationPerTemp: Optional; Number of iterations performed at each temperature step.
        alpha: Optional; Cooling rate.

    Returns:
        (Optional) Result of the optimization process.
    """
    spls=[]
    temp=initialTemp
    global_degrees={i:global_degree(graph,i) for i in graph.edges if graph.layer[i]=='global'}
    graph.distance=compute_distance_to_center(graph.center, graph.edges)
    cur_cost=eval_greedy_sol_center(graph)
    while temp>finalTemp:
        for _ in range(iterationPerTemp):
            r,a=move(graph,global_edges,global_nodes,global_degrees)
            store_dist=compute_distance_to_center(graph.center,graph.edges)
            graph.distance=copy.copy(store_dist)
            E2=eval_greedy_sol_center(graph)
            cost = cur_cost - E2
            # print(cost)
            # if the new solution is better, accept it
            if cost > 0:
                spls.append(E2)
                cur_cost=E2
            # if the new solution is not better, accept it with a probability of e^(-cost/temp)
            else:
                if np.random.uniform(0, 1) < np.exp(cost / temp):#np.exp(cost / self.currTemp):
                    spls.append(E2)
                    cur_cost=E2
                else:
                    # print('not accepted')
                    spls.append(cur_cost)
                    unmove(graph,global_edges,global_nodes,a,r,global_degrees)
                    graph.distance=copy.copy(store_dist)
            #print(temp)
        # decrement the temperature
        temp=temp*alpha
    return spls


def SA_sol(c=0.01,L=12,eta=0.1,initialTemp=10, finalTemp=1e-3,iterationPerTemp=1000, alpha=0.99):
    hc=Lattice(100)
    graph=Network(hc,0.001,eta,0)
    greedy_optimization(graph,L)
    change_c(graph,c)
    global_nodes=list(set([i[0][0] for i in graph.sequence]+[i[0][1] for i in graph.sequence]))
    global_edges=[i[0] for i in graph.sequence]
    global_degrees={i:global_degree(graph,i) for i in graph.edges if graph.layer[i]=='global'}
    for _ in range(50000):
        r,a=move(graph,global_edges,global_nodes,global_degrees)
    spls=SA(graph,global_nodes,global_edges,initialTemp=initialTemp, finalTemp=finalTemp,iterationPerTemp=iterationPerTemp, alpha=alpha)
    return graph,spls
