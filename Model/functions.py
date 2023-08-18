#imports
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, LineString, MultiLineString
import networkx as nx
from shapely import contains
from Model.network import *
import pickle
import copy

'''
d(A, B): This function calculates the Euclidean distance between two points A and B.

get_global_nx(graph, honeycomb): This function constructs a networkx graph from the given graph, setting the positions of nodes based on the honeycomb lattice.

gcc(graph): This function computes the largest connected component of the given graph.

change_real_c(g, c): This function changes the weight of edges in the graph that connect a node of type integer with a node of type tuple to a given value c.

eval_real_sol_center(g, center, population_density, N): This function computes a measure of centrality for a given graph, considering population density.

set_L_to(g, L): This function sets the length of the sequence of edges in the graph to a specified length L.

set_L(g, L): This function sets the length of the sequence of edges in the graph to a specified length L, adjusting the edge weights accordingly.

greedy_multi_wdata(graphs, S, tau, eta=0.1): This function performs a greedy optimization on a list of graphs, adjusting the length of their edge sequences and their edge weights.

change_c(graph, tau, compute_distance=True): This function changes the weight of local edges in the graph and optionally re-computes the shortest distance from each node to the center.

eval_greedy_sol_center(graph, c): This function changes the weight of local edges in the graph, re-computes the shortest distance from each node to the center, and evaluates the centrality measure.

change_eta(graph, eta, compute_distance=True): This function changes the weight of the edges in the graph's sequence and optionally re-computes the shortest distance from each node to the center.

change_real_local_speed(g, hc, new_speed, city='Atlanta'): This function changes the weight of the edges in the graph based on the new speed parameter and the city.

load_system_data(name='Boston'): This function loads the system data for a specified city from a pickle file.

'''

def d(A,B):
    return np.linalg.norm([A[0]-B[0],A[1]-B[1]])
    
def get_global_nx(graph,honeycomb):
    L=honeycomb.L
    g=nx.Graph([i[0] for i in graph.sequence])
    nx.set_node_attributes(g, {i:honeycomb.sites[i-L**2] for i in g.nodes()}, "pos")
    return g

def gcc(graph):
    components = nx.connected_components(graph)
    largest_component = max(components, key=len)
    largest_component_graph = graph.subgraph(largest_component)
    out = largest_component_graph.copy()
    return out


def change_real_c(g,c):
    for edge in g.edges:
        if (type(edge[0])==int and type(edge[1])==tuple) or (type(edge[1])==int and type(edge[0])==tuple):
            g[edge[0]][edge[1]]['weight']=c



def eval_real_sol_center(g,center,population_density,N):
    spl=nx.shortest_path_length(g,target=center,weight='weight')
    return sum([spl[i]*population_density[i+N] for i in g.nodes() if type(i)==int])/(sum(list(population_density.values())))


def set_L_to(g,L):
    for i in g._sequence[L:]:
        g.edges[i[0][0]][i[0][1]],g.edges[i[0][1]][i[0][0]]=1e10,1e10
    g.sequence=g._sequence[:L]
    g.distance=compute_distance_to_center (g.center, g.edges)
    
def set_L(g,L):
    if len(g.sequence)>L:
        set_L_to(g,L)
    elif len(g.sequence)<L:    
        for i in g._sequence[len(g.sequence):L]:
            g.edges[i[0][0]][i[0][1]],g.edges[i[0][1]][i[0][0]]=g.eta,g.eta
    g.sequence=g._sequence[:L]

def greedy_multi_wdata(graphs,S,tau,eta=0.1):
    #make graph
    sols=[0]*(len(graphs))
    for n,graph in enumerate(graphs):
        graph.eta=eta
        set_L(graph,S)
        change_c(graph,tau,compute_distance=True)
        sols[n]=np.mean([graph.weights[graph.N+i]*graph.distance[i] for i in graph.edges if graph.layer[i]=='local'])
    best_id=np.argmin(sols)
    # print(sols)
    out_g=graphs[best_id]
    center_degree=0
    for e in out_g.sequence:
        i = e[0][0]
        j = e[0][1]
        n = i - out_g.N
        m = j - out_g.N
        if n == out_g.center or m == out_g.center:
            center_degree += 1
    return center_degree,sols[best_id],best_id

def change_c(graph,tau,compute_distance=True):
    for node in graph.edges:
        if graph.layer[node]=='local':
            graph.edges[node][node+graph.N]=tau
            graph.edges[node+graph.N][node]=tau  
    if compute_distance:
        graph.distance=compute_distance_to_center(graph.center,graph.edges)



def eval_greedy_sol_center(graph,c):
    change_c(graph,c,compute_distance=True)
    wd=[graph.distance[i]*graph.weights[i+graph.N] for i in graph.distance if i <graph.N]
    return sum(wd)/sum(list(graph.weights.values()))

def change_eta(graph,eta,compute_distance=True):
    graph.eta=eta
    for node in graph.sequence:
        graph.edges[node[0][0]][node[0][1]]=eta
        graph.edges[node[0][1]][node[0][0]]=eta  
    if compute_distance:
        graph.distance=compute_distance_to_center(graph.center,graph.edges)

def change_real_local_speed(g,hc,new_speed,city='Atlanta'):
    for edge in g.edges:
        if type(edge[1])==int and type(edge[0])==int:
            if city=='Atlanta':
                g[edge[0]][edge[1]]['weight']=0.3048*d(hc.sites[edge[0]],hc.sites[edge[1]])/new_speed*60
            elif city=='Boston' or city=='Toronto':
                g[edge[0]][edge[1]]['weight']=d(hc.sites[edge[0]],hc.sites[edge[1]])/new_speed*60

            
def load_system_data(name='Boston'):
    s='Data/'+name+'.pickle'
    out_dict = pickle.load(open(s, "rb"))
    return out_dict
