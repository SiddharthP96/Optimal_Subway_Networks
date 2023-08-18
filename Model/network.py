

import heapq
import random
from time import process_time
import numpy as np
import copy


#############
#Main Lattice Class
'''
This class creates a lattice object with parameters L and R. The L parameter is used to create a honeycomb lattice with 
N sites and bonds using the create_honeycomb_lattice function. The R parameter is used to create boundary conditions using
the create_boundary_conditions function. Then, the center_at_the_origin and transform_coordinates functions are used to 
position the lattice at the origin and transform the coordinates, respectively.
'''
#############
class Lattice:

    def __init__ (lattice, L):
        lattice.L = L
        lattice.N, lattice.center, lattice.sites, lattice.bonds = create_honeycomb_lattice (L)

        R = int(lattice.L/4)
        if R %2 == 0:
            R += 1
        lattice.sites, lattice.bonds = create_boundary_conditions (R, lattice.center, lattice.sites, lattice.bonds)

        center_at_the_origin (lattice.sites, lattice.center)
        transform_coordinates (lattice.sites)


##############
'''
This code creates a honeycomb lattice with a given length L. It starts by initializing two dictionaries, sites and 
positions, and a variable i to track the number of sites. Then, it iterates through each x and y coordinate in the range 
of 0 to L and adds the coordinates to the sites dictionary with the value of i. It also adds the coordinates and i to the 
positions dictionary. If the coordinates are the center of the lattice (x = L/2 and y = L/2), then it sets the center 
variable to i. Finally, it calls the create_honeycomb_lattice_bonds function to create the bonds for the lattice, and sets 
the N variable to the number of sites. The function then returns the N, center, sites, and bonds variables.
'''

def create_honeycomb_lattice (L):
    i = 0
    sites = {}
    positions = {}
    for x in range(0, L):
        for y in range(0, L):
            sites[i] = [x,y]
            positions[x,y] = i
            if x == int(L/2) and y == int(L/2):
                center = i
            i += 1

    bonds = create_honeycomb_lattice_bonds (sites, positions)
    N = len(sites)
                    
    
    return N, center, sites, bonds


################################
'''
create_honeycomb_lattice_bonds
This function creates a dictionary of bonds for a honeycomb lattice given a set of sites and positions.

Parameters:
sites
: A dictionary containing the sites of the honeycomb lattice.
positions
: A dictionary containing the positions of the honeycomb lattice.
Returns:
A dictionary containing the bonds of the honeycomb lattice.

Algorithm:
The function loops through each site in the 
sites dictionary and checks the positions dictionary for any sites in the immediate vicinity. If a site is found in the 
immediate vicinity, then a bond is created  between the two sites and stored in the bonds dictionary. The immediate 
vicinity is defined as the left, right, up and down directions. For sites in odd numbered rows, the upper right and lower 
right direction is also checked. For sites in even numbered rows, the upper left and lower left direction is also checked.

'''

def create_honeycomb_lattice_bonds (sites, positions):
    
    bonds = {}
    for n in sites:
        bonds[n] = {}
        ##left
        if (sites[n][0]-1, sites[n][1]) in positions:
            m = positions[sites[n][0]-1, sites[n][1]]
            if m in sites:
                bonds[n][m] = 1.0
        ##up
        if (sites[n][0], sites[n][1]+1) in positions:
            m = positions[sites[n][0], sites[n][1]+1]
            if m in sites:
                bonds[n][m] = 1.0
        ##right
        if (sites[n][0]+1, sites[n][1]) in positions:
            m = positions[sites[n][0]+1, sites[n][1]]
            if m in sites:
                bonds[n][m] = 1.0
        ##down
        if (sites[n][0], sites[n][1]-1) in positions:
            m = positions[sites[n][0], sites[n][1]-1]
            if m in sites:
                bonds[n][m] = 1.0
              
            
        ## honeycomb lattice      
        ty = sites[n][1]
        if int(ty)%2 == 1:
            ## upper right
            if (sites[n][0]+1, sites[n][1]+1) in positions:
                m = positions[sites[n][0]+1, sites[n][1]+1]
                if m in sites:
                    bonds[n][m] = 1.0
            ## lower right 
            if (sites[n][0]+1, sites[n][1]-1) in positions:
                m = positions[sites[n][0]+1, sites[n][1]-1]
                if m in sites:
                    bonds[n][m] = 1.0 
                    
        ty = sites[n][1]
        if int(ty)%2 == 0:
            ## upper left
            if (sites[n][0]-1, sites[n][1]+1) in positions:
                m = positions[sites[n][0]-1, sites[n][1]+1]
                if m in sites:
                    bonds[n][m] = 1.0
            ## lower left 
            if (sites[n][0]-1, sites[n][1]-1) in positions:
                m = positions[sites[n][0]-1, sites[n][1]-1]
                if m in sites:
                    bonds[n][m] = 1.0
        
        
    return bonds


#########################



#########################################

def create_boundary_conditions (R, center, sites, bonds):

    
    distance = {}
    distance[center] = 0
    
    ##heap##########
    heap_distance = []
    heapq.heappush(heap_distance, (distance[center], center))
    ##################
    

    
    
    ###Dijkstra part
    visited = {}
    while len(heap_distance) > 0:
        control = -1
        while control < 0:
            if len(heap_distance) <= 0:
                control = 2
            if control < 0 :
                tmp = heapq.heappop(heap_distance)
                current = tmp[1]
                dist_current = tmp[0]
                if current not in visited:
                    control = 1
        if control == 1:
            for m in bonds[current]:
                if m not in visited:
                    pot = distance[current] + 1
                    if m not in distance:
                        distance[m] = 1e10
                    if pot < distance[m] and pot <=R:
                        distance[m] = pot
                        heapq.heappush(heap_distance, (distance[m], m))
            visited[current] = 1
    ################
    ########################################

    #remove sites and bonds outside boundaries
    new_bonds, new_sites = {}, {}
    for n in distance:
        if distance[n] <= R:
            new_bonds[n] = {}
            new_sites[n] = sites[n]
            for m in bonds[n]:
                if distance[m] <= R:
                    new_bonds[n][m] = 1.0
            
        
    return new_sites, new_bonds



def transform_coordinates (sites):
    
    for n in sites:
        x = sites[n][0]
        y = sites[n][1]

        if y%2 == 1:
            x += 0.5
        y = y * np.sqrt(3.0) / 2.0
        
        sites[n][0] = x
        sites[n][1] = y


def center_at_the_origin (sites, center):
    
    xc = sites[center][0]
    yc = sites[center][1]
    
    for n in sites:
        sites[n][0] -= xc
        sites[n][1] -= yc
######
        
#######################
#######################  















class Network:

    def __init__ (network, lattice, tau, eta, pop_exp):
        network.N = lattice.N
        network.eta = eta
        network.tau = tau
        network.center = lattice.center
        
        ##to avoid the formation of loops
        network.presence_global_layer = {}

        ##create network edges
        network.edges, network.layer = create_local_and_global_layer (lattice.bonds, network.N, tau)
        ##initialize distance to the center
        network.distance = compute_distance_to_center (network.center, network.edges)
        network.weights={i:np.exp(-pop_exp*np.linalg.norm(lattice.sites[i-network.N])) for i in network.edges if network.layer[i]=='global'}
        
        network.heap_score_edges = []
        network.score_edges = {}
        initialize_score_edges (network)

        network.sequence = []
        network._sequence = []

        network.optimization_time = 0.0

        ##order parameters
        network.average_distance = 0.0
        network.center_degree = 0
        


################
def create_local_and_global_layer (bonds, N, tau):
    
    edges = {}
    for n in bonds:
        edges[n] = {}
        edges[n+N] = {}
        for m in bonds[n]:
            edges[n][m] = 1.0
            edges[n+N][m+N] = 1e10
    for n in bonds:
        edges[n][n+N] = tau
        edges[n+N][n] = tau


    layer = {}
    for n in bonds:
        layer[n] = 'local'
        layer[n+N] = 'global'

    return edges, layer


###########
def compute_distance_to_center (center, edges):
    
    distance = {}
    distance[center] = 0
    
    ##heap##########
    heap_distance = []
    heapq.heappush(heap_distance, (distance[center], center))
    ##################
    
    ###Dijkstra part
    visited = {}
    while len(heap_distance) > 0:
        control = -1
        while control < 0:
            if len(heap_distance) <= 0:
                control = 2
            if control < 0 :
                tmp = heapq.heappop(heap_distance)
                current = tmp[1]
                dist_current = tmp[0]
                if current not in visited:
                    control = 1
        if control == 1:
            #for m in edges[current]:
            #to randomize order of neighbors
            neigh = list(edges[current].keys())
            random.shuffle(neigh)
            for i in range(0, len(neigh)):
                m = neigh[i]
                if edges[current][m] >= 0:
                    if m not in visited:
                        pot = distance[current] + edges[current][m]
                        if m not in distance:
                            distance[m] = 1e10
                        if pot <= distance[m]:
                            distance[m] = pot
                            heapq.heappush(heap_distance, (distance[m], m))
            visited[current] = 1
    ################
    ########################################

    return distance
############################################





###########################################

def evaluate_score_potential_edge (selected, network):

    ##to avoid the formation of loops
    if selected[0] in network.presence_global_layer and selected[1] in network.presence_global_layer:
        return 0.0
    ###########

    
    if network.distance[selected[0]] == network.distance[selected[1]]:
        return 0.0
    S = selected[0]
    if network.distance[selected[0]] < network.distance[selected[1]]:
        S = selected[1]
    
        
    ##modify network edge
    old_value = network.edges[selected[0]][selected[1]]
    network.edges[selected[0]][selected[1]] = network.eta
    network.edges[selected[1]][selected[0]] = network.eta


    variation = 0.0
    ###
    reset = {}
    reset[S] = network.distance[S]

    ##update distance of node S
    for n in network.edges[S]:
        pot = network.distance[n] + network.edges[S][n]
        if pot <= network.distance[S]:
            network.distance[S] = pot

    

            
    heap_distance = []
    heapq.heappush(heap_distance, (network.distance[S], S))
    ##################
    
    ###Dijkstra part
    visited = {}
    while len(heap_distance) > 0:
        control = -1
        while control < 0:
            if len(heap_distance) <= 0:
                control = 2
            if control < 0 :
                tmp = heapq.heappop(heap_distance)
                current = tmp[1]
                dist_current = tmp[0]
                if current not in visited:
                    control = 1
        if control == 1:
            #for m in network.edges[current]:
            #to randomize order of neighbors
            neigh = list(network.edges[current].keys())
            random.shuffle(neigh)
            for i in range(0, len(neigh)):
                m = neigh[i]
                if network.edges[current][m] >= 0:
                    if m not in visited:
                        pot = network.distance[current] + network.edges[current][m]
                        if pot <= network.distance[m]:
                            if m not in reset:
                                reset[m] = network.distance[m]
                            if network.layer[m] == 'local':
                                variation += network.weights[m+network.N]*(network.distance[m] - pot)#change 1
                                #print (m, network.distance[m], pot, network.distance[m] - pot)
                            network.distance[m] = pot
                            heapq.heappush(heap_distance, (network.distance[m], m))
            visited[current] = 1
    ################
    ########################################


    ###


    ##change network edge to its original value
    network.edges[selected[0]][selected[1]] = old_value
    network.edges[selected[1]][selected[0]] = old_value

    ##reset distances
    for n in reset:
        network.distance[n] = reset[n]


    return variation
##########################################

def implement_change (selected, network):

    ##to avoid the formation of loops
    network.presence_global_layer[selected[0]] = 1
    network.presence_global_layer[selected[1]] = 1
    ###########     
    
   
    S = selected[0]
    if network.distance[selected[0]] < network.distance[selected[1]]:
        S = selected[1]
    
        
    ##modify network edge
    network.edges[selected[0]][selected[1]] = network.eta
    network.edges[selected[1]][selected[0]] = network.eta
    


    variation = 0.0
    ###

    ##update distance of node S
    for n in network.edges[S]:
        pot = network.distance[n] + network.edges[S][n]
        if pot <= network.distance[S]:
            network.distance[S] = pot

    heap_distance = []
    heapq.heappush(heap_distance, (network.distance[S], S))
    ##################
    
    ###Dijkstra part
    visited = {}
    while len(heap_distance) > 0:
        control = -1
        while control < 0:
            if len(heap_distance) <= 0:
                control = 2
            if control < 0 :
                tmp = heapq.heappop(heap_distance)
                current = tmp[1]
                dist_current = tmp[0]
                if current not in visited:
                    control = 1
        if control == 1:
            #for m in network.edges[current]:
            #to randomize order of neighbors
            neigh = list(network.edges[current].keys())
            random.shuffle(neigh)
            for i in range(0, len(neigh)):
                m = neigh[i]
                if network.edges[current][m] >= 0:
                    if m not in visited:
                        pot = network.distance[current] + network.edges[current][m]
                        if pot <= network.distance[m]:
                            if network.layer[m] == 'local':
                                variation += network.weights[m+network.N]*(network.distance[m] - pot)#change2
                            network.distance[m] = pot
                            heapq.heappush(heap_distance, (network.distance[m], m))
            visited[current] = 1
    ################
    ########################################


    ###

    return variation
##########################################




def initialize_score_edges (network):

    C = network.center + network.N
    for m in network.edges[C]:
        if network.layer[m] == 'global':
            var = evaluate_score_potential_edge ([C,m], network)
            if C<m:
                network.score_edges[C, m] = var
                heapq.heappush(network.heap_score_edges, (-network.score_edges[C,m], [C,m]))
            else:
                network.score_edges[m, C] = var
                heapq.heappush(network.heap_score_edges, (-network.score_edges[m,C], [m,C]))      #############




def select_best_edge (network):

    if len(network.heap_score_edges) == 0:
        return [-1, -1], 0
    
    tmp_buffer = []
    best_edges = []

    tmp = heapq.heappop(network.heap_score_edges)
    edge = tmp[1]
    score = tmp[0]
    best_score = score
    best_edges.append(edge)
    tmp_buffer.append(edge)
    
    control = 0
    while control == 0 and len(network.heap_score_edges)>0:

        tmp = heapq.heappop(network.heap_score_edges)
        edge = tmp[1]
        score = tmp[0]
        
        if score == best_score:
            best_edges.append(edge)
            tmp_buffer.append(edge)

        else:
            control = 1
            heapq.heappush(network.heap_score_edges, (score, edge))

            
    selected = random.choice(best_edges)

    for edge in tmp_buffer:
        n = edge[0]
        m = edge[1]
        if n != selected[0] or m != selected[1]:
            #print (m, n, selected[0], selected[1])
            heapq.heappush(network.heap_score_edges, (-network.score_edges[n,m], [n,m]))

    var = network.score_edges[selected[0], selected[1]]
    network.score_edges[selected[0], selected[1]] = -1e10
            
    return selected, var




###################


def update_scores (selected, network):

    tmp_buffer = []
    updated = {}
    min_var = 1e10

    C = selected[0]
    for m in network.edges[C]:
        if network.layer[m] == 'global':
            if network.edges[C][m] > 1e7:
                edge = [C, m]
                if m <= C:
                    edge = [m, C]
                if (edge[0], edge[1]) not in updated:
                    var = evaluate_score_potential_edge (edge, network)
                    network.score_edges[edge[0], edge[1]] = var
                    tmp_buffer.append(edge)
                    updated[edge[0], edge[1]] = 1
                    if -var < min_var:
                        min_var = -var


    C = selected[1]
    for m in network.edges[C]:
        if network.layer[m] == 'global':
            if network.edges[C][m] > 1e7:
                edge = [C, m]
                if m <= C:
                    edge = [m, C]
                if (edge[0], edge[1]) not in updated:
                    var = evaluate_score_potential_edge (edge, network)
                    network.score_edges[edge[0], edge[1]] = var
                    tmp_buffer.append(edge)
                    updated[edge[0], edge[1]] = 1
                    if -var < min_var:
                        min_var = -var

                        
    ##lazy update
    control = 0
    while control == 0 and len(network.heap_score_edges)>0:

        tmp = heapq.heappop(network.heap_score_edges)
        edge = tmp[1]
        score = tmp[0]

        
        #print (edge, score, min_var, network.edges[edge[0]][edge[1]])

        ##to avoid multiple updates for the same edge
        if (edge[0], edge[1]) not in updated and network.edges[edge[0]][edge[1]] > 1e7:
            updated[edge[0], edge[1]] = 1
            tmp_buffer.append(edge)
            #print ('--> ', edge, score, min_var)
            if score <= min_var:
                var = evaluate_score_potential_edge (edge, network)
                network.score_edges[edge[0], edge[1]] = var
                if -var < min_var:
                    min_var = -var
            else:
                control = 1

    ####
    for edge in tmp_buffer:
        n = edge[0]
        m = edge[1]
        #print (edge, network.score_edges[n,m])
        heapq.heappush(network.heap_score_edges, (-network.score_edges[n,m], [n,m]))


############
    
        

###########################################
def greedy_optimization (network, STEPS):

    t_start = process_time() 
    
    s = 0
    while s < STEPS:
        
        selected, var = select_best_edge (network)
        
        if var > 0:
            var = implement_change (selected, network)
            update_scores (selected, network)

            network.sequence.append([selected, var])
            network._sequence.append([selected, var])

        s +=1

    t_stop = process_time()
    network.optimization_time = t_stop-t_start
    
#############################################      
             



def measure_order_parameters (network, lattice):

    center = lattice.center

    for e in network.sequence:
        i = e[0][0]
        j = e[0][1]
        n = i - network.N
        m = j - network.N
        if n == center or m == center:
            network.center_degree += 1

        

    
    
    
def set_L_to(g,L):
    for i in g._sequence[L:]:
        g.edges[i[0][0]][i[0][1]],g.edges[i[0][1]][i[0][0]]=1e10,1e10
    g.sequence=g._sequence[:L]
    g.distance=compute_distance_to_center (g.center, g.edges)

def set_L(g,L):
    if len(g._sequence)>L:
        set_L_to(g,L)
    elif len(g._sequence)<L:    
        for i in g._sequence[len(g.sequence):L]:
            g.edges[i[0][0]][i[0][1]],g.edges[i[0][1]][i[0][0]]=g.eta,g.eta
    g.sequence=g._sequence[:L]
    g.distance=compute_distance_to_center (g.center, g.edges)


def unimplement_last_change(graph):
    edge=graph.sequence.pop()
    graph.edges[edge[0][0]][edge[0][1]]=1e10
    graph.edges[edge[0][1]][edge[0][0]]=1e10
    
def unimplement_change(edge,graph):
    i=0
    while True:
        if graph.sequence[i][0]==edge or graph.sequence[i][0]==edge[::-1]:
            break
    graph.edges[edge[0][0]][edge[0][1]]=1e10
    graph.edges[edge[0][1]][edge[0][0]]=1e10
