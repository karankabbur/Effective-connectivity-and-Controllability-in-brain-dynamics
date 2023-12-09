import numpy as np
import regex as re
from scipy.linalg import solve_continuous_lyapunov, solve_discrete_lyapunov
import networkx as nx
import operator
from copy import deepcopy
import pandas as pd
from networkx.algorithms import bipartite
from networkx.algorithms import components
import random
import copy
import rankaggregation as ra
from scipy.io import loadmat
import scipy.stats as stats
import json


#### By convention, A[i,j] is from i to j




def dediag(A):
	return A-np.diag(np.diag(A))


# sorts files by alphanumeric order
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)




# converts adjacency matrix into netowrkX fgraph, assuming A_ij is link from i to j
def get_graph_from_adjacency(adjacency):        

    nodes_list = []
    for i in range( adjacency.shape[0] ):
        nodes_list.append((i+1)) 
 
    # creating an edge list from adjacency matrix
    edge_list=[]
    for i in range( adjacency.shape[0] ) :
        for j in range(adjacency.shape[0]):
            if adjacency[i,j]==0: # if there's no weight, let's forget the edge.
                pass
            else:
                edge_list.append( (i+1,j+1,{"weight":adjacency[i,j]}) )
    # Creating a directed graph from adjacency matrix so that it will be easy to deal using the networkx module

    G = nx.DiGraph()

    # Add all the nodes to the graph
    G.add_nodes_from(nodes_list)
    # Add all the edges to the graph
    G.add_edges_from(edge_list)
    return G




############################### CONTROLLABILITY FUNCTIONS ###################################################################################################


def compute_B(N, control_nodes,acc=None):

    if acc is None:
    	acc=np.ones(N)

    M = len(control_nodes)
    if M==0:
        B = np.ones(N).astype(int)[:,np.newaxis]
        
    else:
        B = np.zeros((N,M))

        for i,j in zip(control_nodes, range(B.shape[1])):
            B[i-1][j] = acc[i-1]
    return B


def controllability_matrix(A, B):
    C = deepcopy(B)
    temp = deepcopy(B)
    
    N = A.shape[0]
    for j in range(N-1):
        temp = np.matmul(A,temp)
        C = np.c_[C, temp] # same as hstack, append them side by side
        #print("C shape",C.shape)   
    return C


def compute_e_min(min_evals):
    # consider only real part of eigenvalues since there 
    # is numerically close to 0 for imaginary part
    min_evals = np.real(np.array(min_evals))
    
    # set all the eigen values less than 10^-12 as 10^-12 for plotting sake 
    min_evals[min_evals<1e-12]=1e-12
    
    e_min = np.array(min_evals)**-1
    return e_min



############################### GRAPH CENTRALITY FUNCTIONS ###################################################################################################


def pq_values(A): #p_i by q_i, high to low
    N = len(A)
    
    centr_dict = {}
    for i in range(1,N+1):
        # either output node to be controlled for qi or input node to control i.e. pi
        e_i = compute_B(N, [i])
        
        # p_i
        W_i = solve_continuous_lyapunov(A.T, -np.matmul(e_i, e_i.T))   ## we are using A.T sincev by convention in the pq paper the matrix has to be from target to source 
        p_i = np.trace(W_i)
        
        # q_i
        M_i = solve_continuous_lyapunov(A, -np.matmul(e_i, e_i.T))
        q_i = np.trace(M_i)
        
        centr_dict[i] = p_i/q_i
    #sorted_d = dict( sorted(centr_dict.items(), key=operator.itemgetter(1),reverse=True)) # reverse = True -> h2l
    return list(centr_dict.values())#sorted_d    
    
def modified_pq_values(A,acc): #p_i by q_i, high to low
    N = len(A)
    
    centr_dict = {}
    for i in range(1,N+1):
        # either output node to be controlled for qi or input node to control i.e. pi
        e_i = compute_B(N, [i],acc=acc)
        
        # p_i
        W_i = solve_continuous_lyapunov(A.T, -np.matmul(e_i, e_i.T))   ## we are using A.T sincev by convention in the pq paper the matrix has to be from target to source 
        p_i = np.trace(W_i)
        
        e_i = compute_B(N, [i])
        # q_i
        M_i = solve_continuous_lyapunov(A, -np.matmul(e_i, e_i.T))
        q_i = np.trace(M_i)
        
        centr_dict[i] = p_i/q_i
    #sorted_d = dict( sorted(centr_dict.items(), key=operator.itemgetter(1),reverse=True)) # reverse = True -> h2l
    return list(centr_dict.values())#sorted_d    



def Degree_values(A,direction='out',diag=False,weight=True,sign=False): # unweighted, low to high ranking 
	if(diag==False):
		A=dediag(A)

	if(direction=='out'):
		axis=1
	elif(direction=='in'):
		axis=0
	else:
		print('set direction=in or direction=out')

	if(weight==True and sign==False):
		degree = np.sum(np.abs(A), axis=axis) # weighted absolute out-degree	
	elif(weight==True and sign==True):
		degree = np.sum(A, axis=axis) # weighted out-degree	
	else:
		degree = np.sum(A!=0, axis=axis).astype(int) # unweighted out-degree	

	return degree



def weight_OutbyIn_Degree_values(A): # weighted, high to low ranking 
    out_degree = Degree_values(A,direction='out',diag='False',weight=True,sign=False) # weighted out-degree
    in_degree = Degree_values(A,direction='in',diag='False',weight=True,sign=False) # weighted In-degree
    
    ratio = out_degree/in_degree
    
    return ratio


def pageRank_values(A,weight=True,diag=True,sign=False): # Unweighted, low to high

	if diag==False:
		A=dediag(A)
	
	if sign==False:
		A=np.abs(A) # reason it doesn't converges is because of negative entries
    

	G = get_graph_from_adjacency(A)
    
	if(weight==True):
		centr_dict = dict(nx.pagerank(G, alpha=0.85, tol=1e-06, max_iter=5000, weight="weight"))
	else:
		centr_dict = dict(nx.pagerank(G, alpha=0.85, tol=1e-06, max_iter=5000, weight=None))

	return list(centr_dict.values())
    
    
def rank_pq(A): #p_i by q_i, high to low
	N = len(A)
	values = pq_values(A)
	nodes = range(1,N+1)
  
	centr_dict = dict(zip(nodes, values))
 
 
	sorted_d = dict( sorted(centr_dict.items(), key=operator.itemgetter(1),reverse=True))
	return sorted_d    

def rank_modified_pq(A,acc): #p_i by q_i, high to low
	N = len(A)
	values = modified_pq_values(A,acc)
	nodes = range(1,N+1)
  
	centr_dict = dict(zip(nodes, values))
 
 
	sorted_d = dict( sorted(centr_dict.items(), key=operator.itemgetter(1),reverse=True))
	return sorted_d    
 
     
def rank_Degree(A,direction='out',diag=False,weight=True,sign=False,rankorder='increasing'): # unweighted, low to high ranking 
    
	N = len(A)
	nodes = range(1,N+1)
	degree = Degree_values(A,direction=direction,diag=diag,weight=weight,sign=sign) 
 
	centr_dict = dict(zip(nodes, degree))

	if rankorder=='increasing':
	    sorted_d = dict( sorted(centr_dict.items(), key=operator.itemgetter(1),reverse=False)) 
	else:
	    sorted_d = dict( sorted(centr_dict.items(), key=operator.itemgetter(1),reverse=True))   
    
	return sorted_d
    

def rank_weight_OutbyIn_Degree(A): 
    
    N = len(A)
    nodes = range(1,N+1)
    
    ratio = weight_OutbyIn_Degree_values(A)
        
    centr_dict = dict(zip(nodes, ratio))
    sorted_d = dict( sorted(centr_dict.items(), key=operator.itemgetter(1),reverse=True)) 
    
    return sorted_d

    
def rank_pageRank(A,weight=True,diag=True,sign=False): 

	N = len(A)
	nodes = range(1,N+1)
    
	values=pageRank_values(A,weight=weight,diag=diag,sign=sign)
  
	centr_dict = dict(zip(nodes, values))
    
	sorted_d = dict( sorted(centr_dict.items(), key=operator.itemgetter(1),reverse=False)) # l2h
	return sorted_d


def rank_deg_comm(A,region_dict,direction='out',rankorder='decreasing'):
	
	
	A=dediag(A)
	A=np.abs(A)
	
	region_names = np.unique(np.array(list(region_dict.values())))
	
	N=np.shape(A)[0]
	
	region_idx=np.zeros(N)



	for key, value in region_dict.items():
	    for i in range(len(region_names)):
	        if value == region_names[i]:   
	        	region_idx[key-1]=i

	rankdegcomm = np.zeros(len(region_dict.items()))

	if(direction=='out'):
		D = np.sum(A,axis=1)
	elif(direction=='in'):
		D = np.sum(A,axis=0)


	for i in range(len(region_names)):
		idx=np.where(region_idx==i)[0]
		
		
		Ai = A[np.ix_(idx,idx)]
		
		if(direction=='out'):
			Di = np.sum(Ai,axis=1)
		elif(direction=='in'):
			Di = np.sum(Ai,axis=0)

		'''
		Di = D[idx]
		'''

		if(rankorder=='decreasing'):	
			ri = np.argsort(np.argsort(-Di))
		
		elif(rankorder=='increasing'):	
			ri = np.argsort(np.argsort(Di))

		rankdegcomm[idx]=ri


	dtype = [('comm',int),('rankcomm', int),('deg', float) ]	
	
	v = []
	
	for i in range(len(region_dict.items())):
		v.append((region_idx[i],rankdegcomm[i],-D[i]))

	v = np.array(v, dtype=dtype)       

	
	order = np.argsort(np.argsort(v,order=['rankcomm','deg']))	
	
	
	nodes = range(1,N+1)
    
	values=order
  
	centr_dict = dict(zip(nodes, values))
    
	sorted_d = dict( sorted(centr_dict.items(), key=operator.itemgetter(1),reverse=False)) # l2h
	return sorted_d





def Nodes_eval_relation(adj_original, centrality,Atype='continuous',region_dict=None,seed=123,acc=None,adj_rank=None):
	A =  adj_original    
	N = len(A)
	
	if acc is None:
	    acc=np.ones(N)
	   
	if adj_rank is None:
		adj_rank=A
	Ar=adj_rank
     
	if centrality=="random":
		np.random.seed(seed)
		control_nodes = list(range(1,N+1))
        # randomize the control nodes
		np.random.shuffle(control_nodes)
		print(control_nodes)

	else:
		if centrality=="pq":
			sorted_d = rank_pq(Ar)
		elif centrality=="out_degree":
			sorted_d = rank_Degree(Ar,rankorder='decreasing') # A should be orginal matrix
		elif centrality=="ratio_degree":
			sorted_d = rank_weight_OutbyIn_Degree(Ar)
		elif centrality=="page_rank":
			sorted_d = rank_pageRank(Ar)    
		elif centrality=="in_degree":
			sorted_d = rank_Degree(Ar,direction='in',rankorder='decreasing')    
		elif centrality=="rankdegcomm":
			sorted_d = rank_deg_comm(Ar,region_dict,direction='out',rankorder='decreasing')
		elif centrality=="modified_pq":
			sorted_d = rank_modified_pq(Ar,acc)

            
		control_nodes = list(sorted_d.keys())     # nodes sorted according to criterion
		print(control_nodes)
   
	#Num_control_nodes = []
    #min_evals_W = []
    
	evals_list = np.zeros((N,N))
    
	#evals_list = []

	Num_control_nodes=list(range(N,0,-1))

	for n in range(N):
	#while len(control_nodes)!=0: 
		#Num_control_nodes.append(len(control_nodes))
        
        # compute B 
        # remember to sort the control nodes before constructing the input matrix B
		B = compute_B(N, sorted(control_nodes[:N-n]), acc=acc)
    
        # check Kalman rank    
		C = controllability_matrix(A.T, B)  # in the controllability framework, A[i,j] is from j to i , see Barabasi 2011 paper
		rank_C = np.linalg.matrix_rank(C,tol=1E-50).item(0) # Full rank matrix
		if rank_C==N:
  			         
        	# compute W
			if(Atype=='discrete'):
				W = solve_discrete_lyapunov(A.T, np.matmul(B, B.T)) 
			else:
				W = solve_continuous_lyapunov(A.T, -np.matmul(B, B.T)) 

			#evals_list.append( np.linalg.eigvals(W) )
			evals_list[n,:]=np.linalg.eigvals(W)
		
        	#print(np.max(1./np.linalg.eigvals(W)))
        
		else:
			print("rank_C != N for Nd = %d"%B.shape[1] +'control nodes: ',control_nodes)

			#evals_list.append( 1.0E-12*np.ones(N))
			evals_list[n,:]=1.0E-12*np.ones(N)


               
        # delete the control node which has lowest out degree.
		#del control_nodes[-1]
	return (Num_control_nodes, evals_list) #np.array(evals_list))




import operator

def Nodes_energy_relation(adj_original, centrality, target_list, energy_type="min", seed=123,acc=None,adj_rank=None):
	A =  adj_original    

	if adj_rank is None:
		adj_rank=A
	Ar=adj_rank


	N = len(A)# N=74 in our case

	C = compute_C(N, target_list)# output target matrix


	if centrality=="random": # atleast 10 realizations has to be done.
		np.random.seed(seed)
		control_nodes = list(range(1,N+1))
		# randomize the control nodes
		np.random.shuffle(control_nodes)
		#control_nodes=list(control_nodes)
	else:
		if centrality=="pq":
			sorted_d = rank_pq(Ar)
		elif centrality=="out_degree":
			sorted_d = rank_Degree(Ar,rankorder='decreasing') # A should be orginal matrix
		elif centrality=="ratio_degree":
			sorted_d = rank_weight_OutbyIn_Degree(Ar)
		elif centrality=="page_rank":
			sorted_d = rank_pageRank(Ar)  
		elif centrality=="modified_pq":
			sorted_d = rank_modified_pq(Ar,acc)  
        
		control_nodes = list(sorted_d.keys())
    # put a constraint that target nodes cannot become control nodes
	for node in target_list: 
		control_nodes.remove(node)

    
    #Num_control_nodes = []
    #min_evals_W = []
    
	n_control = len(control_nodes)
    
	min_evals_W = np.zeros(n_control)

	Num_control_nodes=list(range(n_control,0,-1))

	for n in range(n_control):
	#while len(control_nodes)!=0:    
        #Num_control_nodes.append(len(control_nodes))
        
        # compute B 
        # remember to sort the control nodes before constructing the input matrix B
		
		B = compute_B(N, sorted(control_nodes[:n_control-n]) ) ## Mistake!!! 
        
        # check Kalman rank    
		d = target_controllability_matrix(A.T, B, C) # in the controllability framework, A[i,j] is from j to i , see Barabasi 2011 paper
		rank_d = np.linalg.matrix_rank(d, tol=1E-50).item(0) # Full rank matrix 
		if rank_d==len(target_list):
    	    # compute W
			W = solve_continuous_lyapunov(A.T, -np.matmul(B, B.T))  # in the controllability framework, A[i,j] is from j to i 
			W_c = np.matmul(C, np.matmul(W, C.T))
    
    	    # compute  e_min if possible
    	    #min_evals_W.append( min(np.linalg.eigvals(W_c)) )
			if(energy_type=="min"):
				min_evals_W[n]=min(np.linalg.eigvals(W_c))
			if(energy_type=="median"):
				min_evals_W[n]=np.median(np.linalg.eigvals(W_c))    	        	    
		else:
			sprint("rank_d != S for Nd = ",B.shape[1])
			min_evals_W[n]=1.0E-12*np.ones(N)
    	  
        
        # delete the control node which has lowest out degree.
        #del control_nodes[-1]
    #print("Min eval:", min_evals_W)
	e_min_list = compute_e_min(min_evals_W)
        
	return (Num_control_nodes, e_min_list)





def AllNodes_energy_relation(adj_original, centrality, target_list, seed=123):
	A =  adj_original    
	N = len(A)# N=74 in our case

	v = np.zeros(N)
    
	target_list1 = list(np.asarray(target_list)-1)
    
	v[target_list1]=1
	v = v/np.sqrt(np.sum(np.square(v)))

	#C = compute_C(N, target_list)# output target matrix

	if centrality=="random": # atleast 10 realizations has to be done.
		np.random.seed(seed)
		control_nodes = list(range(1,N+1))
		# randomize the control nodes
		np.random.shuffle(control_nodes)
		#control_nodes=list(control_nodes)
	else:
		if centrality=="pq":
			sorted_d = rank_pq(A)
		elif centrality=="out_degree":
			sorted_d = rank_Degree(A,rankorder='decreasing') # A should be orginal matrix
		elif centrality=="ratio_degree":
			sorted_d = rank_weight_OutbyIn_Degree(A)
		elif centrality=="page_rank":
			sorted_d = rank_pageRank(A)    
        
		control_nodes = list(sorted_d.keys())
    # put a constraint that target nodes cannot become control nodes
	for node in target_list: 
		control_nodes.remove(node)

    
    #Num_control_nodes = []
    #min_evals_W = []
    
	n_control = len(control_nodes)
    
	min_evals_W = np.zeros((n_control))

	Num_control_nodes=list(range(n_control,0,-1))

	for n in range(n_control):
	#while len(control_nodes)!=0:    
        #Num_control_nodes.append(len(control_nodes))
        
        # compute B 
        # remember to sort the control nodes before constructing the input matrix B
		
		B = compute_B(N, sorted(control_nodes[:n_control-n]) ) ### MIstake!!! 
        
        # check Kalman rank    
		#d = target_controllability_matrix(A.T, B, C) # in the controllability framework, A[i,j] is from j to i , see Barabasi 2011 paper
		#rank_d = np.linalg.matrix_rank(d, tol=1E-50).item(0) # Full rank matrix 
		#if rank_d==len(target_list):
    	    # compute W
		W = solve_continuous_lyapunov(A.T, -np.matmul(B, B.T))  # in the controllability framework, A[i,j] is from j to i 
		W_c = np.matmul(v, np.matmul(W, v.T))
    
    	    # compute  e_min if possible
    	    #min_evals_W.append( min(np.linalg.eigvals(W_c)) )

		if(np.real(W_c)>1.0E-12):
			min_evals_W[n]=np.real(W_c) #min(np.linalg.eigvals(W_c))
		else:
	 		min_evals_W[n]=1.0E-12
    	        	    
#		else:
#			sprint("rank_d != S for Nd = ",B.shape[1])
#			min_evals[n,:]=1.0E-12*np.ones(N)
    	  
        
        # delete the control node which has lowest out degree.
        #del control_nodes[-1]
    #print("Min eval:", min_evals_W)
	e_min_list = compute_e_min(min_evals_W)
        
	return (Num_control_nodes, e_min_list)








def compute_e_min(min_evals):
    # consider only real part of eigenvalues since there 
    # is numerically close to 0 for imaginary part
    min_evals = np.real(np.array(min_evals))
    
    # set all the eigen values less than 10^-12 as 10^-12 for plotting sake 
    min_evals[min_evals<1e-12]=1e-12
    
    e_min = np.array(min_evals)**-1
    return e_min









############################################## MAX MATCHING FUNCTIONS ###############################




class max_match_directed:
    def __init__(self,dir_graph):#, position, label):
        self.dir_graph = dir_graph
        #self.position = position
        #self.label = label
        self.temp_d = dir_graph.copy()

    def info(self):
        print(nx.info(self.dir_graph))
    
    def degree_in_out(self):
        self.din = []
        self.dout = []
        for value in self.dir_graph.in_degree():
            self.din.append(value[1])
        for value in self.dir_graph.out_degree():
            self.dout.append(value[1])
        self.d_in = sum(self.din)/float(nx.number_of_nodes(self.dir_graph))
        self.d_out = sum(self.dout)/float(nx.number_of_nodes(self.dir_graph))
        return self.d_in, self.d_out
    
    def bipartition(self):
        #print("Nodes dgraph before:", self.dir_graph.nodes())
        self.Bip_rappr = nx.Graph()
        self.rel_nodes = {}
        self.n_nodes = len(nx.nodes(self.dir_graph))
        
        node_list_dir = list(nx.nodes(self.dir_graph))   
        for i,j in zip(node_list_dir,range(1,self.n_nodes+1)): #range(self.n_nodes):
            self.rel_nodes[i] = j
        #print("mapping",self.rel_nodes)
        
        self.DiG_rel = nx.relabel_nodes(self.dir_graph, self.rel_nodes)
        self.bipart_DiG_rel = []
        
        for i in self.DiG_rel.edges():
            #print('edge',i)
            self.bipart_DiG_rel.append((i[0],-i[1]))
            #print('create edge',i[0],-i[1])
        self.Bip_rappr.add_edges_from(self.bipart_DiG_rel)
        #print("dgraph edges 1:",self.dir_graph.edges())
        #print("dgraph edges 2:",self.DiG_rel.edges())
        #print("bip_graph edges: ",self.Bip_rappr.edges())
        #print("Nodes dgraph after:", self.dir_graph.nodes())
        
        #return self.Bip_rappr
        
    def connection(self):
        if components.is_connected(self.Bip_rappr)==False:
            #self.n_connected_components = len(list(nx.connected_component_subgraphs(self.Bip_rappr)))
            self.n_connected_components = len(list(self.Bip_rappr.subgraph(c).copy()\
                                                   for c in nx.connected_components(self.Bip_rappr)))
            
            #print(self.n_connected_components)
        else:
            print('is connected')
            
            
    def max_matching_bipartite(self):
        self.match_list_reduced= []
        self.matched_edges_b = []
        if nx.is_connected(self.Bip_rappr) == False:
            #self.graphss = list(nx.connected_component_subgraphs(self.Bip_rappr))
            self.graphss = list(self.Bip_rappr.subgraph(c).copy() for c in nx.connected_components(self.Bip_rappr))
            #print("List of connected components:",[c.nodes() for c in (self.graphss)])
            
            for i in range(len(self.graphss)):
                self.matching = bipartite.matching.hopcroft_karp_matching(self.graphss[i])
                #print("Hopcroft matching:",self.matching)
                #print('matching',self.matching)
                for key in self.matching:
                    if key>0:
                        self.match_list_reduced.append(self.matching[key])
                        self.matched_edges_b.append((key,self.matching[key]))
                #print("matched edges b:", self.matched_edges_b)
        else: 
            bipartite.maximum_matching(self.Bip_rappr)
            self.matching = bipartite.matching.hopcroft_karp_matching(self.Bip_rappr)
            for key in self.matching:
                #print('matching',self.matching)
                if key> 0:
                    self.match_list_reduced.append(self.matching[key])
                    self.matched_edges_b.append((key,self.matching[key]))
        
        #print("matched edges b:", self.matched_edges_b)
        #print("matched nodes:", self.match_list_reduced)

        self.match_list_reduced.sort()
        
        self.unmatched_edges_b = []
        self.unmatched_edges_b_temp = []
        self.unmatched_nodes_b = []

        self.temp_b = self.Bip_rappr.copy()
        self.temp_b.remove_edges_from(self.matched_edges_b)
        self.unmatched_edges_b_temp = list(self.temp_b.edges)
        self.temp_b.remove_nodes_from(self.match_list_reduced)
        self.unmatched_nodes_b = list(self.temp_b.nodes)

        #print("unmatched nodes brefore:", self.unmatched_nodes_b)
        
        
        for edges in self.unmatched_edges_b_temp:
            if edges[0]>0 and edges[1]<0:
                self.unmatched_edges_b.append(edges)
            elif edges[0]<0 and edges[1]>0:
                self.unmatched_edges_b.append((edges[1],edges[0]))
            else:
                print('probLLAMA')
        
        #print("unmatched nodes after:", self.unmatched_nodes_b)
        #print("unmatched edges after:", self.unmatched_edges_b)

        
        #print('matched nodes neg bip',self.match_list_reduced)
        #print('unmatched nodes neg bip',self.unmatched_nodes_b)
        #print('matched edges bip',self.matched_edges_b)
        #print('unmatched edge b',self.unmatched_edges_b)

        
    def max_matching_digraph(self):
        self.match_nodes_d = []
        self.unmatch_nodes_d = []
        for node in self.match_list_reduced:
            #self.match_nodes_d.append(-node-1) # remember this was from old code of Samir's student
            self.match_nodes_d.append(-node)
        self.temp_d.remove_nodes_from(self.match_nodes_d)
        for node in self.temp_d.nodes:
            self.unmatch_nodes_d.append(node)
            
        self.unmatched_edges_d = []
        self.matched_edges_d = []

        for edges in self.unmatched_edges_b:
    #        self.unmatched_edges_d.append((edges[0]-1,-edges[1]-1))
            self.unmatched_edges_d.append((edges[0],-edges[1]))
    
        for edges in self.matched_edges_b:
            #self.matched_edges_d.append((edges[0]-1,-edges[1]-1))
            self.matched_edges_d.append((edges[0],-edges[1]))
        
        self.N_D= len(self.unmatch_nodes_d) 
        self.ND = str(self.N_D)

        #print('matched nodes', self.match_nodes_d)
        #print('unmatched nodes',self.unmatch_nodes_d)    
        #print('N_D:',len(self.unmatch_nodes_d)) #AGGIUNGI I NODI ISOLATI
        #print('unmatched edges digraph',self.unmatched_edges_d)
        #print('matched edges digraph',self.matched_edges_d)
        return self.match_nodes_d, self.unmatch_nodes_d,self.N_D, self.matched_edges_d, self.unmatched_edges_d
 
 
def max_matching_info(G):
    m = max_match_directed(G)#, position, label)
    d_in,d_out = m.degree_in_out()
    m.bipartition()
    #m.connection()

    m.max_matching_bipartite()

    matched_nodes, unmatched_nodes, N_D, matched_edges, unmatched_edges = m.max_matching_digraph()
    return matched_nodes, unmatched_nodes, N_D, matched_edges, unmatched_edges
    
    
    
    
    
    
    
    
########################################################## SPARSITY FUNCTIONS ##########################################
        

def kill_entries(A,method='random', f=0.1,random_seed=123):
    if method=='random':
    
        np.random.seed(random_seed)
        # consider the indices of random matrix which are non zero. 
        # It is of form (num x 2), cols are index pos of matrix
        ind_mat = np.argwhere(A)

        # reshuffle them
        np.random.shuffle(ind_mat)

        # number of non zero entries of A
        num_nonZero_A = ind_mat.shape[0]

        # number of entries to be killed
        num_entries_to_kill = int(f*num_nonZero_A)

        # initialize A_sparse to A before we kill the entries
        A_sparse = np.zeros(A.shape)
        A_sparse = deepcopy(A)

        # always select first num_entries_to_kill from ind_mat
        for i in range(num_entries_to_kill):
            A_sparse[ind_mat[i][0],ind_mat[i][1]] = 0

        return A_sparse

    elif method=='absmin':
        B = deepcopy(A)
        B=B-np.diag(np.diag(B))
        B=np.abs(A)    

        th = np.percentile(B[B>0],100*f)

        A_sparse = deepcopy(A)

        A_sparse[np.where(np.abs(A_sparse)<th)]=0 

        return A_sparse

    elif method=='absmax':
        B = deepcopy(A)
        B=B-np.diag(np.diag(B))
        B=np.abs(A)    

        th = np.percentile(B[B>0],100-100*f)

        A_sparse = deepcopy(A)

        A_sparse[np.where(np.abs(A_sparse)>=th)]=0 

        return A_sparse

    elif method=='min':
        B = deepcopy(A)
        B=B-np.diag(np.diag(B))
            

        th = np.percentile(B[B!=0],100*f)
        #print("threshold: ",th)
        #print("min B: ",np.min(B.flatten()))

        A_sparse = deepcopy(A)

        A_sparse[np.where(A_sparse<th)]=0 

        return A_sparse

    elif method=='max':
        B = deepcopy(A)
        B=B-np.diag(np.diag(B))

        th = np.percentile(B[B!=0],100-100*f)

        A_sparse = deepcopy(A)

        A_sparse[np.where(A_sparse>=th)]=0 

        return A_sparse
    else:
        print("Error")




def sparseMatrix_controllability(adj_original,method='random', graph_type="original",\
                                 input_matrix="B_unmatched",seed=123):
    #adj_original = np.load(adj_matfile) 
    A =  adj_original
    
    N = len(A)# N=74 in our case
    #print("Max eval A_original:", max(np.real(np.linalg.eigvals(A))))
    
    # Check for stability of LTI equation using the eigen values of A
    max_evals_A_sp = [] # list of max eigen values of sprices of A
    min_evals_W_sp = [] # list of max eigen values of sprices of A
    
    Nd_list = []
    kalman_rank_list = []
    density_list = []
    
    fraction_removed = np.linspace(0.1,0.95,50)#np.arange(0.1, 0.85,0.05) 
    
    max_eval_A = max(np.real(np.linalg.eigvals(A)))
    #print("Max value of A: ", max_eval_A)
    
    ############ B_max ######################
    if input_matrix == "B_max":

        A_sp = kill_entries(A-np.diag(np.diag(A)), f=0.95, method=method)
        #print("Non zero A_sp, max killing ",np.argwhere(A_sp))


        # reassign N again
        N = A_sp.shape[0]

        # compute unmatched nodes 
        G = get_graph_from_adjacency( A_sp.T -np.diag(np.diag(A_sp.T))) # while computing unmatched nodes, should be original A  

        if graph_type=="erdos":
                # Null model 1
                # Erdos-renyi graph
                #print("Entries before erdos: ", len(G.nodes()))
                N, p = len(A_sp), nx.density(G)
                G_erdos = nx.erdos_renyi_graph(n=N, p=p, seed=seed, directed=True)
                A_erdos = np.array(nx.adjacency_matrix(G_erdos).todense())
                A_erdos = assign_weights_to_randMatrices(A_sp, A_erdos, seed=seed)
                # remove all unwanted diagonal entries if present
                A_erdos = A_erdos-np.diag(np.diag(A_erdos)) 

                A_sp = deepcopy(A_erdos)
                G = deepcopy(G_erdos)
                #print("Entries after erdos: ", len(G.nodes()))


        elif graph_type=="dirConfig":
                # Null model 2
                # Directed configuration model
                in_deg = list(dict(G.in_degree()).values())
                out_deg = list(dict(G.out_degree()).values())
                G_dirConfig = nx.directed_configuration_model(in_deg, out_deg,seed=seed)

                A_dirConfig = np.array(nx.adjacency_matrix(G_dirConfig).todense())

                A_dirConfig = assign_weights_to_randMatrices(A_sp, A_dirConfig, seed=seed)
                # remove all unwanted diagonal entries if present
                A_dirConfig = A_dirConfig-np.diag(np.diag(A_dirConfig)) 


                A_sp = deepcopy(A_dirConfig)
                G = deepcopy(G_dirConfig)

        unmatched_nodes = max_matching_info(G)[1]
        print("Max driver nodes for B_max: ", len(unmatched_nodes))

        # compute B max
        B = compute_B(N, unmatched_nodes)
        
    elif input_matrix=="B_50percent":
        random.seed(seed)
        driver_nodes = random.sample(range(1,N+1), int(0.5*N))
        B = compute_B(N, driver_nodes)


    ##################################################################
    #diag_entries = np.diag(np.diag(A))
        
    for i in fraction_removed:    
        
        # A with diagonal entries 0
        #A_sp = kill_entries(A, i) #+ np.diag(np.diag(A))#till 0.9
        A_sp = kill_entries(A-np.diag(np.diag(A)), f=i,method=method) #+ np.diag(np.diag(A))#till 0.9
        
        
        # reassign N again
        N = A_sp.shape[0]
        
        # transpose is done, since it has to be original matrix when converting to a network
        G = get_graph_from_adjacency( A_sp.T-np.diag(np.diag(A_sp.T)) )  
        
        if graph_type=="erdos":
            # Null model 1
            # Erdos-renyi graph
            N, p = len(A_sp), nx.density(G)
            G_erdos = nx.erdos_renyi_graph(n=N, p=p, seed=seed, directed=True)
            A_erdos = np.array(nx.adjacency_matrix(G_erdos).todense())
            A_erdos = assign_weights_to_randMatrices(A_sp, A_erdos, seed=seed)
            # remove all unwanted diagonal entries if present
            A_erdos = A_erdos-np.diag(np.diag(A_erdos)) 
            
            A_sp = deepcopy(A_erdos)
            G = deepcopy(G_erdos)
        
        elif graph_type=="dirConfig":
            # Null model 2
            # Directed configuration model
            in_deg = list(dict(G.in_degree()).values())
            out_deg = list(dict(G.out_degree()).values())
            G_dirConfig = nx.directed_configuration_model(in_deg, out_deg,seed=seed)

            A_dirConfig = np.array(nx.adjacency_matrix(G_dirConfig).todense())

            A_dirConfig = assign_weights_to_randMatrices(A_sp, A_dirConfig, seed=seed)
            # remove all unwanted diagonal entries if present
            A_dirConfig = A_dirConfig-np.diag(np.diag(A_dirConfig)) 
            
            
            A_sp = deepcopy(A_dirConfig)
            G = deepcopy(G_dirConfig)
        
        # compute unmatched nodes 
        
        unmatched_nodes = max_matching_info(G)[1]
        Nd_list.append( len(unmatched_nodes))

        # compute density
        density_list.append( nx.density(G) )
        
        if input_matrix=="B_unmatched":    
            # compute B
            B = compute_B(N, unmatched_nodes)

        ####### stabilize the LTI before proceeding to check Kalman controllability  ####

        A_sp = A_sp + np.diag(np.diag(A)) 

        # check if max eval of A is negative or not
        max_evals_A_sp.append( max(np.real(np.linalg.eigvals(A_sp)))  )

        # check Kalman rank    
        C = controllability_matrix(A_sp, B)
        rank_C = np.linalg.matrix_rank(C,tol=1E-50).item(0) # Full rank matrix 
        if rank_C!=N:
            kalman_rank_list.append(False)
        else:
            kalman_rank_list.append(True)

        # compute W
        #W = solve_continuous_lyapunov(A_sp, -np.matmul(B, B.T)) 

        # compute avg controllability 
        #avg_contr_list.append( np.trace(W) )

        # compute  e_min if possible
        #min_evals_W_sp.append( min(np.linalg.eigvals(W)) )

            
            
    #return (max_evals_A_sp, min_evals_W_sp, Nd_list, kalman_rank_list, \
    #        avg_contr_list,density_list, fraction_removed)
    return (Nd_list, fraction_removed)

 


def sparseMatrix_controllability_energy(adj_original, method="absmax", graph_type="original", seed=123):
    #adj_original = np.load(adj_matfile) 
    A =  adj_original# should be transpose according to Barabasi 2011 paper
    N = len(A)# N=71 in our case
    
    min_evals_W_sp = [] # list of max eigen values of sprices of A
    
    fraction_removed = np.linspace(0.1,0.95,50)#np.arange(0.1, 0.85,0.05) 
    
    # compute B corresponding to different types of graphs   
    
    if graph_type=="erdos":
        # Null model 1
        # Erdos-renyi graph
        #print("Entries before erdos: ", len(G.nodes()))
        G = get_graph_from_adjacency( A.T ) 
    
        N, p = len(A), nx.density(G)
        G_erdos = nx.erdos_renyi_graph(n=N, p=p, seed=seed, directed=True)
        
        centr_dict = dict(G_erdos.out_degree())
        sorted_d = dict( sorted(centr_dict.items(), key=operator.itemgetter(1),reverse=True)) # h2l
        control_nodes = list(sorted_d.keys())[:int(N/2)]
        B = compute_B(N, control_nodes)

    
    elif graph_type=="original" or graph_type=="dirConfig":
        nodes = range(1,N+1)
        out_degree = np.sum(adj_original.T!=0, axis=1).astype(int) # unweighted out-degree
        centr_dict = dict(zip(nodes, out_degree))
        
        sorted_d = dict( sorted(centr_dict.items(), key=operator.itemgetter(1),reverse=True)) # h2l
        control_nodes = list(sorted_d.keys())[:int(N/2)]
        B = compute_B(N, control_nodes)


    for i in fraction_removed:    
        
        A_sp = kill_entries(A-np.diag(np.diag(A)), f=i,method=method) #+ np.diag(np.diag(A))#till 0.9
       
        G = get_graph_from_adjacency( A_sp.T-np.diag(np.diag(A_sp.T)) )  
        
        if graph_type=="erdos":
            # Null model 1
            # Erdos-renyi graph
            N, p = len(A_sp), nx.density(G)
            G_erdos = nx.erdos_renyi_graph(n=N, p=p, seed=seed, directed=True)
            A_erdos = np.array(nx.adjacency_matrix(G_erdos).todense())
            A_erdos = assign_weights_to_randMatrices(A_sp, A_erdos, seed=seed)
            # remove all unwanted diagonal entries if present
            A_erdos = A_erdos-np.np(diag.diag(A_erdos)) 
            
            A_sp = deepcopy(A_erdos)
        
        elif graph_type=="dirConfig":
            # Null model 2
            # Directed configuration model
            in_deg = list(dict(G.in_degree()).values())
            out_deg = list(dict(G.out_degree()).values())
            G_dirConfig = nx.directed_configuration_model(in_deg, out_deg,seed=seed)

            A_dirConfig = np.array(nx.adjacency_matrix(G_dirConfig).todense())

            A_dirConfig = assign_weights_to_randMatrices(A_sp, A_dirConfig, seed=seed)
            # remove all unwanted diagonal entries if present
            A_dirConfig = A_dirConfig-np.diag(np.diag(A_dirConfig)) 
            A_sp = deepcopy(A_dirConfig)
        
       ####### stabilize the LTI before proceeding to check Kalman controllability  ####

        A_sp = A_sp + np.diag(np.diag(A)) 

        # check Kalman rank    
        #C = controllability_matrix(A_sp, B)
        #rank_C = np.linalg.matrix_rank(C,tol=1E-50).item(0) # Full rank matrix 
        #if rank_C!=N:
        #    kalman_rank_list.append(False)
        #else:
        #    kalman_rank_list.append(True)

        # compute W
        W = solve_continuous_lyapunov(A_sp, -np.matmul(B, B.T)) 

        # compute  e_min
        min_evals_W_sp.append( min(np.linalg.eigvals(W)) )
    return (min_evals_W_sp, fraction_removed)
 





########################################################## RANDOM GRAPH ###########################################################################

def assign_weights_to_randMatrices(A, A_rand, seed=123):
    
    # select all the non zero values of A
    nonZero_A = np.array(A[np.nonzero(A)])
    np.random.seed(seed)
    # reshuffle them
    np.random.shuffle(nonZero_A)
    
    # consider the indices of random matrix which are non zero. 
    # It is of form (num x 2), cols are index pos of matrix
    ind_mat = np.argwhere(A_rand)
    
    # length of non zero entries of A_rand
    num_nonZero_A_rand = len(A_rand[np.nonzero(A_rand)])
    #print(num_nonZero_A_rand)
    
    A_weighted_rand = np.zeros(A_rand.shape)
    for j in range(num_nonZero_A_rand): 
        # assign shuffles values of A to A_rand
        if j >=len(nonZero_A):
            break
        else:
            A_weighted_rand[ind_mat[j][0],ind_mat[j][1]] = nonZero_A[j]
        
    return A_weighted_rand








########################################################## TARGET CONTROLLABILITY ###########################################################################


def compute_C(N, target_nodes_list): # target matrix
    return compute_B(N, target_nodes_list).T
    

def target_controllability_matrix(A, B, C):
    d = np.matmul(C,B)#deepcopy(B)
    
    N = A.shape[0]
    for j in range(1,N):
        temp = np.matmul(C, np.matmul(np.linalg.matrix_power(A,j), B) )
        d = np.c_[d, temp] # same as hstack, append them side by side
        #print("C shape",C.shape)   
    return d


def target_nodes(region_dict, region_name):
    targetNodes = []
    for node, region in region_dict.items():
        if region==region_name:
            targetNodes.append(node)
    return targetNodes








######################################## AGGREGATION #########################################################

def _smooth(x):
    smooth = x.copy()
    n_changes = 1
    while n_changes != 0:
        prev = smooth.copy()
        for i in range(1, len(x)-1):
            smooth[i] = np.median(prev[i-1:i+2])
        smooth[0] = np.median([prev[0], smooth[1], 3 * smooth[1] - 2 * smooth[2]])
        smooth[-1] = np.median([prev[-1], smooth[-2], 3 * smooth[-2] - 2 * smooth[-3]])
        n_changes = np.sum(smooth != prev)
    return smooth


def _cummax(x):
    y = np.array([np.max(x[:i]) for i in range(1, len(x)+1)])
    return y


def rra(data, prior=0.2, num_bin=15, num_iter=30,
        return_all=False, corr_stop=1):
    # data:
    #   - data is a numpy matrix with objects in rows and different ranked lists in columns.
    #   - data is a real-valued matrix where entries with higher values indicate stronger preference.
    #
    # Note 1: Note that a higher value in data matrix is better and indicates a higher-priority of an object. When
    # converted to ranks the largest value gets rank 1. If input data are ranks (i.e., lower values indicate higher
    # priority), the ranks might need to be reversed.
    nr, nc = data.shape
    nrp = int(np.floor(nr * prior))
    #print(nrp)
    #print ('Nrp: %d'% nrp)
    rank_data = np.zeros(data.shape)
    for i in range(nc):
        rank_data[:, i] = stats.rankdata(-data[:, i]) / float(nr)

    bayes_factors = np.zeros((num_bin, nc))
    binned_data = np.ceil(rank_data * num_bin).astype('int')
    bayes_data = np.zeros((nr, nc))

    # estimated ranks, smaller is better (= closer to the top)
    
    # we need to reset the numbering in the end. 
    guess = np.mean(rank_data, 1)
    cprev = 0
    for iter in range(num_iter):
        if corr_stop - cprev < 1e-15:
            print ('CRank Converged!')
            break

        # assign the top np of aggregated predictions to be the positive class and the
        # rest of the predictions to the negative class
        guess_last = guess.copy()
        oo = np.argsort(guess)
        guess[oo[:nrp]] = 1.
        guess[oo[nrp:]] = 0.

        # Heuristic to make the approach more robust:
        # -- computing Bayes factors cumulatively starting from the top bin
        for i in range(nc):
            for bin in range(1, num_bin + 1):
                tpr = np.sum(guess[binned_data[:, i] <= bin])
                fpr = np.sum((1. - guess)[binned_data[:, i] <= bin])
                bayes_factors[bin-1, i] = np.log((tpr + 1.) / (fpr + 1.) / (prior / (1. - prior)))

        # Heuristic to make the approach more robust:
        # -- smoothing using Tukey's running median
        # -- enforcing a monotone decrease of Bayes factors
        for i in range(nc):
            bayes_factors[:, i] = _smooth(bayes_factors[:, i])
            bayes_factors[:, i] = _cummax(bayes_factors[:, i][::-1])[::-1]

        # winzorization step: for each bin we decrease the maximum Bayes factor
        # to that of the second maximum
        # for bin in range(num_bin):
        #     oo = np.argsort(-bayes_factors[bin, :])
        #     bayes_factors[bin, oo[0]] = bayes_factors[bin, oo[1]]

        for i in range(nc):
            # if j-th entry in i-th ranking falls into k-th bin,
            # then bayes_data[j, i] should be the bayes factor of k-th
            # bin in i-th ranking
            bayes_data[:, i] = bayes_factors[binned_data[:, i]-1, i]

        # bb = np.exp(np.sum(bayes_data, 1))
        # f = prior / (1. - prior)
        # prob = bb * f / (1. + bb * f)
        # exp = np.sort(-prob)[nrp]

        guess = stats.rankdata(-np.sum(bayes_data, 1))
        cprev = stats.pearsonr(guess, guess_last)[0]
        #print ('Correlation with previous iteration: %f'% cprev)
    if return_all:
        return guess, bayes_data, bayes_factors
    else:
        return guess.astype(int)




def agg_rank(n_nodes,ranked_lists, agg_type, target_list, num_top_nodes=10):
    # ranked lists: list of ranked lists of objects high priority to low priority
    
    
    
    if agg_type == "crank":
        # rows have to be 1 to 74, while columns are different subjects
        # each element have to be a value while higher value getting higher priority
        #print(ranked_lists)
        data = np.array(ranked_lists).T 
        rank_indices = list(rra(data)) # smaller value, the better
        all_nodes = list(range(1,n_nodes+1)) # [1,2,3,...n_nodes]
        for node in target_list:
            all_nodes.remove(node)
        #print(len(all_nodes), len(rank_indices))
        nodes_ranked = [x for i, x in sorted(zip(rank_indices, all_nodes))]
        top_rank_nodes = nodes_ranked[:num_top_nodes]
        #print(top_rank_nodes)
    
    else: # all the remaining type of aggregators. 
        agg = ra.RankAggregator()

        if agg_type=="average":
            rank_dict = agg.average_rank(ranked_lists)
        elif agg_type=="borda":
            rank_dict = agg.borda(ranked_lists)
        elif agg_type=="dowdall":
            rank_dict = agg.dowdall(ranked_lists)
        else:
            print("agg_type: average, borda, dowdall")
        # convert it to dictionary since it was just a list of tuples when agg.borda was applied
        rank_dict = dict(rank_dict)
        if(agg_type=='average'):
            sorted_rank = dict( sorted(rank_dict.items(), key=operator.itemgetter(1),reverse=False)) # l2hs priority    
        else:
	        sorted_rank = dict( sorted(rank_dict.items(), key=operator.itemgetter(1),reverse=True)) # h2l priority

        top_rank_nodes  = list(sorted_rank.keys())[:num_top_nodes]
    
    return top_rank_nodes
    
    
def get_ranked_lists(fpath, mat_files, centrality, target_list,A_type='DCM'):
    
    all_ranked_lists = []
    for i in range(len(mat_files)):
        if(A_type=='DCM'):
            A = loadmat(fpath+mat_files[i])["A_sparse"]
            A=A.T
        if(A_type=='MOU'):
            A = np.loadtxt(fpath+'/J_'+mat_files[i]+'.txt')
    		
        if centrality=="pq":
            sorted_d = rank_pq(A) # should be A.t according to Barabasi 2011
        elif centrality=="out_degree":
            sorted_d = rank_Degree(A,rankorder='decreasing') # A should be orginal matrix
        elif centrality=="ratio_degree":
            sorted_d = rank_weight_OutbyIn_Degree(A)
        elif centrality=="page_rank":
            sorted_d = rank_pageRank(A)
        else:
        	STOP
     
            
        ranked_list = list(sorted_d.keys())
        for node in target_list: 
            ranked_list.remove(node)

        all_ranked_lists.append( ranked_list ) # keys are the node numbers
        
    return all_ranked_lists
    
    
    
def get_ranked_lists_with_values(fpath, mat_files, centrality, target_list,A_type='DCM'):
    # this only works for Crank
    
    all_ranked_lists = []
    # if we are doing crank, we need to pass the centrality values of 1 to 74 nodes instead
    for i in range(len(mat_files)):
        if(A_type=='DCM'):
    	    A = loadmat(fpath+mat_files[i])["A_sparse"]
    	    A=A.T
        if(A_type=='MOU'):
	        A = np.loadtxt(fpath+'/J_'+mat_files[i]+'.txt')


        if centrality=="pq":
            vals = list(pq_values(A)) # should be A.t according to Barabasi 2011
        elif centrality=="out_degree":
            vals = list(np.array(Degree_values(A))) # low out degree given high priority by multiplying by minus
        elif centrality=="ratio_degree":
            vals = list(weight_OutbyIn_Degree_values(A))
        elif centrality=="page_rank":
            vals = list(np.array(pageRank_values(A))*-1)
        else:
        	STOP

        # remove all the vals belonging to target nodes
        indices = [node-1 for node in target_list]
        delete_multiple_element(vals, indices)
        all_ranked_lists.append( vals ) # keys are the node numbers

    return all_ranked_lists
    
def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)
    
    
            
def get_statistics(fpath,mat_files, region_dict, region_name, centrality, top_nodes=10, A_type='DCM'):
    info_dict = {}
    
    print("Region name, centrality: %s, %s"%(region_name,centrality))
    info_dict["region_name"]=region_name #
    
    target_list = target_nodes(region_dict, region_name)
    #print("Target nodes: ", target_list )
    info_dict["target_nodes"]=target_list #
    info_dict["centrality"] = centrality #
    info_dict["num_top_nodes"] = top_nodes #
    
    #all_ranked_lists = get_ranked_lists(fpath, mat_files, centrality, target_list,A_type=A_type)
    
    if(centrality=="singlenode"):
        f=open("./target/singleNode_nodes.json")          #singleNode_nodes.json")
        data = json.load(f)
        all_ranked_lists=data[region_name]
		#print(all_ranked_lists)
        
    else:
        all_ranked_lists = get_ranked_lists(fpath, mat_files, centrality, target_list,A_type="DCM")
		#print(all_ranked_lists)
          
    
    emin_centr = []  # different top nodes for different A for the same centrality
    N = len(loadmat(fpath+mat_files[0])["A_sparse"].T)# N=74 in our case

    C = compute_C(N, target_list)# output target matrix
    for i in range(len(mat_files)):
        if(A_type=='DCM'):
            A = loadmat(fpath+mat_files[i])["A_sparse"]
            A=A.T
        if(A_type=='MOU'):
	        A = np.loadtxt(fpath+'/J_'+mat_files[i]+'.txt')

        control_nodes = all_ranked_lists[i][:top_nodes]
        B = compute_B(N, sorted(control_nodes) ) 
        # compute W
        W = solve_continuous_lyapunov(A.T, -np.matmul(B, B.T)) 

        W_c = np.matmul(C, np.matmul(W, C.T))
        emin = min(np.linalg.eigvals(W_c))**-1
        emin_centr.append(emin)
    
    Avg_emin_centr = np.mean(emin_centr)
    info_dict["avg_emin_centr"]=Avg_emin_centr #
    
    
    
    #print("Avg. min. energy(top %d control nodes different each case): %.3f"%(top_nodes,Avg_emin_centr ) )
    #print("")
    
    info_dict["rank_agg"] = {}
    
    for agg_type in ["average", "borda", "dowdall","crank"]:
        #print("rank aggregation type:", agg_type)
        
        if agg_type=="crank":
            if(centrality=="singlenode"):
                f=open("./target/singleNode_energy.json")  
                data = json.load(f)
                all_ranked_values=data[region_name]
            else:
                all_ranked_values = get_ranked_lists_with_values(fpath, mat_files, centrality, target_list,A_type=A_type)

            agg_control_nodes = agg_rank(N,all_ranked_values, agg_type=agg_type, 
                                         target_list=target_list, num_top_nodes=top_nodes) 

        else:
            agg_control_nodes = agg_rank(N,all_ranked_lists, agg_type=agg_type,
                                         target_list=target_list, num_top_nodes=top_nodes)

        emin_agg = []
        N = len(loadmat(fpath+mat_files[0])["A_sparse"].T)# N=74 in our case

        C = compute_C(N, target_list)# output target matrix
        B = compute_B(N, sorted(agg_control_nodes) )
        for i in range(len(mat_files)):
            if(A_type=='DCM'):
    	        A = loadmat(fpath+mat_files[i])["A_sparse"]
    	        A=A.T
            if(A_type=='MOU'):
                A = np.loadtxt(fpath+'/J_'+mat_files[i]+'.txt')

            # compute W
            W = solve_continuous_lyapunov(A.T, -np.matmul(B, B.T)) 

            W_c = np.matmul(C, np.matmul(W, C.T))
            emin = min(np.linalg.eigvals(W_c))**-1
            emin_agg.append(emin)
        
        Avg_emin_agg = np.mean(emin_agg)
        #print("Aggregated control nodes:", agg_control_nodes)
        #print("Avg. min. energy(top %d control nodes = agg. control nodes):%.3f"%(top_nodes, Avg_emin_agg ) )
        info_dict["rank_agg"][agg_type] = {}
        info_dict["rank_agg"][agg_type]["agg_control_nodes"] = agg_control_nodes
        info_dict["rank_agg"][agg_type]["avg_emin_agg"] = Avg_emin_agg
        
        
        #percent_dict={}
        #print("Percentage of agg. control nodes in each cluster:")
        #all_region_names = set(region_dict.values())

        #for name in all_region_names:
        #    targets_of_name = target_nodes(region_dict, name)
        #    common = set( targets_of_name ).intersection(set(agg_control_nodes))
            #percent = np.round((len(common)/len(targets_of_name)) * 100,2)
        #    percent = np.round((len(common)/top_nodes) * 100,2)
            
            #print("%s : %.2f"%(name, percent))
        #    percent_dict[name] = percent
        #info_dict["rank_agg"][agg_type]["percent_agg_nodes"] = percent_dict
        
        #target_list = target_nodes(region_dict, region_name)
    
    
    #print("\n")
    return info_dict
        
        
        
   
        
    

