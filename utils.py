import numpy as np 
import networkx as nx
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import scipy
import os 
from sklearn import datasets

def generate_data(dimension, item_num):
	item_f=np.random.uniform(low=-1, high=1, size=(item_num, dimension))
	item_f=Normalizer().fit_transform(item_f)
	user_f=np.random.uniform(low=-1, high=1, size=dimension)
	user_f=user_f/np.linalg.norm(user_f)
	return user_f, item_f


def create_networkx_graph(node_num, adj_matrix):
	G=nx.Graph()
	G.add_nodes_from(list(range(node_num)))
	for i in range(node_num):
		for j in range(node_num):
			if adj_matrix[i,j]!=0.0:
				G.add_edge(i,j,weight=adj_matrix[i,j])
			else:
				pass
	return G, G.number_of_edges()


def RBF_graph(node_num, dimension, gamma=None, thres=None, clusters=False): ##
	if clusters==False:
		node_f=np.random.uniform(low=-0.5, high=0.5, size=(node_num, dimension))
	else:
		node_f, _=datasets.make_blobs(n_samples=node_num, n_features=dimension, centers=5, cluster_std=0.1, center_box=(-1,1),  shuffle=False, random_state=2019)
	if gamma==None:
		gamma=0.5
	else:
		pass 
	adj=rbf_kernel(node_f, gamma=gamma)
	adj=np.round(adj, decimals=2)
	if thres!=None:
		adj[adj<=thres]=0.0
	else:
		pass
	np.fill_diagonal(adj,1)
	return adj 

def ER_graph(node_num, prob):## unweighted
	G=nx.erdos_renyi_graph(node_num, prob)
	adj=nx.to_numpy_matrix(G)
	adj=np.asarray(adj)
	#adj=adj+np.identity(node_num)
	return adj 

def BA_graph(node_num, edge_num): #unweighted 
	G=nx.barabasi_albert_graph(node_num, edge_num)
	adj=nx.to_numpy_matrix(G)
	adj=np.asarray(adj)
	#adj=adj+np.identity(node_num)
	return adj 
 

def WS_graph(node_num, neighbor_num, prop):
	G=nx.watts_strogatz_graph(node_num, neighbor_num, prop)
	adj=nx.to_numpy_matrix(G)
	adj=np.asarray(adj)
	return adj


def graph_signal_samples_from_laplacian(laplacian, node_num, dimension):
	cov=np.linalg.pinv(laplacian)
	samples=np.random.multivariate_normal(np.zeros(node_num),cov, size=dimension).T
	return samples 

def dictionary_matrix_generator(node_num, dimension, laplacian, lambda_):
	np.random.seed(2018)
	D0=np.random.normal(size=(node_num, dimension))
	D=np.dot(np.linalg.pinv(np.identity(node_num)+lambda_*laplacian), D0)
	D=Normalizer().fit_transform(D)
	return D


def normalized_trace(matrix, target_trace):
	normed_matrix=target_trace*matrix/np.trace(matrix)
	return normed_matrix


def calculate_graph_approximation(i, dimension, user_num, user_index, alpha, normed_lap, user_v_i, user_f_matrix_ls, user_f_matrix_ridge):
	ridge=user_f_matrix_ridge[user_index]
	avg=np.dot(user_f_matrix_ls.T, -normed_lap[user_index])+normed_lap[user_index, user_index]*user_f_matrix_ls[user_index]
	graph=ridge+alpha*np.dot(np.linalg.pinv(user_v_i), avg)
	return graph 


def calculate_graph_approximation_2(i, dimension, user_num, user_index, alpha, normed_lap, user_v_i, user_f_matrix_ls, user_f_matrix_ridge):
	ridge=user_f_matrix_ridge[user_index]
	if i>=1*user_num:
		avg=np.dot(user_f_matrix_ls.T, -normed_lap[user_index])+user_f_matrix_ls[user_index]
		graph=ridge+alpha*np.dot(np.linalg.pinv(user_v_i), avg)
	else:
		graph=ridge
	return graph 


def modify_normed_lap(normed_lap):
	for i in range(normed_lap.shape[0]):
		if normed_lap[i,i]==0.0:
			normed_lap[i,i]=1
		else:
			pass
	return normed_lap

def graph_error_bound(user_index, user_v_i, normed_lap, user_f_matrix_ls, user_f_matrix_graph_i, alpha, xnoise_i):
	a=np.trace(np.linalg.pinv(user_v_i))
	avg=np.dot(user_f_matrix_ls.T, -normed_lap[user_index])+user_f_matrix_ls[user_index]
	b=alpha*np.linalg.norm(user_f_matrix_graph_i-avg)
	c=np.linalg.norm(xnoise_i)
	bound=a*(b+c)
	return bound 

def ridge_error_bound(user_v_i, user_f_matrix_ridge_i, alpha, xnoise_i):
	a=np.trace(np.linalg.pinv(user_v_i))
	b=alpha*np.linalg.norm(user_f_matrix_ridge_i)
	c=np.linalg.norm(xnoise_i)
	bound=a*(b+c)
	return bound 


def graph_UCB(dimension,sigma, delta, user_index, user_v_i, normed_lap, user_f_matrix_ls, user_f_matrix_graph_i, alpha, xnoise_i):
	a=alpha*np.sqrt(np.trace(np.linalg.pinv(user_v_i)))
	avg=np.dot(user_f_matrix_ls.T, -normed_lap[user_index])+user_f_matrix_ls[user_index]
	b=np.linalg.norm(user_f_matrix_graph_i-avg)
	#v_inv=np.linalg.pinv(user_v_i)
	#c=np.sqrt(np.dot(np.dot(xnoise_i,v_inv), xnoise_i))
	c1=np.linalg.det(user_v_i)**(1/2)
	c2=np.linalg.det(alpha*np.identity(dimension))**(-1/2)
	c=sigma*np.sqrt(2*np.log(c1*c2/delta))
	ucb=a*b+c
	return ucb,b


def ridge_UCB(dimension, sigma, delta, user_v_i, user_f_matrix_ridge_i, alpha, xnoise_i):
	a=alpha*np.sqrt(np.trace(np.linalg.pinv(user_v_i)))
	b=np.linalg.norm(user_f_matrix_ridge_i)
	#v_inv=np.linalg.pinv(user_v_i)
	#c=np.sqrt(np.dot(np.dot(xnoise_i,v_inv), xnoise_i))
	c1=np.linalg.det(user_v_i)**(1/2)
	c2=np.linalg.det(alpha*np.identity(dimension))**(-1/2)
	c=sigma*np.sqrt(2*np.log(c1*c2/delta))
	ucb=a*b+c
	return ucb,b

def ridge_UCB_old(dimension, sigma, delta, user_v_i, user_f_matrix_ridge_i, alpha, xnoise_i):
	a=np.sqrt(alpha)*np.linalg.norm(user_f_matrix_ridge_i)
	#v_inv=np.linalg.pinv(user_v_i)
	#c=np.sqrt(np.dot(np.dot(xnoise_i,v_inv), xnoise_i))
	c1=np.linalg.det(user_v_i)**(1/2)
	c2=np.linalg.det(alpha*np.identity(dimension))**(-1/2)
	c=sigma*np.sqrt(2*np.log(c1*c2/delta))
	ucb=a+c
	return ucb 


def graph_true_UCB(user_index, user_v_i, normed_lap, user_f_matrix_ls, user_f_matrix_graph_i, alpha, xnoise_i):
	avg=np.dot(user_f_matrix_ls.T, -normed_lap[user_index])+user_f_matrix_ls[user_index]
	a=alpha*(user_f_matrix_graph_i-avg)-xnoise_i
	v_inv=np.linalg.pinv(user_v_i)
	ucb=np.sqrt(np.dot(np.dot(a,v_inv), a))
	return ucb 


def ridge_true_UCB(user_v_i, user_f_matrix_ridge_i, alpha, xnoise_i):
	a=alpha*(user_f_matrix_ridge_i)-xnoise_i
	v_inv=np.linalg.pinv(user_v_i)
	ucb=np.sqrt(np.dot(np.dot(a,v_inv), a))
	return ucb 