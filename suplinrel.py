import numpy as np 
import scipy 

class SupLinRel():
	def __init__(self, dimension, iteration, item_num, user_feature, item_features, true_payoffs, alpha, delta, sigma, beta):
		self.dimension=dimension
		self.iteration=iteration
		self.item_num=item_num
		self.user_feature=user_feature
		self.item_features=item_features
		self.true_payoffs=true_payoffs
		self.alpha=alpha
		self.sigma=sigma
		self.delta=delta
		self.beta=beta
		self.S=np.int(np.log(self.iteration))
		self.s=1
		self.time_sets={}
		self.cov=self.alpha*0.1*np.identity(self.dimension)
		self.bias=np.zeros(self.dimension)
		self.user_f=np.zeros(self.dimension)
		self.item_index=[]
		self.est_y_list=np.zeros(self.item_num)
		self.width_list=np.zeros(self.item_num)
		self.upper_bound_list=np.zeros(self.item_num)
		self.item_set=list(range(self.item_num))
		self.upper_matrix=np.zeros((self.item_num, self.iteration))
		self.lower_matrix=np.zeros((self.item_num, self.iteration))
		self.U=np.identity(self.dimension)
		self.V=np.diag(np.ones(self.dimension))
		self.x_matrix=np.zeros((self.item_num, self.dimension))
		self.u_matrix=np.zeros((self.item_num, self.dimension))
		self.v_matrix=np.zeros((self.item_num, self.dimension))
		self.D=np.zeros((self.iteration,self.dimension))
		self.payoffs=np.zeros(self.iteration)
		self.noise_payoffs=np.zeros(self.iteration)

	def initial(self):
		for i in range(self.S+1):
			self.time_sets[i]=[]

	def base_linrel(self, time):
		time_set=self.time_sets[self.s]
		self.cov=self.alpha*0.1*np.identity(self.dimension)
		self.bias=np.zeros(self.dimension)
		self.user_f=np.zeros(self.dimension)
		if len(time_set)==0:
			self.user_f=np.zeros(self.dimension)
			cov_inv=np.linalg.pinv(self.cov)
		else:
			self.D=np.zeros((len(time_set), self.dimension))
			self.payoffs=np.zeros(len(time_set))
			for idx, t in enumerate(time_set):
				ind=self.item_index[t]
				x=self.item_features[ind]
				y=self.noise_payoffs[t]
				self.D[idx]=x
				self.payoffs[idx]=y
				self.cov=np.dot(self.D.T, self.D)+self.alpha*0.1*np.identity(self.dimension)
				self.bias+=y*x
			cov_inv=np.linalg.pinv(self.cov)
		self.user_f=np.dot(cov_inv, self.bias)

		eig_values, self.U=np.linalg.eig(self.cov)
		large_values=eig_values[eig_values>=1]
		k=len(large_values)
		small_values=eig_values[eig_values<1]
		inv_large_val=1/large_values
		temp=np.zeros(self.dimension)
		temp[:k]=inv_large_val
		A=np.diag(temp)
		self.est_y_list=np.zeros(self.item_num)
		self.width_list=np.zeros(self.item_num)
		for i in self.item_set:
			x=self.item_features[i]
			x_tilde=np.dot(self.U.T, x)
			self.x_matrix[i]=x_tilde
			self.u_matrix[i][:k]=x_tilde[:k]
			self.v_matrix[i][k:]=x_tilde[k:]
			a=np.dot(self.u_matrix[i], A)
			b=np.dot(a, self.U.T)
			c=np.dot(b, self.D.T)
			self.est_y_list[i]=np.dot(c, self.payoffs)
			self.width_list[i]=np.linalg.norm(c)*self.beta+np.linalg.norm(self.v_matrix[i])
		self.upper_bound_list=self.est_y_list+self.width_list
		self.upper_matrix[:,time]=self.est_y_list+self.width_list
		self.lower_matrix[:,time]=self.est_y_list-self.width_list


	def select_arm(self,time):
		index=None
		while index==None:
			self.base_linrel(time)
			if np.max(self.width_list)<=1/np.sqrt(self.iteration):
				index=np.argmax(self.upper_bound_list)
				self.item_index.extend([index])
				self.noise_payoffs[time]=self.true_payoffs[index]+np.random.normal(scale=self.sigma)
				print('a')
			elif np.max(self.width_list)<=2**(-self.s):
				max_lower_bound=np.max(self.upper_bound_list)-2**(1-self.s)
				temp_item_set=self.item_set.copy()
				self.item_set=[]
				for i in temp_item_set:
					if self.upper_bound_list[i]>=max_lower_bound:
						self.item_set.extend([i])
				self.s+=1
				print('b')
			else:
				candidate_index=np.where(self.width_list>2**(-self.s))[0]
				index=np.random.choice(candidate_index)
				self.item_index.extend([index])
				self.time_sets[self.s].extend([time])
				self.noise_payoffs[time]=self.true_payoffs[index]+np.random.normal(scale=self.sigma)
				print('c')
		return index

	def find_regret(self, index, time):
		regret=np.max(self.true_payoffs)-self.true_payoffs[index]
		return regret

	def run(self):
		self.initial()
		cum_regret=[0]
		error_list=np.zeros(self.iteration)
		for time in range(self.iteration):
			print('time/iteration, %s/%s, item_num=%s, s=%s ~~~~~ SupLinRel'%(time, self.iteration, len(self.item_set), self.s))
			index=self.select_arm(time)
			regret=self.find_regret(index, time)
			error=np.abs(self.true_payoffs[index]-np.dot(self.user_f, self.item_features[index]))
			cum_regret.extend([cum_regret[-1]+regret])
			error_list[time]=error
		return cum_regret[1:], error_list, self.upper_matrix, self.lower_matrix























