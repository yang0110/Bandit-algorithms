import numpy as np 

class SupCB_GLM():
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
		self.cov=self.alpha*np.identity(self.dimension)
		self.bias=np.zeros(self.dimension)
		self.user_f=np.zeros(self.dimension)
		self.item_index=[]
		self.est_y_list=np.zeros(self.item_num)
		self.width_list=np.zeros(self.item_num)
		self.upper_bound_list=np.zeros(self.item_num)
		self.item_set=list(range(self.item_num))
		self.upper_matrix=np.zeros((self.item_num, self.iteration))
		self.lower_matrix=np.zeros((self.item_num, self.iteration))


	def initial(self):
		for i in range(self.S+1):
			self.time_sets[i]=[]

	def base_linucb(self, time):
		time_set=self.time_sets[self.s]
		self.cov=self.alpha*np.identity(self.dimension)
		self.bias=np.zeros(self.dimension)
		self.user_f=np.zeros(self.dimension)
		if len(time_set)==0:
			self.user_f=np.zeros(self.dimension)
			cov_inv=np.linalg.pinv(self.cov)
		else:
			for t in time_set:
				ind=self.item_index[t]
				x=self.item_features[ind]
				noise=np.random.normal(scale=self.sigma)
				y=self.true_payoffs[ind]+noise 
				self.cov+=np.outer(x,x)
				self.bias+=y*x 
			cov_inv=np.linalg.pinv(self.cov)
			self.user_f=np.dot(cov_inv, self.bias)

		self.est_y_list=np.zeros(self.item_num)
		self.width_list=np.zeros(self.item_num)
		for i in self.item_set:
			x=self.item_features[i]
			self.est_y_list[i]=np.dot(x, self.user_f)
			self.width_list[i]=self.beta*np.sqrt(np.dot(np.dot(x, cov_inv),x))

		self.upper_bound_list=self.est_y_list+self.width_list
		self.upper_matrix[:,time]=self.est_y_list+self.width_list
		self.lower_matrix[:,time]=self.est_y_list-self.width_list

	def select_arm(self,time):
		index=None
		while index==None:
			self.base_linucb(time)
			if np.max(self.width_list)<=1/np.sqrt(self.iteration):
				index=np.argmax(self.upper_bound_list)
			elif np.max(self.width_list)<=2**(-self.s):
				# max_lower_bound=np.max(self.upper_bound_list)-2**(1-self.s)
				temp_item_set=self.item_set.copy()
				self.item_set=[]
				for i in temp_item_set:
					if self.est_y_list[i]>=np.max(self.est_y_list)-2**(1-self.s):
						self.item_set.extend([i])
				self.s+=1
			else:
				candidate_index=np.where(self.width_list>2**(-self.s))[0]
				index=np.random.choice(candidate_index)
				self.time_sets[self.s].extend([time])
		return index

	def find_regret(self, index, time):
		regret=np.max(self.true_payoffs)-self.true_payoffs[index]
		self.item_index.extend([index])
		return regret

	def run(self):
		self.initial()
		cum_regret=[0]
		error_list=np.zeros(self.iteration)
		for time in range(self.iteration):
			print('time/iteration, %s/%s, item_num=%s, s=%s ~~~~~ SupCB-GLM'%(time, self.iteration, len(self.item_set), self.s))
			item_index=self.select_arm(time)
			regret=self.find_regret(item_index, time)
			error=np.abs(self.true_payoffs[item_index]-np.dot(self.user_f, self.item_features[item_index]))
			cum_regret.extend([cum_regret[-1]+regret])
			error_list[time]=error
		return cum_regret[1:], error_list, self.upper_matrix, self.lower_matrix























