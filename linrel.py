import numpy as np 
import scipy
from scipy.special import softmax

class LinRel():
	def __init__(self, dimension, iteration, item_num, user_feature, item_feature, true_payoffs, alpha, delta, sigma, beta):
		self.dimension=dimension
		self.iteration=iteration 
		self.item_num=item_num
		self.user_feature=user_feature
		self.item_feature=item_feature
		self.true_payoffs=true_payoffs
		self.alpha=alpha
		self.delta=delta
		self.sigma=sigma
		self.beta=beta
		self.cov=self.alpha*0.1*np.identity(self.dimension)
		self.bias=np.zeros(self.dimension)
		self.user_f=np.zeros(self.dimension)
		self.item_index=np.zeros(self.iteration)
		self.U=np.identity(self.dimension)
		self.V=np.diag(np.ones(self.dimension))
		self.x_matrix=np.zeros((self.item_num, self.dimension))
		self.u_matrix=np.zeros((self.item_num, self.dimension))
		self.v_matrix=np.zeros((self.item_num, self.dimension))
		self.s_list=np.zeros(self.item_num)
		self.D=np.zeros((self.iteration, self.dimension))
		self.payoffs=np.zeros(self.iteration)
		self.error_list=np.zeros(self.iteration)
		self.upper_matrix=np.zeros((self.item_num, self.iteration))
		self.lower_matrix=np.zeros((self.item_num, self.iteration))


	def update_beta(self, time):
		self.beta=np.sqrt(self.alpha)+self.sigma*np.sqrt(self.dimension*np.log(1+time/self.dimension)+2*np.log(1/self.delta))

	def select_arm(self, time):
		index=np.argmax(self.s_list)
		self.item_index[time]=index
		x=self.item_feature[index]
		noise=np.random.normal(scale=self.sigma)
		payoff=self.true_payoffs[index]+noise 
		regret=np.max(self.true_payoffs)-self.true_payoffs[index]
		self.D[time]=x
		self.payoffs[time]=payoff
		return x, payoff, regret, index

	def update_feature(self, time):
		eig_values, self.U=np.linalg.eig(self.cov)
		large_values=eig_values[eig_values>=1]
		k=len(large_values)
		small_values=eig_values[eig_values<1]
		inv_large_val=1/large_values
		temp=np.zeros(self.dimension)
		temp[:k]=inv_large_val
		A=np.diag(temp)
		self.s_list=np.zeros(self.item_num)
		for i in range(self.item_num):
			x=self.item_feature[i]
			x_tilde=np.dot(self.U.T, x)
			self.x_matrix[i]=x_tilde
			self.u_matrix[i][:k]=x_tilde[:k]
			self.v_matrix[i][k:]=x_tilde[k:]
			a=np.dot(self.u_matrix[i], A)
			b=np.dot(a, self.U.T)
			c=np.dot(b, self.D[:time].T)
			self.s_list[i]=np.dot(c,self.payoffs[:time])+np.linalg.norm(c)*self.beta+np.linalg.norm(self.v_matrix[i])
			self.upper_matrix[i,time]=self.s_list[i]
			self.lower_matrix[i,time]=np.dot(self.user_f, x)-np.linalg.norm(c)*self.beta-np.linalg.norm(self.v_matrix[i])

	def update_cov(self, x, y, index, time):
		self.cov=np.dot(self.D[:time].T, self.D[:time])+self.alpha*0.1*np.identity(self.dimension)
		# self.cov+=np.outer(x,x)
		self.bias+=y*x
		self.user_f=np.dot(np.linalg.pinv(self.cov), self.bias)
		self.error_list[time]=np.abs(self.true_payoffs[index]-np.dot(self.user_f, self.item_feature[index]))

	def run(self):
		cum_regret=[0]
		for time in range(self.iteration):
			print('time=%s/%s ~~~~~~LinRel'%(time, self.iteration))
			# self.update_beta(time)
			self.update_feature(time)
			x, y,regret, index=self.select_arm(time)
			self.update_cov(x, y, index, time)
			cum_regret.extend([cum_regret[-1]+regret])
		return cum_regret[1:], self.error_list, self.upper_matrix, self.lower_matrix









