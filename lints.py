import numpy as np 
from scipy.special import softmax

class LINTS():
	def __init__(self, dimension, iteration, item_num, user_feature, item_feature, true_payoffs, alpha, delta, sigma, v):
		self.dimension=dimension
		self.iteration=iteration 
		self.item_num=item_num
		self.user_feature=user_feature
		self.item_feature=item_feature
		self.true_payoffs=true_payoffs
		self.alpha=alpha
		self.delta=delta
		self.sigma=sigma
		self.v=v
		self.cov=self.alpha*np.identity(self.dimension)
		self.bias=np.zeros(self.dimension)
		self.user_f=np.zeros(self.dimension)


	def select_arm(self, time):
		est_y_list=np.zeros(self.item_num)
		cov_inv=np.linalg.pinv(self.cov)
		sample_theta=np.random.multivariate_normal(mean=self.user_f, cov=self.v*cov_inv)
		for i in range(self.item_num):
			x=self.item_feature[i]
			est_y_list[i]=np.dot(sample_theta, x)

		index=np.argmax(est_y_list)
		x=self.item_feature[index]
		noise=np.random.normal(scale=self.sigma)
		payoff=self.true_payoffs[index]+noise 
		regret=np.max(self.true_payoffs)-self.true_payoffs[index]
		return x, payoff, regret, index

	def update_feature(self, x,y):
		self.cov+=np.outer(x,x)
		self.bias+=x*y
		self.user_f=np.dot(np.linalg.pinv(self.cov), self.bias)

	def run(self):
		cum_regret=[0]
		error=np.zeros(self.iteration)
		for time in range(self.iteration):
			print('time=%s/%s ~~~~~~LinTS'%(time, self.iteration))
			x,y,regret, index=self.select_arm(time)
			self.update_feature(x,y)
			cum_regret.extend([cum_regret[-1]+regret])
			error[time]=np.abs(self.true_payoffs[index]-np.dot(self.user_f, x))
		return cum_regret[1:], error








