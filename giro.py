import numpy as np 

class GIRO():
	def __init__(self, dimension, iteration, item_num, user_feature, item_features, true_payoffs, alpha, sigma, a):
		self.dimension=dimension
		self.iteration=iteration 
		self.item_num=item_num
		self.user_feature=user_feature
		self.item_features=item_features
		self.true_payoffs=true_payoffs
		self.alpha=alpha
		self.sigma=sigma
		self.a=a
		self.cov=self.alpha*np.identity(self.dimension)
		self.bias=np.zeros(self.dimension)
		self.user_f=np.zeros(self.dimension)

	def select_arm(self, time):
		est_y_list=np.dot(self.item_features, self.user_f)
		if time<=self.dimension*10:
			index=np.random.choice(range(self.item_num))
		else:
			index=np.argmax(est_y_list)
		# print('index', index)
		payoff=self.true_payoffs[index]+np.random.normal(scale=self.sigma)+np.random.normal(scale=self.a)
		regret=np.max(self.true_payoffs)-self.true_payoffs[index]
		return index, payoff, regret

	def update_feature(self, index, y):
		x=self.item_features[index]
		self.cov+=np.outer(x,x)
		self.bias+=y*x
		self.user_f=np.dot(np.linalg.pinv(self.cov), self.bias)
		# print('self.user_f', self.user_f)

	def run(self):
		cum_regret=[0]
		error=np.zeros(self.iteration)
		for time in range(self.iteration):
			print('time/iteration, %s/%s, ~~~~~~~ Giro'%(time, self.iteration))
			ind, y, regret=self.select_arm(time)
			self.update_feature(ind, y)
			cum_regret.extend([cum_regret[-1]+regret])
			error[time]=np.linalg.norm(self.user_f-self.user_feature)
		return cum_regret[1:], error 



