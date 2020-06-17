from scipy.special import softmax
import numpy as np 


class EXP3():
	def __init__(self, dimension, iteration, item_num, user_feature, item_features, true_payoffs, alpha, sigma, gamma, eta):
		self.dimension=dimension
		self.iteration=iteration 
		self.item_num=item_num
		self.user_feature=user_feature
		self.item_features=item_features
		self.true_payoffs=true_payoffs
		self.alpha=alpha
		self.sigma=sigma
		self.gamma=gamma
		self.eta=eta
		self.cov=self.alpha*np.identity(self.dimension)
		self.bias=np.zeros(self.dimension)
		self.user_f=np.zeros(self.dimension)
		self.prob_list=np.zeros(self.item_num)
		self.uniform_prob_list=np.zeros(self.item_num)
		self.s_list=np.ones(self.item_num)
		self.s_prob_list=np.zeros(self.item_num)
		self.prob_matrix=np.zeros((self.iteration, self.item_num))
		self.est_y_list=np.zeros(self.item_num)
		self.est_y_matrix=np.zeros((self.iteration, self.item_num))
		self.uniform_prob_list=(1/self.item_num)*np.ones(self.item_num)
		self.item_count=np.ones(self.item_num)

	def select_arm(self, time):
		# self.eta=0.1
		self.prob=self.eta*self.uniform_prob_list+(1-self.eta)*self.s_prob_list
		self.prob_matrix[time]=self.prob
		index=np.random.choice(range(self.item_num), p=self.prob)
		self.item_count[index]+=1
		payoff=self.true_payoffs[index]+np.random.normal(scale=self.sigma)
		regret=np.max(self.true_payoffs)-self.true_payoffs[index]
		return index, payoff, regret

	def update_feature(self, index, y, time):
		x=self.item_features[index]
		self.cov+=np.outer(x,x)
		self.bias+=y*x
		self.user_f=np.dot(np.linalg.pinv(self.cov), self.bias)
		self.est_y_list=np.dot(self.item_features, self.user_f)
		self.est_y_matrix[time]=self.est_y_list
		self.s_list=np.sum(self.est_y_matrix[:time], axis=0)/self.item_count
		x_norm_list=np.zeros(self.item_num)
		cov_inv=np.linalg.pinv(self.cov)
		# for i in range(self.item_num):
		# 	x=self.item_features[i]
		# 	x_norm_list[i]=np.sqrt(np.dot(np.dot(x, cov_inv), x))

		# self.uniform_prob_list=softmax(x_norm_list)



	def update_s_list(self):
		self.s_prob_list=softmax(self.gamma*self.s_list)

	def run(self):
		cum_regret=[0]
		error=np.zeros(self.iteration)
		for time in range(self.iteration):
			print('time/iteration, %s/%s, est=%s ~~~~~~~ EXP3'%(time, self.iteration, np.round(self.eta)))
			self.update_s_list()
			index, payoff, regret=self.select_arm(time)
			self.update_feature(index, payoff, time)
			cum_regret.extend([cum_regret[-1]+regret])
			error[time]=np.linalg.norm(self.user_f-self.user_feature)
		return cum_regret[1:], error, self.prob_matrix.T



















