import numpy as np 
from scipy.special import softmax

class LINPHE():
	def __init__(self, dimension, iteration, item_num, user_feature, item_feature, true_payoffs, alpha, lambda_, sigma):
		self.dimension=dimension
		self.iteration=iteration 
		self.item_num=item_num
		self.user_feature=user_feature
		self.item_feature=item_feature
		self.true_payoffs=true_payoffs
		self.alpha=alpha
		self.lambda_=lambda_
		self.sigma=sigma
		# self.cov=self.alpha*np.identity(self.dimension)
		self.cov=np.zeros((self.dimension, self.dimension))
		self.random_cov=np.zeros((self.dimension, self.dimension))
		self.bias=np.zeros(self.dimension)
		self.random_bias=np.zeros((self.item_num, self.dimension))
		self.user_f=np.zeros(self.dimension)
		self.item_count=np.zeros(self.item_num)
		self.random_item_features=np.zeros((self.item_num, self.dimension))
		self.item_reward_list=np.zeros(self.item_num)

	def generate_random_reward(self):
		self.random_item_features=np.zeros((self.item_num, self.dimension))
		self.random_bias=np.zeros((self.item_num, self.dimension))
		for i in range(self.item_num):
			number_of_perturbation=np.ceil(self.lambda_*self.item_count[i])
			self.random_bias[i]=self.item_feature[i]*(np.random.binomial(number_of_perturbation, 0.5)+self.item_reward_list[i])
			self.random_item_features[i]=self.item_feature[i]*number_of_perturbation

	def select_arm(self, time):
		est_y_list=np.zeros(self.item_num)
		for i in range(self.item_num):
			x=self.item_feature[i]
			est_y_list[i]=np.dot(self.user_f, x)

		max_index=np.argmax(est_y_list)
		self.item_count[max_index]+=1
		x=self.item_feature[max_index]
		noise=np.random.normal(scale=self.sigma)
		payoff=self.true_payoffs[max_index]+noise 
		regret=np.max(self.true_payoffs)-payoff+noise
		self.item_reward_list[max_index]+=payoff
		return x, payoff, regret

	def update_feature(self,x,y):
		self.cov+=np.outer(x,x)
		# self.bias+=x*y
		self.random_cov=(self.lambda_+1)*self.cov+(self.lambda_+1)*np.identity(self.dimension)
		# self.random_cov=np.zeros((self.dimension, self.dimension))
		# for i in range(self.item_num):
			# self.random_cov+=np.outer(self.random_item_features[i], self.random_item_features[i])
		# self.new_bias=np.sum(self.random_bias, axis=0)+self.bias
		# self.new_cov=self.random_cov+self.cov+self.lambda_*np.identity(self.dimension)
		self.user_f=np.dot(np.linalg.pinv(self.random_cov), np.sum(self.random_bias, axis=0))

	def run(self):
		cum_regret=[0]
		error=np.zeros(self.iteration)
		for i in range(self.item_num):
			regret=np.max(self.true_payoffs)-self.true_payoffs[i]
			cum_regret.extend([cum_regret[-1]+regret])
			error[i]=np.linalg.norm(self.user_f-self.user_feature)
			self.item_count[i]+=1
		for time in range(self.iteration-self.item_num):
			time=time+self.item_num
			print('time=%s/%s ~~~~~~LinPHE'%(time, self.iteration))
			self.generate_random_reward()
			x,y,regret=self.select_arm(time)
			self.update_feature(x,y)
			cum_regret.extend([cum_regret[-1]+regret])
			error[time]=np.linalg.norm(self.user_f-self.user_feature)
		return cum_regret[1:], error








