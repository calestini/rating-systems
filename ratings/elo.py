# -*- coding: utf-8 -*-
import numpy as np
import math

from .exceptions import InvalidSDMedhod



class Elo(object):

	"""
	A python object to calculate Elo ratings

	Useful documentation on elo ratings:
		- http://www.eloratings.net/about
		- https://en.wikipedia.org/wiki/Elo_rating_system

	"""

	def __init__(self, start_rating=1500, sd_method='logistic', h=0, m=1, width=400, K=20, **kwargs):
		"""
		Params:
		-------
			- start_rating (integer): where the ratings start. Default: 1,500.
			- sd_method (string)[logistic, power, fte]: score difference method.
			- h (float): home-team advantage, defaults to 0
			- m (float): multiplier for score difference - defaults to 1, is
						not used with 'Five_thirty_eight' method
			- width (float): defaults to 400.
			- K (float): constant used to gauge magnitude of effect for each
						match outcome
		"""
		self.start_rating = start_rating
		self.sd_method = sd_method
		self.h = h
		self.m = m
		self.K = K
		self.width = width

		if sd_method.lower() not in ['logistic','power','fte']:
			raise InvalidSDMedhod(f'Method {sd_method} does not exist')
		self._sd_function = getattr(self, f'_score_diff_{sd_method.lower()}')


	def probability(self, rating1, rating2):
		"""
		Calculates the probability given two elo ratings.
		Params:
		-------
			- rating1: home team rating
			- rating2: visitor team rating
		"""
		p1_w = 1.0 * 1.0 / \
			(1 + 1.0 * math.pow(10,1.0*(rating2-(rating1 + self.h))/self.width))

		return p1_w, 1 - p1_w

	def update_elo(self):

		self.p_home, self.p_vis = self.probability(self.home_prev,self.vis_prev)

		hfactor = 1. - abs(self.d-1.)
		vfactor = abs(hfactor-1)

		self.home_post = self.home_prev + self.K * self.sd * (hfactor - self.p_home)
		self.vis_post = self.vis_prev + self.K * self.sd * (vfactor - self.p_vis)

		self.returns = [
			'phome','pvis',
			'hpost','vpost'
		]

		return self.p_home, self.p_vis, self.home_post, self.vis_post


	def rate(self, home_prev, vis_prev, outcome, score_diff=0):
		"""
		Elo rating calculation.

		Params:
		-------
			- home_prev: home team initial ratings
			- vis_prev: vis team initial ratings
			- d (float): outcome (1,0.5,0)
			- start_rating: start_rating
			- score_diff (float): score difference
		"""
		self.home_prev = home_prev
		self.vis_prev = vis_prev
		self.d = outcome
		self.score_diff = score_diff
		# Calculate winning probabilities

		### apply score difference function
		self._sd_function()

		return self.update_elo()


	def _weighted_r_diff(self, r_chng, a):
		"""
		It returns the ratings change for the current matchup based on the
		history of rating change since beginning of period. So the new rating
		will add this change to <start_rating> to know the current rating.
		The r_chng vector includes the current matchup change. It seems to
		smoothen the ratings changes up until now.

		Note:
			- Perhaps this should be moved to the rate function as it is only
		applicable for multiple observations;

		Parameters:
		-----------
			- r_chng (vector / list / array): ratings changes over time
			- a (float): time_scale_factor
		Return:
		-------
			- ratings change (float)
		"""
		length = len(r_chng)
		ordered_arr = np.arange(1, length +1)

		weights = np.power(((1 + ordered_arr - 1) / (1 + length - 1)), a)
		# print (weights)

		return np.sum(weights * r_chng)


	def _ewma(r_chng, alpha):
		'''
		Calculates the exponential moving average over a vector.
		Params:
		-------
			- r_chng: historical change in ratings
			- alpha: The alpha parameter for the moving average (0,1)
		'''
		alpha = np.array(alpha, copy=False)
		scaling_factors = np.power(1. - alpha, np.arange(r_chng.size + 1))

		# create cumulative sum array
		res = np.multiply(r_chng,(alpha*scaling_factors[-2])/scaling_factors[:-1])

		# print (res)
		res = np.cumsum(res)
		# cumsums / scaling
		res /= scaling_factors[-2::-1]

		return res


	def _score_diff_logistic(self):
		"""
		logistic function that intersects 1 at 1
		"""
		self.sd = 2/(1+np.exp(-self.m * (self.score_diff - 1)))


	def _score_diff_fte(self):
		"""
		Margin of victory multiplier borrowed from FiveThirtyEight
		Params:
		-------
			- ELOW: elo rating of favorite
			- ELOL: elo rating underdog
		"""
		if self.d == 1:
			ELOW = self.home_prev
			ELOL = self.vis_prev
		else:
			ELOW = self.vis_prev
			ELOL = self.home_prev

		self.sd = np.log(np.abs(self.score_diff)+1) * (2.2/((ELOW-ELOL)*.001+2.2))
		return self.sd


	def _score_diff_power(self):
		"""
		absolute score difference that is scaled by a power
		"""
		self.sd =  np.power((1 + self.score_diff), self.m)
		return self.sd
