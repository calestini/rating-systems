# -*- coding: utf-8 -*-
import numpy as np
import math

# q = math.log(10) / 400
# c = 50
# start_rating = 1500
# Sigma = 350
# Sigma_min = 30

class Glicko(object):

	"""
	A python object to calculate Glicko 1 ratings.

	For more information on GLicko ratings:
		- http://www.glicko.net/glicko/glicko.pdf
		- https://en.wikipedia.org/wiki/Glicko_rating_system

	"""

	def __init__(self, start_rating=1500, c=50, q=None, sigma=350, sigma_min=30, h=0, width=400,):
		"""
		Params:
		-------
			- c: determines how much RD goes back up between periods of assessment, defaults to 50
			- start_rating: determines where the ratings start, defaults to 1500
			- sigma: the start RD for every team, defaults to 350
		"""
		self.start_rating = start_rating
		self.sigma = sigma
		self.sigma_min = sigma_min
		self.q = q or math.log(10) / 400
		self.c = c
		self.h = h
		self.width = width


	def onset_rd(self, rd_old, t):
		"""
		Calculate the onset ratings deviation at the beginning of a period
		"""
		return np.sqrt(np.power(rd_old, 2) + np.power(self.c, 2) * t)


	def g(self, rd):
		"""
		Calculate 'g' for the opponent.
		Params:
		-------
			- rd: ratings deviation;
		"""
		return 1 / np.sqrt((1 + 3 * np.power(self.q, 2) * np.power(rd, 2)) / np.power(np.pi, 2))


	def probability(self, g, home_prev, vis_prev):
		"""
		Calculate expected result
		"""
		return 1 / (1 + np.power(10, -g * ((home_prev - vis_prev) / self.width)))


	def rd_new(self, g, E, rd):
		"""
		Calculate post-match RD (ratings deviation)
		"""
		d = 1 / (np.power(self.q, 2) * np.power(g, 2) * E * (1- E))
		return 1 / np.sqrt(1 / np.power(rd, 2) + 1 / d)


	def glicko_rating(self, home_prev, vis_prev, rd_home, rd_vis, outcome):
		"""
		Calculate post-match rating
		"""
		rd_home = self.onset_rd(rd_home, 1)
		rd_vis = self.onset_rd(rd_vis, 1)

		visitor_outcome = abs(outcome - 1)

		g_b = self.g(rd_vis)

		p_home = self.probability(g_b, home_prev, vis_prev)

		if rda > self.sigma_min:
			rda_new = self.rd_new(g_b, p_home, rd_home)
		else:
			rda_new = rd_home

		home_post = home_prev + self.q * np.power(rda_new, 2) * g_b * (outcome - p_home)

		g_a = self.g(rd_home)

		p_vis = self.probability(g_a, vis_prev, home_prev)

		if rdb > Sigma_min:
			rdb_new = self.rd_new(g_b, p_vis, rd_vis)
		else:
			rdb_new = rd_vis

		rdb_new = self.rd_new(g_a, p_vis, rd_vis)
		vis_post = vis_prev + self.q * np.power(rdb_new, 2) * g_a * (visitor_outcome - p_vis)

		return p_home, p_vis, home_post, vis_post, rda_new, rdb_new
