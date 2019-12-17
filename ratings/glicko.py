# -*- coding: utf-8 -*-
import numpy as np
import math
import logging


class Glicko(object):

	"""
	A python object to calculate Glicko [1] ratings.

	For more information on GLicko ratings:
		- http://www.glicko.net/glicko/glicko.pdf
		- https://en.wikipedia.org/wiki/Glicko_rating_system

	"""

	def __init__(self, start_rating=1500, c=50,
							q=None, sigma=350, sigma_min=30, h=0, width=400):
		"""
		Params:
		-------
			- c: determines how much RD goes back up between periods of
				assessment, defaults to 50
			- start_rating: determines where the ratings start, defaults to 1500
			- sigma: the start RD for every team, defaults to 350
		"""
		self.start_rating = start_rating
		self.sigma = sigma
		self.sigma_min = sigma_min
		self.width = width
		self.q = q or math.log(10) / self.width
		self.c = c
		self.h = h


	def onset_rd(self, rd_old, t=1):
		"""
		Calculate the onset ratings deviation at the beginning of a period.
		This calulation updates the rd given the time passed, to add/remove
		uncertainty based on time passed.
		Params:
		-------
			- rd_old: rating deviation coming to match
		"""
		rd = np.sqrt(np.power(rd_old, 2) + (np.power(self.c, 2) * t))
		# logging.warning(f'Onset RD: {rd}')
		return rd


	def g(self, rd):
		"""
		Calculate 'g' for the opponent using onset rd
		Params:
		-------
			- rd: ratings deviation;
		"""
		return 1 / np.sqrt(1 + (3 \
				* np.power(self.q, 2) * np.power(rd, 2)) / np.power(np.pi, 2))


	def probability(self, g, ra, rb):
		"""
		Calculate expected result
		"""
		return 1 / (1 + np.power(10, -g * ((ra - rb) /self.width)))


	def dsqrd(self, g, E):
		"""
		Calculate d^2
		"""
		return 1 / (np.power(self.q, 2) * np.power(g, 2) * E * (1- E))


	def rd_new(self, g, E, rd):
		"""
		Calculate post-match RD (ratings deviation)
		"""
		if rd <= self.sigma_min:
			return rd

		d2 = self.dsqrd(g=g, E=E)
		# logging.warning(f'd2: {d2}')
		return 1 / np.sqrt(1 / np.power(rd, 2) + 1 / d2)


	def update_rating(self, r_pre, rd_pre, r_opp, outcome):
		"""
		Update a player's rating given their rd and the opponent rating, and
		the outcome.
		"""
		onsetrd = self.onset_rd(rd_pre, 1)
		g = self.g(onsetrd)
		# logging.warning(f'G: {g}')
		E = self.probability(g, r_pre, r_opp)
		rd_new = self.rd_new(g, E, onsetrd)

		r_post =r_pre + self.q*(np.power(rd_new, 2))*g*(outcome - E)

		return E, r_post, rd_new


	def rate(self, home_prev, vis_prev, outcome, rd_home, rd_vis):
		"""
		Calculate post-match rating for home and visitor teams
		Params:
		-------
			- outcome (1.0, 0.5, 0.0): always from home-team perspective
		"""
		## flipping the outcome's perspective
		v_outcome = abs(outcome - 1.)

		E_home, home_post, rd_home_new = \
					self.update_rating(home_prev, rd_home, vis_prev, outcome)
		E_vis, vis_post, rd_vis_new = \
					self.update_rating(vis_prev, rd_vis, home_prev, v_outcome)
		self.returns = [
			'phome','pvis',
			'hpost','vpost',
			'h_rd_new','v_rd_new'
		]
		return E_home, E_vis, home_post, vis_post, rd_home_new, rd_vis_new
