# -*- coding: utf-8 -*-
import numpy as np
import math

start_rating = 1500

class Elo(object):

	"""
	A python object to calculate Elo ratings
		:param start_rating: determines where the ratings start
	"""

	def __init__(self):
		self.start_rating = start_rating


	def probability(self, rating1, rating2, h1=0, h2=0):

		p1_w = 1.0 * 1.0 / (1 + 1.0 * math.pow(10, 1.0 * (rating2 - (rating1 + h1)) / 400))
		p2_w = 1 - p1_w
		#note that 400 is width, we can change this

		return p1_w, p2_w


	def elo_rating(self, Ra, Rb, K, d, start_rating=start_rating, score_diff=0, Sd_method='Logistic', h=0, m=1):

		if start_rating != self.start_rating:
			self.start_rating = start_rating

		# Calculate winning probabilities
		Pa, Pb = self.probability(Ra, Rb, h1=h)

		if d == 0.5:
			Sd = 1
		else:
			if Sd_method == 'Logistic':
				Sd = self._score_diff_logistic(score_diff, m=m)
			elif Sd_method == 'Power':
				Sd = self._score_diff_pwr(score_diff, m=m)
			elif Sd_method == 'Five_thirty_eight':
				if d == 1:
					ELOW = Ra
					ELOL = Rb
				else:
					ELOW = Rb
					ELOL = Ra
				Sd = self._score_diff_fte(score_diff, ELOW, ELOL)

		# Case -1 When Player A wins
		# Updating the Elo Ratings
		if (d == 1) :
			Ra_new = Ra + K * Sd * (1 - Pa)
			Rb_new = Rb + K * Sd * (0 - Pb)

		# Case -2 When Player B wins
		# Updating the Elo Ratings
		elif (d == 0.5):
			Ra_new = Ra + K * Sd * (0.5 - Pa)
			Rb_new = Rb + K * Sd * (0.5 - Pb)

		# Case -3 When it's a tie
		# Updating the Elo Ratings
		else:
			Ra_new = Ra + K * Sd * (0 - Pa)
			Rb_new = Rb + K * Sd * (1 - Pb)

		return Pa, Pb, Ra_new, Rb_new


	def _weighted_r_diff(self, r_chng, a):
		"""
		Determine time-weighted rating change
		"""
		length = len(r_chng)
		ordered_arr = np.arange(1, length +1)
		weights = np.power(((1 + ordered_arr - 1) / (1 + length - 1)), a)

		return np.sum(weights * r_chng)


	def _score_diff_logistic(self, Sd, m=1):
		"""
		logistic function that intersects 1 at 1
		"""
		Sdm = 2/(1+np.exp(-m * (Sd - 1)))

		return Sdm


	def _score_diff_pwr(self, Sd, m=1):
		"""
		absolute score difference that is scaled by a power
		"""
		Sdm = np.power((1 + Sd), m)

		return Sdm


	def _score_diff_fte(self, Sd, ELOW, ELOL):
		"""
		margin of victory multiplier borrowed from FiveThirtyEight
		"""
		Sdm = np.log(np.abs(Sd)+1) * (2.2/((ELOW-ELOL)*.001+2.2))

		return Sdm
