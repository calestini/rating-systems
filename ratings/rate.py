import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import scipy.stats as stats

from .helpers import get_names

'''
TODO:

	- Add weight decay to Elo ratings;
	- Regress to the mean at the end of a season
	- Maybe make sigma part of kwargs, and handle within Glicko model obj;

'''

class Rate(object):
	"""
		Class to apply ratings across a list of fixture dataset.

	Needed Fields:
	--------------
		- has_finished: (bool) whether the game finished
		- start_datetime: (datetime, numeric) order of games
		- season_name: (str) season name
		- localteam_id / visitorteam_id: (str, int) team ids
		- localteam_score / visitorteam_score: (str, float) team scores


	Example
	-------
	>>> elo = Rate(fixtures, model=Elo, season_regress=0.33)
	>>> elo.rate_fixtures()
	>>> glicko = Rate(fixtures, model=Glicko)
	>>> glicko.rate_fixtures();

	>>> plt.figure(figsize=(8,8))
	>>> elo.plot_roc_curve()
	>>> glicko.plot_roc_curve()
	>>> plt.show()
	"""

	def __init__(self, model, start_rating=1500, sigma=350,
				season_regress=0.33, seed=1234, **kwgs):

		self.seed=seed
		self.season_regress=season_regress
		self.start_rating = start_rating
		### dataset used for prediction
		# self.prediction = fixtures[fixtures['has_finished']==False]\
		# 		.sort_values('start_datetime').reset_index(drop=True)

		### dataset used for historical assessment / regression
		# self.ht, self.vt, self.hs, self.vs, self.ss, self.sd = get_names(train)
		# self.fixtures = train.sort_values(self.sd).reset_index(drop=True)
		#
		# self.upcoming_fixtures = fixtures[fixtures['has_finished']==False]\
		# 		.sort_values('start_datetime').reset_index(drop=True)

		self.sigma = sigma
		self.model = model(sigma=sigma, **kwgs)
		self.nba_gamma_function =  self._nba_spread_gamma(use_preset=True)

		# self._init_ratings(start_rating)



	def _init_ratings(self, start_rating):
		"""
			Initialize all ratings to start_rating
		"""
		h = self.fixtures[[self.ht]].rename(columns={self.ht:'team_id'})
		v = self.fixtures[[self.vt]].rename(columns={self.vt:'team_id'})
		teams = pd.concat([h,v], axis=0, sort=False, ignore_index=True)\
					.drop_duplicates()
		teams['rating'] = start_rating
		teams['rd'] = self.sigma

		self.team_ratings = teams.set_index('team_id').to_dict()#['rating']
		self.historical_ratings = {
				team_id: [] for team_id in self.team_ratings['rating'].keys()
			}
		self._compute_outcomes()
		self.fixtures['is_new_season'] = (self.fixtures[self.ss] != \
									self.fixtures[self.ss].shift(1)).astype(int)

		return True


	def _compute_outcomes(self):
		conditions = [
			self.fixtures[self.hs]>self.fixtures[self.vs],
			self.fixtures[self.vs]>self.fixtures[self.hs],
			self.fixtures[self.hs]==self.fixtures[self.vs]
		]
		choices = [ 1, 0, 0.5 ]

		self.fixtures['outcome'] = np.select(conditions, choices)

		### DROP NA FOR SCORE FIELDS. IF NO SCORE, NOT USED IN TRAINING
		self.fixtures.dropna(subset=[self.hs,self.vs], inplace=True)

		### CALCULATING SCORE_DIFF AS A FUNCTION OF HOMETEAM
		self.fixtures['score_diff'] = self.fixtures[self.hs].astype(float) - \
											self.fixtures[self.vs].astype(float)

		return True


	def _nba_spread_gamma(self, use_preset=True):
		"""
		Function to create spread distribution.
		"""
		np.random.seed(self.seed)

		if use_preset:
			### pre-set values (based on nba data from 2012-2019)
			alpha = 1.8415843797113318
			loc = -0.006494101048288639
			beta = 6.086268833321849

			return {
				'alpha': 1.8415843797113318,
				'loc': -0.006494101048288639,
				'beta': 6.086268833321849}

		### we can also fit our score_diff data to a gamma distribution.
		alpha, loc, beta=stats.gamma.fit(
				np.abs(self.fixtures['score_diff'].values)
			)

		return {'alpha': alpha, 'loc': loc, 'beta': beta}


	def _nba_spread(self, prob, digits=1):
		"""
		Function to calculate the spread based on the probability.
		"""
		prob = np.select([prob<0.95, prob>=0.95], [prob, 0.95])
		pct = np.abs(prob-0.50)*2
		# spread = np.round(np.percentile(self.nba_spread_dist, pct), digits)
		spread = np.round(
			stats.gamma.ppf(pct,
				self.nba_gamma_function['alpha'],
				loc=self.nba_gamma_function['loc'],
				scale=self.nba_gamma_function['beta']
			), digits)

		spread = np.where(prob>.50,-spread,spread)
		return spread


	def _regress_to_mean(self, rate=None):
		"""
		Regress the ratings to the mean, using a rate (0<rate<1) of decay.
		Parameters:
		----------
			- rate: (float): rate of decay for all ratings.
		"""
		rate = rate or self.season_regress
		mean = np.array([
			self.team_ratings['rating'][k] for k in self.team_ratings['rating']
		]).mean()

		for team in self.team_ratings['rating'].keys():
			self.team_ratings['rating'][team] += \
								(mean - self.team_ratings['rating'][team])*rate
		return True


	def _update_team_rating(self, team_id, rating):
		self.team_ratings['rating'][team_id] = rating
		self.historical_ratings[team_id].append(rating)
		return True


	def _update_team_rd(self, team_id, rd):
		self.team_ratings['rd'][team_id] = rd
		return True


	def _rate_match_glicko(self, row):
		phome, pvis, hpost, vpost, h_rd_post, v_rd_post = self.model.rate(
			self.hpre, self.vpre, self.outcome, self.h_rd_pre, self.v_rd_pre
		)

		self._update_team_rd(row[self.ht], h_rd_post)
		self._update_team_rd(row[self.vt], v_rd_post)

		self._update_team_rating(row[self.ht], hpost)
		self._update_team_rating(row[self.vt], vpost)

		return {
			'hpre':self.hpre, 'vpre': self.vpre,
			'h_rd_pre': self.h_rd_pre, 'v_rd_pre': self.v_rd_pre,
			'phome': phome, 'pvis': pvis,
			'hpost': hpost, 'vpost': vpost,
			'h_rd_post': h_rd_post, 'v_rd_post': v_rd_post
		}


	def _rate_match_elo(self, row):

		if row['is_new_season'] == 1:
			## only applicable to ELO
			self._regress_to_mean()

		phome, pvis, hpost, vpost = self.model.rate(
			self.hpre,self.vpre,outcome=self.outcome, score_diff=self.score_diff
		)

		self._update_team_rating(row[self.ht], hpost)
		self._update_team_rating(row[self.vt], vpost)

		return {
			'hpre': self.hpre, 'vpre': self.vpre, 'phome': phome,
			'pvis': pvis, 'hpost':hpost, 'vpost':vpost,
		}


	def rate_match(self, row):
		"""
		Parameters
		----------
			- row: matchup, or row of fixtures
			- use_rd (default=False): whether to use ratings deviation (std.)
		"""
		self.hpre = self.team_ratings['rating'][row[self.ht]]
		self.vpre = self.team_ratings['rating'][row[self.vt]]
		self.h_rd_pre = self.team_ratings['rd'][row[self.ht]]
		self.v_rd_pre = self.team_ratings['rd'][row[self.vt]]
		self.outcome = row['outcome']
		self.score_diff = row['score_diff']

		return getattr(self, f'_rate_match_{self.model.__modelname__}')(row)


	def rate_fixtures(self, train, start_rating=1500):#, use_rd=False):
		"""
		Rate all fixtures. Perhaps it could be done with a rolling function.
		"""
		self.ht, self.vt, self.hs, self.vs, self.ss, self.sd = get_names(train)
		self.fixtures = train.sort_values(self.sd).reset_index(drop=True)
		self._init_ratings(start_rating=start_rating)

		# ratings = []
		ratings = self.fixtures.apply(self.rate_match, axis=1)
		self.fixtures = self.fixtures.join(pd.DataFrame(ratings.values.tolist()))

		### apply predicted spreads
		self.fixtures['hp_spread'] = self._nba_spread(self.fixtures['phome'].values)
		self.fixtures['vp_spread'] = self._nba_spread(self.fixtures['pvis'].values)

		return self.fixtures


	def _predict_prob_elo(self, row):
		team1 = row[self.ht]
		team2 = row[self.vt]

		home_prob, vis_prob = self.model.predict_prob(
			self.team_ratings['rating'][team1],
			self.team_ratings['rating'][team2],
		)
		return {'phome': home_prob, 'pvis': vis_prob}

	def _predict_prob_glicko(self,row):
		team1 = row[self.ht]
		team2 = row[self.vt]

		home_prob, vis_prob = self.model.predict_prob(
			self.team_ratings['rating'][team1],
			self.team_ratings['rating'][team2],
			self.team_ratings['rd'][team1],
			self.team_ratings['rd'][team2]
		)

		return {'phome': home_prob, 'pvis': vis_prob}


	def predict_prob(self, row):
		"""
		Return the predicted probability of a match-up.
		"""
		return getattr(self, f'_predict_prob_{self.model.__modelname__}')(row)


	def predict_fixtures(self, fixtures):
		"""
		Predict winning probability for future matches.
		"""
		fixtures = fixtures.reset_index(drop=True) ## needed
		self.ht, self.vt = get_names(fixtures, simplified=True)
		# fixtures = fixtures or self.upcoming_fixtures
		ratings = fixtures.apply(self.predict_prob, axis=1)

		fixtures = fixtures.join(
			pd.DataFrame(ratings.values.tolist())
		)

		### apply predicted spreads
		fixtures['hp_spread'] = \
							self._nba_spread(fixtures['phome'].values)
		fixtures['vp_spread'] = \
							self._nba_spread(fixtures['pvis'].values)
		return fixtures


	def plot_team_ratings(self, team_id, **kwargs):
		## local import
		import matplotlib.pyplot as plt

		_ratings = self.historical_ratings[team_id]
		return plt.plot(_ratings, **kwargs)

	def auc_score(self):
		## local import
		from sklearn import metrics

		y_pred_proba = self.fixtures['phome'].values
		y_test = (self.fixtures['score_diff'] > 0).values
		return metrics.roc_auc_score(y_test, y_pred_proba)

	def plot_roc_curve(self):
		## local import
		import matplotlib.pyplot as plt
		from sklearn import metrics

		y_pred_proba = self.fixtures['phome'].values
		y_test = (self.fixtures['score_diff'] > 0).values

		auc_score = self.auc_score()
		fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

		plt.plot(fpr,tpr,
				label=f'{self.model.__modelname__.capitalize()}, auc={auc_score:.3f}')
		plt.plot([0.0, 1.0],[0.0, 1.0],'--', c='black', ) #baseline (random)
		plt.legend(loc=4)
		return
