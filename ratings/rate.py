import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import metrics
'''
TODO:

	- Add weight decay to Elo ratings;
	- Regress to the mean at the end of a season
	- Maybe make sigma part of kwargs, and handle within Glicko model obj;

'''

class Rate(object):
	"""
		Class to apply ratings across a list of fixture dataset.

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

	def __init__(self,fixtures,model, start_rating=1500,sigma=350, season_regress=0.33, seed=1234,**kwgs):
		self.seed=seed
		self.season_regress=season_regress
		### dataset used for prediction
		self.prediction = fixtures[fixtures['has_finished']==False]\
				.sort_values('starting_datetime').reset_index(drop=True)

		### dataset used for historical assessment / regression
		self.fixtures = fixtures[fixtures['has_finished']==True]\
				.sort_values('starting_datetime').reset_index(drop=True)

		self.upcoming_fixtures = fixtures[fixtures['has_finished']==False]\
				.sort_values('starting_datetime').reset_index(drop=True)

		self.sigma = sigma
		self.model = model(sigma=sigma, **kwgs)
		self.__init_ratings__(start_rating)
		self._compute_outcomes()
		self.fixtures['is_new_season'] = (self.fixtures['season_name'] != self.fixtures['season_name'].shift(1)).astype(int)

		self.nba_gamma_function =  self._nba_spread_gamma(use_preset=True)


	def __init_ratings__(self, start_rating):
		"""
			Initialize all ratings to start_rating
		"""
		h = self.fixtures[['localteam_id']]\
				.rename(columns={'localteam_id':'team_id'})
		v = self.fixtures[['visitorteam_id']]\
				.rename(columns={'visitorteam_id':'team_id'})
		teams = pd.concat([h,v], axis=0, sort=False, ignore_index=True)\
					.drop_duplicates()
		teams['rating'] = start_rating
		teams['rd'] = self.sigma

		self.team_ratings = teams.set_index('team_id').to_dict()#['rating']

		self.historical_ratings = {
				team_id: [] for team_id in self.team_ratings['rating'].keys()
			}

		return True


	def _compute_outcomes(self):
		conditions = [
			self.fixtures['localteam_score']>self.fixtures['visitorteam_score'],
			self.fixtures['visitorteam_score']>self.fixtures['localteam_score'],
			self.fixtures['localteam_score']==self.fixtures['visitorteam_score']
		]
		choices = [ 1, 0, 0.5 ]

		self.fixtures['outcome'] = np.select(conditions, choices)

		### CALCULATING SCORE_DIFF AS A FUNCTION OF HOMETEAM
		self.fixtures['score_diff'] = self.fixtures['localteam_score'] - \
										self.fixtures['visitorteam_score']

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

		self._update_team_rd(row['localteam_id'], h_rd_post)
		self._update_team_rd(row['visitorteam_id'], v_rd_post)

		self._update_team_rating(row['localteam_id'], hpost)
		self._update_team_rating(row['visitorteam_id'], vpost)

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

		self._update_team_rating(row['localteam_id'], hpost)
		self._update_team_rating(row['visitorteam_id'], vpost)

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
		self.hpre = self.team_ratings['rating'][row['localteam_id']]
		self.vpre = self.team_ratings['rating'][row['visitorteam_id']]
		self.h_rd_pre = self.team_ratings['rd'][row['localteam_id']]
		self.v_rd_pre = self.team_ratings['rd'][row['visitorteam_id']]
		self.outcome = row['outcome']
		self.score_diff = row['score_diff']

		return getattr(self, f'_rate_match_{self.model.__modelname__}')(row)


	def rate_fixtures(self):#, use_rd=False):
		"""
		Rate all fixtures. Perhaps it could be done with a rolling function.
		"""
		# ratings = []
		ratings = self.fixtures.apply(self.rate_match, axis=1)
		self.fixtures = self.fixtures.join(pd.DataFrame(ratings.values.tolist()))

		### apply predicted spreads
		self.fixtures['hp_spread'] = self._nba_spread(self.fixtures['phome'].values)
		self.fixtures['vp_spread'] = self._nba_spread(self.fixtures['pvis'].values)

		return self.fixtures


	def _predict_prob_elo(self, row):
		team1 = row['localteam_id']
		team2 = row['visitorteam_id']

		home_prob, vis_prob = self.model.predict_prob(
			self.team_ratings['rating'][team1],
			self.team_ratings['rating'][team2],
		)
		return {'phome': home_prob, 'pvis': vis_prob}

	def _predict_prob_glicko(self,row):
		team1 = row['localteam_id']
		team2 = row['visitorteam_id']

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


	def predict_fixtures(self):
		"""
		Predict winning probability for future matches.
		"""
		ratings = self.upcoming_fixtures.apply(self.predict_prob, axis=1)
		self.upcoming_fixtures = self.upcoming_fixtures.join(
			pd.DataFrame(ratings.values.tolist())
		)

		### apply predicted spreads
		self.upcoming_fixtures['hp_spread'] = \
							self._nba_spread(self.upcoming_fixtures['phome'].values)
		self.upcoming_fixtures['vp_spread'] = \
							self._nba_spread(self.upcoming_fixtures['pvis'].values)
		return self.upcoming_fixtures


	def plot_team_ratings(self, team_id, **kwargs):
		_ratings = self.historical_ratings[team_id]
		return plt.plot(_ratings, **kwargs)

	def auc_score(self):
		y_pred_proba = self.fixtures['phome'].values
		y_test = (self.fixtures['score_diff'] > 0).values
		return metrics.roc_auc_score(y_test, y_pred_proba)

	def plot_roc_curve(self):
		y_pred_proba = self.fixtures['phome'].values
		y_test = (self.fixtures['score_diff'] > 0).values

		auc_score = self.auc_score()
		fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

		plt.plot(fpr,tpr,
				label=f'{self.model.__modelname__.capitalize()}, auc={auc_score:.3f}')
		plt.plot([0.0, 1.0],[0.0, 1.0],'--', c='black', ) #baseline (random)
		plt.legend(loc=4)
		return
