import numpy as np
import pandas as pd
'''
TODO:

	- Add weight decay to Elo ratings;
	- Add possibility of multiple seasons (not leagues.);
	- Maybe make sigma part of kwargs, and handle within Glicko model obj;

'''

class Rate(object):
	"""
		Class to apply ratings across a list of fixture dataset.
	"""
	def __init__(self, fixtures, model, start_rating=1500, sigma=350, **kwgs):
		self.fixtures = fixtures.sort_values('starting_datetime')\
								.reset_index(drop=True)
		self.sigma = sigma
		self.model = model(sigma=sigma, **kwgs)
		self.__init_ratings__(start_rating)


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


	def _update_team_rating(self, team_id, rating):
		self.team_ratings['rating'][team_id] = rating
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
		phome, pvis, hpost, vpost = self.model.rate(
			self.hpre,self.vpre,outcome=self.outcome, score_diff=self.score_diff
		)

		self._update_team_rating(row['localteam_id'], hpost)
		self._update_team_rating(row['visitorteam_id'], vpost)

		return {
			'hpre': self.hpre, 'vpre': self.vpre, 'phome': phome,
			'pvis': pvis, 'hpost':hpost, 'vpost':vpost
		}

	def rate_match(self, row, use_rd=False):
		"""
		Params:
			- row: matchup, or row of fixtures
			- use_rd (default=False): whether to use ratings deviation (std.)
		"""
		self.hpre = self.team_ratings['rating'][row['localteam_id']]
		self.vpre = self.team_ratings['rating'][row['visitorteam_id']]
		self.h_rd_pre = self.team_ratings['rd'][row['localteam_id']]
		self.v_rd_pre = self.team_ratings['rd'][row['visitorteam_id']]
		self.outcome = row['outcome']
		self.score_diff = row['score_diff']

		if use_rd:
			return self._rate_match_glicko(row)

		else:
			return self._rate_match_elo(row)


	def rate_fixtures(self, use_rd=False):
		"""
		Rate all fixtures. Perhaps it could be done with a rolling function.
		"""
		self._compute_outcomes()
		ratings = []
		ratings = self.fixtures.apply(self.rate_match, args=(use_rd,), axis=1)

		return self.fixtures.join(pd.DataFrame(ratings.values.tolist()))

#         for i, row in self.fixtures.iterrows():
#             ratings.append(self.rate_row(row, use_rd=use_rd))

#         return self.fixtures.join(pd.DataFrame(ratings))
