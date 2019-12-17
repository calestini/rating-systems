import pandas as pd
import numpy as np
from numpy import vectorize
from time import time

from .elo import Elo
from .glicko import Glicko

### SILENCE WARNING IN PRODUCTION
pd.options.mode.chained_assignment = None

class Rate(object):

	def __init__(self,
			localteam_id = 'localteam_id',
			visitorteam_id = 'visitorteam_id',
			localteam_name = 'localteam_name',
			visitorteam_name = 'visitorteam_name',
			localteam_score = 'localteam_score',
			visitorteam_score = 'visitorteam_score',
			order = 'starting_datetime', league = 'league_name',
			season = 'season_name',	start_rating = 1500, sigma=350,
			start_rating=1500, c=20):

		self.ratings_fixtures = pd.DataFrame()
		self.ratings_teams = pd.DataFrame()
		self.ratings_teams_fixtures = pd.DataFrame()
		self.ratings_teams_seasons = pd.DataFrame()
		self.localteam_id = localteam_id
		self.visitorteam_id = visitorteam_id
		self.localteam_name = localteam_name
		self.visitorteam_name = visitorteam_name
		self.localteam_score = localteam_score
		self.visitorteam_score = visitorteam_score
		self.order = order
		self.league = league
		self.season = season
		self.start_rating = start_rating
		self.sigma = sigma
		self.c = c

		self.progress = 0
		self.ties_exist = True

		### PREDICTIONS IN PROBABILITIES
		self.true = np.empty([1, 1])
		self.pred = np.empty([1, 1])

		### PREDICTION IN CLASSES (i.e. 1, 0)
		self.true_classes = np.empty([1,1])
		self.pred_classes = np.empty([1,1])

		self.cm = np.empty([1,1])
		self.c_report = ''
		self.roc_auc = dict()
		self.log_loss = 0


	def _set_teams(self, data, rating_method):
		"""
		Turn list of fixtures into teams, and set initial score
		"""
		teams_local = data[[self.localteam_id, self.localteam_name]] \
					.drop_duplicates().rename(columns={
						self.localteam_id: 'team_id',
						self.localteam_name: 'team_name'
					})

		teams_visitor = data[[self.visitorteam_id, self.visitorteam_name]] \
						.drop_duplicates().rename(columns={
							self.visitorteam_id: 'team_id',
							self.visitorteam_name: 'team_name'
						})

		teams = teams_local.append(teams_visitor, ignore_index = True)\
														.drop_duplicates()
		teams['rating'] = self.start_rating

		if rating_method == 'Glicko':
			teams['RD'] = self.Sigma

		teams.set_index('team_id', inplace = True)

		return teams


	def _set_team_id(self, data):
		"""
		Set the team id if team id does not exist.
		NOTE:
			- In the ideal world this should not be needed.
		"""
		teams_local = data[[self.localteam_name]]\
					.drop_duplicates().rename(columns={
						self.localteam_name: 'team_name'
					})
		teams_visitor = data[[self.visitorteam_name]]\
					.drop_duplicates().rename(columns={
						self.visitorteam_name: 'team_name'
					})
		teams = teams_local.append(teams_visitor, ignore_index = True)\
														.drop_duplicates()
		teams['team_id'] = teams.index.values

		data = data.merge(teams,
					left_on = self.localteam_name,  right_on = 'team_name')\
					.rename(columns={'team_id': 'localteam_id'}
				)
		data = data.merge(teams,
					left_on = self.visitorteam_name, right_on = 'team_name')\
					.rename(columns={'team_id': 'visitorteam_id'}
				)

		self.localteam_id = 'localteam_id'
		self.visitorteam_id = 'visitorteam_id'

		return data


	def _process_data(self, data):
		"""
		Process fixture data - order fixtures and assign outcomes
		"""
		### DETERMINE OUTCOME FOR EACH FIXTURE / MATCHUP
		conditions = [
			data[self.localteam_score] > data[self.visitorteam_score],
			data[self.visitorteam_score] > data[self.localteam_score],
			data[self.localteam_score] == data[self.visitorteam_score]
		]
		choices = [ 1, 0, 0.5 ]
		data['outcome'] = np.select(conditions, choices)

		conditions = [
			data[self.localteam_score] > data[self.visitorteam_score],
			data[self.visitorteam_score] > data[self.localteam_score],
			data[self.localteam_score] == data[self.visitorteam_score]
		]
		choices = [ 0, 1, 0.5 ]
		data['visitorteam_outcome'] = np.select(conditions, choices)

		### SORT FIXTURES / MATCHUPS
		data.sort_values(self.order, inplace = True)

		if (self.localteam_id == '') | (self.visitorteam_id == '') :
			data = self._set_team_id(data)

		return data


	def calculate_tie_prob(self, s, p):
		"""
		Calculate the win, loss, tie probabilities using actual outcomes and
		predicted home team probabilities
		"""
		#turning home, tie, away into three different columns
		true_h = np.where(s==1, 1, 0)
		true_a = np.where(s==0, 1, 0)
		true_t = np.where(s==0.5, 1, 0)

		nbins = 50

		y_binned_h, x_bins_h = np.histogram(p, bins=nbins, weights=true_h)
		y_total_h, x_bins_h = np.histogram(p, bins=nbins)

		y_binned_a, x_bins_a = np.histogram(p, bins=nbins, weights=true_a)
		y_total_a, x_bins_a = np.histogram(p, bins=nbins)

		y_binned_t, x_bins_t = np.histogram(p, bins=nbins, weights=true_t)
		y_total_t, x_bins_t = np.histogram(p, bins=nbins)

		### Excluding instances where the number of samples in the bin is < 15
		mask_h = y_total_h > 20
		mask_a = y_total_a > 20
		mask_t = y_total_t > 20

		y_binned_h = y_binned_h[mask_h]
		y_binned_a = y_binned_a[mask_a]
		y_binned_t = y_binned_t[mask_t]

		y_total_h = y_total_h[mask_h]
		y_total_a = y_total_a[mask_a]
		y_total_t = y_total_t[mask_t]

		x_bins_h = x_bins_h[1:][mask_h]
		x_bins_a = x_bins_a[1:][mask_a]
		x_bins_t = x_bins_t[1:][mask_t]

		mean_h = np.divide(y_binned_h, y_total_h)
		mean_h[np.isnan(mean_h)]=0

		mean_a = np.divide(y_binned_a, y_total_a)
		mean_a[np.isnan(mean_a)]=0

		mean_t = np.divide(y_binned_t, y_total_t)
		mean_t[np.isnan(mean_t)]=0

		#Calculate lines of best fit
		z_h = np.polyfit(x_bins_h, mean_h, 2)
		p_h = np.poly1d(z_h)

		z_a = np.polyfit(x_bins_a, mean_a, 2)
		p_a = np.poly1d(z_a)

		z_t = np.polyfit(x_bins_t, mean_t, 2)
		p_t = np.poly1d(z_t)

		def three_way_probability(p):
			p_tie =  p_t(p)
			p_away =  p_a(p)
			p_home = p_h(p)

			return p_home, p_away, p_tie

		p_home, p_away, p_tie = three_way_probability(p)

		true = np.column_stack((true_h, true_a, true_t))
		pred = np.column_stack((p_home, p_away, p_tie))

		true_outcome = np.where(s == 0.5, 2, s)
		vpredict_outcome = vectorize(self.predict_outcome)
		pred_outcome = vpredict_outcome(p_home, p_away, p_tie)

		self.true = true
		self.pred = pred
		self.true_classes = true_outcome
		self.pred_classes = pred_outcome


	def predict_outcome(self, p_home, p_away, p_tie, margin=0.1):
		"""
		Predict outcomes in order to generate confusion matrix and precision, recall, and f-1 score
		"""
		#Establish a margin where if probability of home and away winning are similar, then it's a tie
		margin = margin
		h_a_diff = np.abs(p_away - p_home)

		#Using an integer to represent ties because evaluation functions require this
		if (h_a_diff < margin) or (p_tie > p_home and p_tie > p_away):
			return 2
		elif p_home > p_away and p_home > p_tie:
			return 1
		else:
			return 0


	def calculate_elo(self, data, K, h=0, Sd=False, Sd_method='Logistic', m=1,
						results_by_league=False, results_by_season=False,
						rt_mean=False, rt_mean_amnt=1/3, time_scale=False,
						time_scale_factor=2, start_rating=None,
						tie_probability=True):
		"""
		Calculate elo rating for fixtures, accepts a dataframe

		:param K: constant used to gauge magnitude of effect for each match outcome
		:param h: home-team advantage, defaults to 0
		:param Sd: either True or False - takes into account score difference when set to True
		:param Sd_method: method used to account for score difference, options are 'Logistic', 'Constant', or 'Five_thirty_eight', defaults to 'Logistic'
		:param m: multiplier for score difference - defaults to 0, is not used with 'Five_thirty_eight' method
		:param results_by_league: calculate ratings for each league individually, defaults to False
		:param results_per_season: calculate ratings for each season individually, defaults to False
		:param rt_mean: regress ratings to league mean after each season, defaults to False
		:param rt_mean_degree: amount to regress to mean, defaults to 1/3
		:param tie_probability: recommended to be left to 'True' - calculates tie probability when there are tie outcomes

		Note that if results_by_league is False and results_by_season is True, only one league should be passed as data
		"""

		self.ratings_fixtures = pd.DataFrame()
		self.ratings_teams = pd.DataFrame()
		self.ratings_teams_fixtures = pd.DataFrame()
		self.ratings_teams_seasons = pd.DataFrame()
		self.progress = 0

		start = time()

		self.start_rating = start_rating or self.start_rating

		print ('Prepping data...')
		data = self._process_data(data)

		print ('Starting calculations...')
		self.total = len(data.index)

		if results_by_league == False and results_by_season == False:

			ratings_teams = self._set_ratings_elo(data=data, K=K, h=h, Sd=Sd, Sd_method=Sd_method, m=m, time_scale=time_scale, time_scale_factor=time_scale_factor)
			ratings_teams.sort_values('rating', ascending = False, inplace = True)
			self.ratings_teams = ratings_teams

		elif results_by_league == True and results_by_season == False:

			leagues = data[self.league].unique()
			league_team_r = pd.DataFrame()

			for league in leagues:
				league_subset = data[data[self.league] == league]
				league_subset_r = self._set_ratings_elo(league_subset, K, h=h, Sd=Sd, Sd_method=Sd_method, m=m, time_scale=time_scale, time_scale_factor=time_scale_factor)
				league_subset_r[self.league] = league
				league_team_r = league_team_r.append(league_subset_r)

			league_team_r.sort_values([self.league, 'rating'], ascending = False, inplace = True)
			self.ratings_teams = league_team_r

		elif results_by_league == False and results_by_season == True:
			print('Results being calculated by season, please make sure that only one league is passed in the data. \nIf you would like to pass multiple leagues, please set "results_by_league to True')

			seasons = data[self.season].unique()
			season_team_r = pd.DataFrame()
			team_r = self._set_teams(data, rating_method='Elo')

			for season in seasons:

				#turn original team ratings into preseason ratings
				team_r_pre = team_r.rename(columns={'rating': 'rating_preseason'}).drop('team_name', axis= 1)

				#get team ratings post season
				season_subset_r = self._set_ratings_elo(data[data[self.season] == season], K, h=h, Sd=Sd, Sd_method=Sd_method,
													m=m, prev_ratings=team_r, time_scale=time_scale, time_scale_factor=time_scale_factor)

				#update team ratings
				if rt_mean == True:
					ratings = season_subset_r['rating'].values.copy()
					season_avg = np.mean(ratings)
					season_subset_r['rating'] = ratings + (season_avg - ratings) * rt_mean_amnt
					team_r = team_r.combine_first(season_subset_r)
					season_subset_r['o_rating_postseason'] = ratings
				else:
					team_r = team_r.combine_first(season_subset_r)

				#join preseason to post season ratings
				season_subset_r = season_subset_r.join(team_r_pre).rename(columns={'rating': 'rating_postseason'})

				#add season to pre and post season ratings
				season_subset_r[self.season] = season
				season_team_r = season_team_r.append(season_subset_r)

			season_team_r.sort_values([self.season, 'rating_postseason'], ascending = False, inplace = True)
			self.ratings_teams = team_r.sort_values('rating', ascending = False)
			self.ratings_teams_seasons = season_team_r

		elif results_by_league == True and results_by_season == True:

			leagues = data[self.league].unique()
			league_team_season_r = pd.DataFrame()

			for league in leagues:
				league_subset = data[data[self.league] == league]
				seasons = league_subset[self.season].unique()

				season_team_r = pd.DataFrame()
				team_r = self._set_teams(league_subset, rating_method='Elo')

				for season in seasons:

					#turn original team ratings into preseason ratings
					team_r_pre = team_r.rename(columns={'rating': 'rating_preseason'}).drop('team_name', axis= 1)

					#get team ratings post season
					season_subset_r = self._set_ratings_elo(data[data[self.season] == season], K, h=h, Sd=Sd,
														Sd_method=Sd_method, m=m, prev_ratings=team_r, time_scale=time_scale, time_scale_factor=time_scale_factor)

					#update team ratings
					team_r = team_r.update(season_subset_r)

					#join preseason to post season ratings
					season_subset_r = season_subset_r.join(team_r_pre).rename(columns={'rating': 'rating_postseason'})

					#add season to pre and post season ratings
					season_subset_r[self.season] = season
					season_team_r = season_team_r.append(season_subset_r)

				season_team_r[self.league] = league
				league_team_season_r = league_team_season_r.append(season_team_r)

			self.ratings_teams = league_team_season_r.sort_values([self.league, 'rating'], ascending = False)

		"""
		#calculate log loss of the predictions
		s = data['outcome'].values
		p = self.ratings_fixtures['localteam_p'].values

		l_loss = log_loss(s, p)
		sq_err = squared_error(s, p)

		self.log_loss = l_loss
		self.squared_error = sq_err

		p_tie = np.full((self.total, ), 0.5, dtype=float)
		l_loss_tie = log_loss(s, p_tie)
		sq_err_tie = squared_error(s, p_tie)
		self.log_loss_tie = l_loss_tie
		self.squared_error_tie = sq_err_tie

		p_home = np.full((self.total, ), 0.999, dtype=float)
		l_loss_home = log_loss(s, p_home)
		sq_err_home = squared_error(s, p_home)
		self.log_loss_home = l_loss_home
		self.squared_error_tie = sq_err_home
		"""
		s = data['outcome'].values
		p = self.ratings_fixtures['localteam_p'].values
		ties_exist = ((s == 0.5).sum) != 0

		if ties_exist:
			self.ties_exist = True
		else:
			self.ties_exist = False
			self.true = s
			self.pred = p
			self.true_classes = s
			self.pred_classes = np.where(p > 0.5, 1, 0)

		if tie_probability == True and ties_exist:
			self.calculate_tie_prob(s, p)

		# self.evaluate_predictions()

		end = time()

		progress(100, 100, status="Hooray!")
		print ('Calculations completed in %f seconds' % (end-start))


	def _set_ratings_elo(self, data, K, h=0, Sd=False, Sd_method='Logistic', m=1, prev_ratings=None, time_scale=False, time_scale_factor=2):
		"""
		Calculates elo rating for every row of fixture data passed
		"""

		if prev_ratings is None:
			teams = self._set_teams(data, rating_method='Elo')
		else:
			teams = prev_ratings

		if Sd == False:
			ratings_function = 'Elo.elo_rating(localteam_r, visitorteam_r, K, outcome, start_rating=self.start_rating, score_diff=0, h=h)'
		else:
			ratings_function = 'Elo.elo_rating(localteam_r, visitorteam_r, K, outcome, score_diff=score_diff, start_rating=self.start_rating, Sd_method=Sd_method, m=m, h=h)'

		"""
		#get location of columns
		index_loc = 0
		order_loc = data.columns.get_loc(self.order)
		localteam_id_loc = data.columns.get_loc(self.localteam_id)
		visitorteam_id_loc = data.columns.get_loc(self.visitorteam_id)
		localteam_name_loc = data.columns.get_loc(self.localteam_name)
		visitorteam_name_loc = data.columns.get_loc(self.visitorteam_name)
		outcome_loc = data.columns.get_loc("outcome")
		visitorteam_outcome_loc = data.columns.get_loc("visitorteam_outcome")
		localteam_r_loc = data.columns.get_loc(self.localteam_r)
		visitorteam_r_loc = data.columns.get_loc(self.visitorteam_r)
		"""
		recarray = data.to_records(index=True)
		fixture_ratings = dict()
		new_local_ratings = dict()
		new_visitor_ratings = dict()

		## loop through every row of the dataset
		for row in recarray:

			'''
			####################################################################
			### EXPLANATION TO THE LOOP ACTIONS
			####################################################################
			1) GET THE INFO FROM THE ROW FOR BOTH TEAMS;
			2) GET SCORE DIFFERENCE
			3) CALCULATE RATING
			4) DECAY BASED ON TIME FACTOR IS APPLICABLE
			####################################################################
			####################################################################
			'''

			self.progress += 1
			progress(self.progress, self.total, status=row[0])

			index = row[0] #get row index
			order = getattr(row, self.order)
			localteam_id = getattr(row, self.localteam_id)
			visitorteam_id = getattr(row, self.visitorteam_id)
			localteam_name = getattr(row, self.localteam_name)
			visitorteam_name = getattr(row, self.visitorteam_name)
			outcome = row.outcome
			visitorteam_outcome = row.visitorteam_outcome
			localteam_r = teams.loc[localteam_id]['rating']
			visitorteam_r = teams.loc[visitorteam_id]['rating']
			score_diff = np.abs(getattr(row, self.localteam_score) \
										- getattr(row, self.visitorteam_score))

			### CALCULATE RATINGS
			localteam_p, visitorteam_p, localteam_post_r, visitorteam_post_r = eval(ratings_function)

			### ASSIGN PROBABILITIES FOR CURRENT GAME AND POST RATINGS
			fixture_ratings[index] = { 'result_order': order,
									   'localteam_r': localteam_r,
									  'visitorteam_r': visitorteam_r,
									  'localteam_p': localteam_p,
									  'visitorteam_p': visitorteam_p,
									  'localteam_post_r': localteam_post_r,
									  'visitorteam_post_r': visitorteam_post_r}

			### UPDATE TEAM RATING WITHOUT TIME SCALING
			if time_scale == False:
				teams.loc[localteam_id, 'rating'] = localteam_post_r
				teams.loc[visitorteam_id, 'rating'] = visitorteam_post_r

				### ASSIGN PROBABILITIES FOR EACH TEAM OVER TIME
				new_local_ratings[index] = {'id': index,
									'order': order,
									'position': 'local',
									'team_id': localteam_id,
									'team_name': localteam_name,
									'team_r': localteam_r,
									'team_p': localteam_p,
									'outcome': outcome,
									'team_post_r': localteam_post_r}
				new_visitor_ratings[index] = {'id': index,
										'order': order,
										'position': 'visitor',
										'team_id': visitorteam_id,
										'team_name': visitorteam_name,
										'team_r': visitorteam_r,
										'team_p': visitorteam_p,
										'outcome': visitorteam_outcome,
										'team_post_r': visitorteam_post_r}

			### UPDATE TEAM RATING WITH TIME SCALING
			elif time_scale == True:
				'''
				################################################################
				### EXPLANATION (LC)
				################################################################
				Then time_scale is used, the function _weighted_r_diff is
				applied to the delta rating history of a team, together with the
				delta of the current rating (post-pre)
				################################################################
				################################################################
				'''
				if (len(self.ratings_teams_fixtures.index) > 0):

					lteam_fixtures = self.ratings_teams_fixtures[self.ratings_teams_fixtures['team_id'] == localteam_id]
					vteam_fixtures = self.ratings_teams_fixtures[self.ratings_teams_fixtures['team_id'] == visitorteam_id]

					if (len(lteam_fixtures.index) > 0):
						lteam_r_chng = lteam_fixtures['team_post_r'].values - lteam_fixtures['team_r'].values
						lteam_r_chng_append = localteam_post_r - localteam_r
						lteam_r_chng = np.append(lteam_r_chng, lteam_r_chng_append)

						lteam_r_chng_current = Elo._weighted_r_diff(lteam_r_chng, time_scale_factor)
						teams.loc[localteam_id, 'rating'] = self.start_rating + lteam_r_chng_current
						lteam_post_r_ts = self.start_rating + lteam_r_chng_current

					if (len(vteam_fixtures.index) > 0):
						vteam_r_chng = vteam_fixtures['team_post_r'].values - vteam_fixtures['team_r'].values
						vteam_r_chng_append = visitorteam_post_r - visitorteam_r
						vteam_r_chng = np.append(vteam_r_chng, vteam_r_chng_append)

						vteam_r_chng_current = Elo._weighted_r_diff(vteam_r_chng, time_scale_factor)
						teams.loc[visitorteam_id, 'rating'] = self.start_rating + vteam_r_chng_current
						vteam_post_r_ts = self.start_rating + vteam_r_chng_current

					if (len(lteam_fixtures.index) == 0):
						teams.loc[localteam_id, 'rating'] = localteam_post_r
						lteam_post_r_ts = localteam_post_r

					if (len(vteam_fixtures.index) == 0):
						teams.loc[visitorteam_id, 'rating'] = visitorteam_post_r
						vteam_post_r_ts = visitorteam_post_r

				else:
					teams.loc[localteam_id, 'rating'] = localteam_post_r
					lteam_post_r_ts = localteam_post_r
					teams.loc[visitorteam_id, 'rating'] = visitorteam_post_r
					vteam_post_r_ts = visitorteam_post_r

				#assign probabilities for each team over time
				new_local_ratings[index] = {'id': index,
									'order': order,
									'position': 'local',
									'team_id': localteam_id,
									'team_name': localteam_name,
									'team_r': localteam_r,
									'team_p': localteam_p,
									'outcome': outcome,
									'team_post_r': localteam_post_r,
									'team_post_r_ts': lteam_post_r_ts}
				new_visitor_ratings[index] = {'id': index,
										'order': order,
										'position': 'visitor',
										'team_id': visitorteam_id,
										'team_name': visitorteam_name,
										'team_r': visitorteam_r,
										'team_p': visitorteam_p,
										'outcome': visitorteam_outcome,
										'team_post_r': visitorteam_post_r,
										'team_post_r_ts': vteam_post_r_ts}

		self.ratings_fixtures = self.ratings_fixtures.append(pd.DataFrame.from_dict(fixture_ratings, orient='index'), ignore_index = False).sort_values('result_order')
		self.ratings_teams_fixtures = self.ratings_teams_fixtures.append(pd.DataFrame.from_dict(new_local_ratings, orient='index'), ignore_index = True)
		self.ratings_teams_fixtures = self.ratings_teams_fixtures.append(pd.DataFrame.from_dict(new_visitor_ratings, orient='index'), ignore_index = True)

		return teams


	def calculate_glicko(self, data, c=c, start_rating=None, Sigma=Sigma, h=0,
				results_by_league=False, results_by_season=False, tie_probability=True):
		"""
		Calculate Glicko rating for fixtures, accepts a dataframe

		:param c: determines how much RD goes back up between periods of assessment, defaults to 20
		:param start_rating: determines where the ratings start, defaults to 1500
		:param Sigma: the start RD for every team, defaults to 350
		:param results_by_league: calculate ratings for each league individually, defaults to False
		:param results_per_season: calculate ratings for each season individually, defaults to False
		:param tie_probability: recommended to be left to 'True' - calculates tie probability when there are tie outcomes

		Note that if results_by_league is False and results_by_season is True, only one league should be passed as data
		"""
		self.start_rating = start_rating or self.start_rating

		self.ratings_fixtures = pd.DataFrame()
		self.ratings_teams_fixtures = pd.DataFrame()
		self.ratings_teams_seasons = pd.DataFrame()
		self.progress = 0

		if Sigma != self.Sigma:
			self.Sigma = Sigma

		if c != self.c:
			self.c = c

		start = time()

		print ('Prepping data...')
		data = self._process_data(data)

		print ('Starting calculations...')
		self.total = len(data.index)

		if results_by_league == False and results_by_season == False:

			ratings_teams = self._set_ratings_glicko(data, h=h)
			ratings_teams.sort_values('rating', ascending = False, inplace = True)
			self.ratings_teams = ratings_teams

		elif results_by_league == True and results_by_season == False:

			leagues = data[self.league].unique()
			league_team_r = pd.DataFrame()

			for league in leagues:
				league_subset = data[data[self.league] == league]
				league_subset_r = self._set_ratings_glicko(data, h=h)
				league_subset_r[self.league] = league
				league_team_r = league_team_r.append(league_subset_r)

			league_team_r.sort_values([self.league, 'rating'], ascending = False, inplace = True)
			self.ratings_teams = league_team_r

		elif results_by_league == False and results_by_season == True:
			print('Results being calculated by season, please make sure that only one league is passed in the data. If you would like to pass multiple leagues, please set "results_by_league to True')

			seasons = data[self.season].unique()
			season_team_r = pd.DataFrame()
			team_r = self._set_teams(data, rating_method='Glicko')

			for season in seasons:

				#turn original team ratings into preseason ratings
				team_r_pre = team_r.rename(columns={'rating': 'rating_preseason'}).drop('team_name', axis= 1)

				#get team ratings post season
				season_subset_r = self._set_ratings_glicko(data[data[self.season] == season], prev_ratings=team_r, h=h)

				#update team ratings
				team_r = team_r.combine_first(season_subset_r)

				#join preseason to post season ratings
				season_subset_r = season_subset_r.join(team_r_pre).rename(columns={'rating': 'rating_postseason'})

				#add season to pre and post season ratings
				season_subset_r[self.season] = season
				season_team_r = season_team_r.append(season_subset_r)

			season_team_r.sort_values([self.season, 'rating_postseason'], ascending = False, inplace = True)
			self.ratings_teams = team_r.sort_values('rating', ascending = False)
			self.ratings_teams_seasons = season_team_r

		elif results_by_league == True and results_by_season == True:

			leagues = data[self.league].unique()
			league_team_season_r = pd.DataFrame()

			for league in leagues:
				league_subset = data[data[self.league] == league]
				seasons = league_subset[self.season].unique()

				season_team_r = pd.DataFrame()
				team_r = self._set_teams(league_subset, rating_method='Glicko')

				for season in seasons:

					#turn original team ratings into preseason ratings
					team_r_pre = team_r.rename(columns={'rating': 'rating_preseason'}).drop('team_name', axis= 1)

					#get team ratings post season
					season_subset_r = self._set_ratings_glicko(data[data[self.season] == season], prev_ratings=team_r, h=h)

					#update team ratings
					team_r = team_r.update(season_subset_r)

					#join preseason to post season ratings
					season_subset_r = season_subset_r.join(team_r_pre).rename(columns={'rating': 'rating_postseason'})

					#add season to pre and post season ratings
					season_subset_r[self.season] = season
					season_team_r = season_team_r.append(season_subset_r)

				season_team_r[self.league] = league
				league_team_season_r = league_team_season_r.append(season_team_r)

			self.ratings_teams = league_team_season_r.sort_values([self.league, 'rating'], ascending = False)

		#calculate log loss of the predictions
		"""
		l_loss = log_loss(s, p)
		sq_err = squared_error(s, p)

		self.log_loss = l_loss
		self.squared_error = sq_err

		p_tie = np.full((self.total, ), 0.5, dtype=float)
		l_loss_tie = log_loss(s, p_tie)
		sq_err_tie = squared_error(s, p_tie)
		self.log_loss_tie = l_loss_tie
		self.squared_error_tie = sq_err_tie

		p_home = np.full((self.total, ), 0.999, dtype=float)
		l_loss_home = log_loss(s, p_home)
		sq_err_home = squared_error(s, p_home)
		self.log_loss_home = l_loss_home
		self.squared_error_tie = sq_err_home
		"""

		s = data['outcome'].values
		p = self.ratings_fixtures['localteam_p'].values
		ties_exist = ((s == 0.5).sum) != 0

		if ties_exist:
			self.ties_exist = True
		else:
			self.ties_exist = False
			self.true = s
			self.pred = p
			self.true_classes = s
			self.pred_classes = np.where(p > 0.5, 1, 0)

		if tie_probability == True and ties_exist:
			self.calculate_tie_prob(s, p)

		self.evaluate_predictions()

		end = time()

		progress(100, 100, status="Hooray!")
		print ('Calculations completed in %f seconds' % (end-start))


	def _set_ratings_glicko(self, data, prev_ratings=None, h=0):

		#Glicko = Glicko(c=c, start_rating=start_rating, Sigma=Sigma)

		if prev_ratings is None:
			teams = self._set_teams(data, rating_method='Glicko')
		else:
			teams = prev_ratings

		recarray = data.to_records(index=True)
		fixture_ratings = dict()
		new_local_ratings = dict()
		new_visitor_ratings = dict()

		for row in recarray:

			self.progress += 1
			progress(self.progress, self.total, status=row.Index)

			index = row[0]
			order = getattr(row, self.order)
			localteam_id = getattr(row, self.localteam_id)
			visitorteam_id = getattr(row, self.visitorteam_id)
			localteam_name = getattr(row, self.localteam_name)
			visitorteam_name = getattr(row, self.visitorteam_name)
			outcome = row.outcome
			visitorteam_outcome = row.visitorteam_outcome
			localteam_r = teams.loc[localteam_id]['rating']
			visitorteam_r = teams.loc[visitorteam_id]['rating']
			localteam_RD = teams.loc[localteam_id]['RD']
			visitorteam_RD = teams.loc[visitorteam_id]['RD']

			#calculate ratings
			localteam_p, visitorteam_p, localteam_post_r, \
			visitorteam_post_r, localteam_post_RD, visitorteam_post_RD = Glicko.glicko_rating(localteam_r, visitorteam_r,
																							  localteam_RD, visitorteam_RD, outcome,
																							  start_rating=self.start_rating, Sigma=self.Sigma, c=self.c, h=h)

			#assign probabilities for current game and post-game ratings
			fixture_ratings[index] = {'localteam_r': localteam_r,
											  'visitorteam_r': visitorteam_r,
											  'localteam_RD': localteam_RD,
											  'visitorteam_RD': visitorteam_RD,
											  'localteam_p': localteam_p,
											  'visitorteam_p': visitorteam_p,
											  'localteam_post_r': localteam_post_r,
											  'visitorteam_post_r': visitorteam_post_r,
											  'localteam_post_RD': localteam_post_RD,
											  'visitorteam_post_RD': visitorteam_post_RD}

			#assign probabilities for each team over time
			new_local_ratings[index] = {'id': index,
								'order': order,
								'position': 'local',
								'team_id': localteam_id,
								'team_name': localteam_name,
								'team_r': localteam_r,
								'team_RD': localteam_RD,
								'team_p': localteam_p,
								'outcome': outcome,
								'team_post_r': localteam_post_r,
								'team_post_RD': localteam_post_RD}
			new_visitor_ratings[index] = {'id': index,
									'order': order,
									'position': 'visitor',
									'team_id': visitorteam_id,
									'team_name': visitorteam_name,
									'team_r': visitorteam_r,
									'team_RD': visitorteam_RD,
									'team_p': visitorteam_p,
									'outcome': visitorteam_outcome,
									'team_post_r': visitorteam_post_r,
									'team_post_RD': visitorteam_post_RD}

			teams.loc[localteam_id, 'rating'] = localteam_post_r
			teams.loc[visitorteam_id, 'rating'] = visitorteam_post_r
			teams.loc[localteam_id, 'RD'] = localteam_post_RD
			teams.loc[visitorteam_id, 'RD'] = visitorteam_post_RD

		self.ratings_fixtures = self.ratings_fixtures.append(pd.DataFrame.from_dict(fixture_ratings, orient='index'), ignore_index = False).sort_values('result_order')
		self.ratings_teams_fixtures = self.ratings_teams_fixtures.append(pd.DataFrame.from_dict(new_local_ratings, orient='index'), ignore_index = True)
		self.ratings_teams_fixtures = self.ratings_teams_fixtures.append(pd.DataFrame.from_dict(new_visitor_ratings, orient='index'), ignore_index = True)

		return teams
