# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from time import time
from .helpers import progress, log_loss, squared_error
from .Elo import Elo
from .Glicko import Glicko 

"""
Possible improvements:
    1. Algorithmic
    - Change K based on tournament played
    - give high k factor to new players and lower as number of games played increases
    - root mean squared error to test for stability
    - Handle draws

    2. Function:
    - can turn calculations into lists
    - logging
    - parallel computing (not sure if this can be done)

    3. Other rating systems:
    - Glicko, Glicko_2
    - TrueSkill
    - FIDE
    - Whole History (https://www.remi-coulom.fr/WHR/WHR.pdf)
    - Edo
    - Decayed History
    - pi-rating

"""

start_rating = 1500
Sigma = 350
c = 50

Elo = Elo()
Glicko = Glicko()

class Rate(object):
    
    """
    A python object to rank sports teams based on different rating systems e.g. Elo Ratings and Glicko Ratings

        :param localteam_id: name of column for localteam id, defaults to 'localteam_id'
        :param visitorteam_id: name of column for visitorteam id, defaults to 'visitorteam_id'
        :param localteam_name: name of column for localteam name, defaults to 'localteam_name'
        :param visitorteam_name: name of column for visitorteam name, defaults to 'visitorteam_name'
        :param localteam_score: name of column for localteam score, defaults to 'localteam_score'
        :param visitorteam_score: name of column for visitorteam score, defaults to 'visitorteam_score'
        :param order: name of column used to sort fixtures, defaults to 'starting_datetime'
        :param league: name of column used to identify league, defaults to 'league_name'
        :param season: name of column used to identify season, defaults to 'season_name'

        Resulting dataframes that can be accessed after calculate_elo is run:
            ratings_fixures: ratings before and after each fixture for each team, as well as probability of each team winning
            ratings_teams: most recent ratings for each team in the fixture list
            ratings_teams_fixtures: ratings for each team fixture over time - can be used to plot trends
            ratings_teams_seasons: ratings for each team in each season (only available when results_by_season is set to true) 
    """
    
    def __init__(self,
                localteam_id = 'localteam_id', visitorteam_id = 'visitorteam_id',
                localteam_name = 'localteam_name', visitorteam_name = 'visitorteam_name',
                localteam_score = 'localteam_score', visitorteam_score = 'visitorteam_score',
                order = 'starting_datetime', league = 'league_name', season = 'season_name'):
        
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
        
        self.log_loss = 0
        self.log_loss_tie = 0
        self.log_loss_home = 0
        self.squared_error = 0
        self.squared_error_tie = 0
        self.squared_error_home = 0
        self.accuracy = 0
        
        self.progress = 0
        
        self.start_rating = start_rating
        self.Sigma = Sigma
        self.c = c
        
    
    def _set_teams(self, data, rating_method):
        
        teams_local = data[[self.localteam_id, self.localteam_name]] \
                    .drop_duplicates() \
                    .rename(columns={self.localteam_id: 'team_id', self.localteam_name: 'team_name'})

        teams_visitor = data[[self.visitorteam_id, self.visitorteam_name]] \
                        .drop_duplicates() \
                        .rename(columns={self.visitorteam_id: 'team_id', self.visitorteam_name: 'team_name'})

        teams = teams_local.append(teams_visitor, ignore_index = True).drop_duplicates()
        teams['rating'] = self.start_rating
        
        if rating_method == 'Glicko':
            teams['RD'] = self.Sigma
            
        teams.set_index('team_id', inplace = True)   
            
        return teams
    
    
    def _set_team_id(self, data):
        
        teams_local = data[[self.localteam_name]].drop_duplicates().rename(columns={self.localteam_name: 'team_name'})
        teams_visitor = data[[self.visitorteam_name]].drop_duplicates().rename(columns={self.visitorteam_name: 'team_name'})
        teams = teams_local.append(teams_visitor, ignore_index = True).drop_duplicates()
        teams['team_id'] = teams.index.values
        
        data = data.merge(teams, left_on = self.localteam_name,  right_on = 'team_name').rename(columns={'team_id': 'localteam_id'})
        data = data.merge(teams, left_on = self.visitorteam_name, right_on = 'team_name').rename(columns={'team_id': 'visitorteam_id'})
        
        self.localteam_id = 'localteam_id'
        self.visitorteam_id = 'visitorteam_id'
        
        return data


    def _process_data(self, data):

        #Determine outcome for each fixture
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

        #Sort fixtures
        data.sort_values(self.order, inplace = True)
        
        if (self.localteam_id == '') | (self.visitorteam_id == '') :
            data = self._set_team_id(data)

        return data


    def calculate_elo(self, data, K, h=0, Sd=False, Sd_method='Logistic', m=1, results_by_league=False, results_by_season=False, 
                      rt_mean=False, rt_mean_amnt=1/3, time_scale=False, time_scale_factor=2, start_rating=start_rating):
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

        Note that if results_by_league is False and results_by_season is True, only one league should be passed as data
        """
        self.ratings_fixtures = pd.DataFrame()
        self.ratings_teams_fixtures = pd.DataFrame()
        self.ratings_teams_seasons = pd.DataFrame()
        self.progress = 0

        start = time()
        
        if start_rating != self.start_rating:
            self.start_rating = start_rating

        print ('Prepping data...')
        data = self._process_data(data)

        print ('Starting calculations...')
        self.total = len(data.index)

        if results_by_league == False and results_by_season == False:

            ratings_teams = self._set_ratings_elo(data, K, h=h, Sd=Sd, Sd_method=Sd_method, m=m, time_scale=time_scale, time_scale_factor=time_scale_factor)
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
            print('Results being calculated by season, please make sure that only one league is passed in the data. If you would like to pass multiple leagues, please set "results_by_league to True')

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

        end = time()
        
        progress(100, 100, status="Hooray!")
        print ('Calculations completed in %f seconds with a log loss of %f and a mean squared error of %f' % (end-start, l_loss, sq_err))
        print ('All games tied: log loss of %f and a mean squared error of %f; all games won by home team: log loss of %f and a mean squared error of %f' % (l_loss_tie, sq_err_tie, l_loss_home, sq_err_home))


    def _set_ratings_elo(self, data, K, h=0, Sd=False, Sd_method='Logistic', m=1, prev_ratings=None, time_scale=False, time_scale_factor=2):

        if prev_ratings is None:
            teams = self._set_teams(data, rating_method='Elo')
        else:
            teams = prev_ratings        

        if Sd == False:
            ratings_function = 'Elo.elo_rating(localteam_r, visitorteam_r, K, outcome, start_rating=self.start_rating, score_diff=0, h=h)'
        else:
            ratings_function = 'Elo.elo_rating(localteam_r, visitorteam_r, K, outcome, score_diff=score_diff, start_rating=self.start_rating, Sd_method=Sd_method, m=m, h=h)'
        
        for row in data.itertuples(index=True):
            
            self.progress += 1
            progress(self.progress, self.total, status=row.Index)

            index = row.Index
            order = getattr(row, self.order)
            localteam_id = getattr(row, self.localteam_id)
            visitorteam_id = getattr(row, self.visitorteam_id)
            localteam_name = getattr(row, self.localteam_name)
            visitorteam_name = getattr(row, self.visitorteam_name)
            outcome = row.outcome
            visitorteam_outcome = row.visitorteam_outcome
            localteam_r = teams.loc[localteam_id]['rating']
            visitorteam_r = teams.loc[visitorteam_id]['rating']
            score_diff = np.abs(getattr(row, self.localteam_score) - getattr(row, self.visitorteam_score))

            #calculate ratings
            localteam_p, visitorteam_p, localteam_post_r, visitorteam_post_r = eval(ratings_function)

            #assign probabilities for current game and post-game ratings
            new_rating = pd.DataFrame(data = {'localteam_r': localteam_r,
                                              'visitorteam_r': visitorteam_r,
                                              'localteam_p': localteam_p,
                                              'visitorteam_p': visitorteam_p,
                                              'localteam_post_r': localteam_post_r,
                                              'visitorteam_post_r': visitorteam_post_r},
                                    index = [index])
            self.ratings_fixtures = self.ratings_fixtures.append(new_rating)

            #update team rating without time scaling
            if time_scale == False:
                teams.loc[localteam_id, 'rating'] = localteam_post_r
                teams.loc[visitorteam_id, 'rating'] = visitorteam_post_r
                
                #assign probabilities for each team over time
                new_local_rating = {'id': index,
                                    'order': order,
                                    'position': 'local',
                                    'team_id': localteam_id,
                                    'team_name': localteam_name,
                                    'team_r': localteam_r,
                                    'team_p': localteam_p,
                                    'outcome': outcome,
                                    'team_post_r': localteam_post_r}
                new_visitor_rating = {'id': index,
                                        'order': order,
                                        'position': 'visitor',
                                        'team_id': visitorteam_id,
                                        'team_name': visitorteam_name,
                                        'team_r': visitorteam_r,
                                        'team_p': visitorteam_p,
                                        'outcome': visitorteam_outcome,
                                        'team_post_r': visitorteam_post_r}
                self.ratings_teams_fixtures = self.ratings_teams_fixtures.append([new_local_rating, new_visitor_rating], ignore_index = True)
                
            #update team rating with time scaling
            elif time_scale == True:
                
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
                new_local_rating = {'id': index,
                                    'order': order,
                                    'position': 'local',
                                    'team_id': localteam_id,
                                    'team_name': localteam_name,
                                    'team_r': localteam_r,
                                    'team_p': localteam_p,
                                    'outcome': outcome,
                                    'team_post_r': localteam_post_r,
                                    'team_post_r_ts': lteam_post_r_ts}
                new_visitor_rating = {'id': index,
                                        'order': order,
                                        'position': 'visitor',
                                        'team_id': visitorteam_id,
                                        'team_name': visitorteam_name,
                                        'team_r': visitorteam_r,
                                        'team_p': visitorteam_p,
                                        'outcome': visitorteam_outcome,
                                        'team_post_r': visitorteam_post_r,
                                        'team_post_r_ts': vteam_post_r_ts}
                self.ratings_teams_fixtures = self.ratings_teams_fixtures.append([new_local_rating, new_visitor_rating], ignore_index = True)
                
        return teams
    
    
    def calculate_glicko(self, data, c=c, start_rating=start_rating, Sigma=Sigma, h=0, results_by_league=False, results_by_season=False):
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
    
        Note that if results_by_league is False and results_by_season is True, only one league should be passed as data
        """
        self.ratings_fixtures = pd.DataFrame()
        self.ratings_teams_fixtures = pd.DataFrame()
        self.ratings_teams_seasons = pd.DataFrame()
        self.progress = 0
        
        if start_rating != self.start_rating:
            self.start_rating = start_rating
        
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
    
        end = time()
        
        progress(100, 100, status="Hooray!")
        print ('Calculations completed in %f seconds with a log loss of %f and a mean squared error of %f' % (end-start, l_loss, sq_err))
        print ('All games tied: log loss of %f and a mean squared error of %f; all games won by home team: log loss of %f and a mean squared error of %f' % (l_loss_tie, sq_err_tie, l_loss_home, sq_err_home))
        
    
    def _set_ratings_glicko(self, data, prev_ratings=None, h=0):
        
        #Glicko = Glicko(c=c, start_rating=start_rating, Sigma=Sigma)

        if prev_ratings is None:
            teams = self._set_teams(data, rating_method='Glicko')
        else:
            teams = prev_ratings        
        
        for row in data.itertuples(index=True):
            
            self.progress += 1
            progress(self.progress, self.total, status=row.Index)

            index = row.Index
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
            new_rating = pd.DataFrame(data = {'localteam_r': localteam_r,
                                              'visitorteam_r': visitorteam_r,
                                              'localteam_RD': localteam_RD,
                                              'visitorteam_RD': visitorteam_RD,
                                              'localteam_p': localteam_p,
                                              'visitorteam_p': visitorteam_p,
                                              'localteam_post_r': localteam_post_r,
                                              'visitorteam_post_r': visitorteam_post_r,
                                              'localteam_post_RD': localteam_post_RD,
                                              'visitorteam_post_RD': visitorteam_post_RD},
                                    index = [index])
            self.ratings_fixtures = self.ratings_fixtures.append(new_rating)
            
            #assign probabilities for each team over time
            new_local_rating = {'id': index,
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
            new_visitor_rating = {'id': index,
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
            self.ratings_teams_fixtures = self.ratings_teams_fixtures.append([new_local_rating, new_visitor_rating], ignore_index = True)

            teams.loc[localteam_id, 'rating'] = localteam_post_r
            teams.loc[visitorteam_id, 'rating'] = visitorteam_post_r
            teams.loc[localteam_id, 'RD'] = localteam_post_RD
            teams.loc[visitorteam_id, 'RD'] = visitorteam_post_RD
                
        return teams
