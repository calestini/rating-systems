# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
import time
import sys

"""
 Possible improvements:
     
     1. Algorithmic
     - Change K based on tournament played
     - Change K based on goal difference (Margin of Victory Multiplier = LN(ABS(PD)+1) * (2.2/((ELOW-ELOL)*.001+2.2)))
     - Change difference in ratings based on home/away (add 100 points)
     https://fivethirtyeight.com/features/introducing-nfl-elo-ratings/
     https://www.eloratings.net/about
     https://www.ufs.ac.za/docs/librariesprovider22/mathematical-statistics-and-actuarial-science-documents/technical-reports-documents/teg418-2069-eng.pdf?sfvrsn=243cf921_0
     https://math.stackexchange.com/questions/850002/improving-the-elo-rating-system-to-account-for-game-results
     
     2. Function:
     - Check calculation by season/league
     - Fix progress bar
"""

class Elo(object):
    
    """
    A python object to calculate Elo ratings
    
        :param localteam_id: name of column for localteam id, defaults to 'localteam_id'
        :param visitorteam_id: name of column for visitorteam id, defaults to 'visitorteam_id'
        :param localteam_name: name of column for localteam name, defaults to 'localteam_name'
        :param visitorteam_name: name of column for visitorteam name, defaults to 'visitorteam_name'
        :param localteam_score: name of column for localteam score, defaults to 'localteam_score'
        :param visitorteam_score: name of column for visitorteam score, defaults to 'visitorteam_score'
        :param order: name of column used to sort fixtures, defaults to 'starting_datetime'
        :param league: name of column used to identify league, defaults to 'league_name'
        :param season: name of column used to identify season, defaults to 'season_name'
        
        Resulting dataframes that can be accessed after calcualte_elo is run:
            ratings: ratings before and after each fixture for each team, as well as probability of each team winning
            teams: ratings for each team in the fixture list
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
        
    
    def _progress(self, count, total, status=''):
        """
        Adapted from https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
        """
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('[{0}] {1}{2} ... {3}\r'.format(bar, percents, '%', status))
        sys.stdout.flush()
        

    def probability(self, rating1, rating2):
        p1_w = 1.0 * 1.0 / (1 + 1.0 * math.pow(10, 1.0 * (rating2 - rating1) / 400))
        p2_w = 1.0 * 1.0 / (1 + 1.0 * math.pow(10, 1.0 * (rating1 - rating2) / 400))
        #note that 400 is width, we can change this

        return p1_w, p2_w
    

    def elo_rating(self, Ra, Rb, K, d):

        # Calculate winning probabilities
        Pa, Pb = self.probability(Ra, Rb)

        # Case -1 When Player A wins
        # Updating the Elo Ratings
        if (d == 1) :
            Ra = Ra + K * (1 - Pa)
            Rb = Rb + K * (0 - Pb)    

        # Case -2 When Player B wins
        # Updating the Elo Ratings
        elif (d == 0.5):
            Ra = Ra + K * (0.5 - Pa)
            Rb = Rb + K * (0.5 - Pb)
            
        # Case -3 When it's a tie
        # Updating the Elo Ratings
        else:
            Ra = Ra + K * (0 - Pa)
            Rb = Rb + K * (1 - Pb)   

        return Pa, Pb, Ra, Rb
    
    
    def _set_teams(self, data):
    
        teams_local = data[[self.localteam_id, self.localteam_name]] \
                    .drop_duplicates() \
                    .rename(index=str, columns={self.localteam_id: 'team_id', self.localteam_name: 'team_name'})
                
        teams_visitor = data[[self.visitorteam_id, self.visitorteam_name]] \
                        .drop_duplicates() \
                        .rename(index=str, columns={self.visitorteam_id: 'team_id', self.visitorteam_name: 'team_name'})
        
        teams = teams_local.append(teams_visitor, ignore_index = True).drop_duplicates()
        teams['rating'] = 1500
        teams.set_index('team_id', inplace = True)
        
        return teams
        
        
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
        
        return data
        
    
    def calculate_elo(self, data, K, results_by_league = False, results_by_season = False):
        """
        Calculate elo rating for fixtures, accepts a dataframe
        
        :param results_by_league: calculate ratings for each league individually, defaults to False
        :param results_per_season: calculate ratings for each season individually, defaults to False
        
        Note that if results_by_league is False and results_by_season is True, only one league should be passed as data 
        """
        
        #start = time()
        #total = len(data.index)
        
        print ('Prepping data...')
        data = self._process_data(data)
        self._set_teams(data)
        
        print ('Starting calculations...')
        
        if results_by_league == False and results_by_season == False:
            
            ratings_teams = self._set_ratings(data, K)
            ratings_teams.sort_values('rating', ascending = False, inplace = True)
            self.ratings_teams = ratings_teams
        
        elif results_by_league == True and results_by_season == False:
            
            leagues = data[self.league].unique()
            league_team_r = pd.DataFrame()
            
            for league in leagues:
                league_subset = data[data[self.league] == league]
                league_subset_r = self._set_ratings(league_subset, K)
                league_subset_r['league'] = league
                league_team_r = league_team_r.append(league_subset_r)
            
            league_team_r.sort_values([self.league, 'rating'], ascending = False, inplace = True)
            self.ratings_teams = league_team_r
        
        elif results_by_league == False and results_by_season == True:
            print('Results being calculated by season, please make sure that only one league is passed in the data')
            
            seasons = data[self.season].unique()
            season_team_r = pd.DataFrame()
            team_r = self._set_teams(data)
            
            for season in seasons:
                season_subset = data[data[self.season] == season]
                
                #get team ratings post season
                season_subset_r = self._set_ratings(season_subset, K, prev_ratings = team_r)
                
                #update team ratings
                team_r = season_subset_r.combine_first(team_r)
                
                #turn original team ratings into preseason ratings
                team_r_pre = team_r.rename(index=str, columns={'rating': 'rating_preseason'}).drop('team_name', axis= 1)
                
                #join preseason to post season ratings
                season_subset_r.join(team_r_pre, how = 'left').rename(index=str, columns={'rating': 'rating_postseason'})
                
                #add season to pre and post season ratings
                season_subset_r[self.season] = season
                season_team_r = season_team_r.append(season_subset_r)
            
            season_team_r.sort_values([self.season, 'rating'], ascending = False, inplace = True)
            self.ratings_teams = team_r.sort_values('rating', ascending = False)
            self.ratings_teams_seasons = season_team_r
                
        elif results_by_league == True and results_by_season == True:
            
            leagues = data[self.league].unique()
            league_team_season_r = pd.DataFrame()
            
            for league in leagues:
                league_subset = data[data[self.league] == league]
                seasons = league_subset[self.season].unique()
                
                season_team_r = pd.DataFrame()
                team_r = self._set_teams(league)
                
                for season in seasons:
                 season_subset = data[data[self.season] == season]
                
                #get team ratings post season
                season_subset_r = self._set_ratings(season_subset, K, prev_ratings = team_r)
                
                #update team ratings
                team_r = season_subset_r.combine_first(team_r)
                
                #turn original team ratings into preseason ratings
                team_r_pre = team_r.rename(index=str, columns={'rating': 'rating_preseason'}).drop('team_name', axis= 1)
                
                #join preseason to post season ratings
                season_subset_r.join(team_r_pre, how = 'left').rename(index=str, columns={'rating': 'rating_postseason'})
                
                #add season to pre and post season ratings
                season_subset_r[self.season] = season
                season_team_r = season_team_r.append(season_subset_r)
                    
                season_team_r['league'] = league
                league_team_season_r = league_team_season_r.append[season_team_r]
            
            self.ratings_teams = league_team_season_r
        
        #end = time()
        #self._progress(100,100, status="Hoooray!!")
        #print ('Calculations completed in {} seconds'.format(end-start))
        
        return self.ratings_fixtures
    
    
    def _set_ratings(self, data, K, prev_ratings = None):
        
        if prev_ratings is None:
            teams = self._set_teams(data)
        else:
            teams = prev_ratings
    
        for count, row in enumerate(data.itertuples(index=True), 1):
            
            #self._progress(count, total, status=row.Index)

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

            #calculate ratings
            localteam_p, visitorteam_p, localteam_post_r, visitorteam_post_r = self.elo_rating(localteam_r, visitorteam_r, K, outcome)

            #assign probabilities for current game and post-game ratings
            new_rating = pd.DataFrame(data = {'localteam_r': localteam_r, 
                                              'visitorteam_r': visitorteam_r, 
                                              'localteam_p': localteam_p, 
                                              'visitorteam_p': visitorteam_p, 
                                              'localteam_post_r': localteam_post_r,
                                              'visitorteam_post_r': visitorteam_post_r},
                                    index = [index])
            self.ratings_fixtures = self.ratings_fixtures.append(new_rating)
            
            #assign probabilities for each team over time
            new_local_rating = pd.DataFrame(data = {'id': index,
                                                    'order': order,
                                                    'position': 'local',
                                                    'team_id': localteam_id,
                                                    'team_name': localteam_name,
                                                    'team_r': localteam_r,
                                                    'team_p': localteam_p,
                                                    'outcome': outcome,
                                                    'team_post_r': localteam_post_r}, 
                                            index = [1])
            new_visitor_rating = pd.DataFrame(data = {'id': index,
                                                    'order': order,
                                                    'position': 'visitor',
                                                    'team_id': visitorteam_id,
                                                    'team_name': visitorteam_name,
                                                    'team_r': visitorteam_r,
                                                    'team_p': visitorteam_p,
                                                    'outcome': visitorteam_outcome,
                                                    'team_post_r': visitorteam_post_r},
                                            index = [2])
            self.ratings_teams_fixtures = self.ratings_teams_fixtures.append([new_local_rating, new_visitor_rating], ignore_index = True)

            #update team rating
            teams.loc[localteam_id, 'rating'] = localteam_post_r
            teams.loc[visitorteam_id, 'rating'] = visitorteam_post_r
            
        return teams
        

    
