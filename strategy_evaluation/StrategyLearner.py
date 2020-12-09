""""""  		  	   		     		  		  		    	 		 		   		 		  
"""  		  	   		     		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		     		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		     		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		     		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		     		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		     		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		     		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		     		  		  		    	 		 		   		 		  
or edited.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		     		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		     		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		     		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		  	   		     		  		  		    	 		 		   		 		  
GT User ID: tb34 (replace with your User ID)  		  	   		     		  		  		    	 		 		   		 		  
GT ID: 900897987 (replace with your GT ID)  		  	   		     		  		  		    	 		 		   		 		  
"""  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		     		  		  		    	 		 		   		 		  
import random  		  	   		     		  		  		    	 		 		   		 		  
import indicators as ind  		  	   		     		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		     		  		  		    	 		 		   		 		  
import util as ut  		  
import numpy as np	
import QLearner as ql 
import matplotlib as mat  		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
class StrategyLearner(object):  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		     		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output.  		  	   		     		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		     		  		  		    	 		 		   		 		  
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		     		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		     		  		  		    	 		 		   		 		  
    :param commission: The commission amount charged, defaults to 0.0  		  	   		     		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    # constructor  		  	   		     		  		  		    	 		 		   		 		  
    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        Constructor method  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        self.verbose = verbose  		  	   		     		  		  		    	 		 		   		 		  
        self.impact = impact  		  	   		     		  		  		    	 		 		   		 		  
        self.commission = commission  
        self.learner = ql.QLearner(num_states=1000, num_actions=3, alpha=0.2, gamma=0.9, rar=0.5, radr=0.99, dyna=0, verbose=False)	

    def author(self):
        return 'jlaurent6' # replace tb34 with your Georgia Tech username.
  		  	   		     		  		  		    	 		 		   		 		  
    # this method should create a QLearner, and train it for trading  		  	   		     		  		  		    	 		 		   		 		  
    def add_evidence(  		  	   		   #this is where you could use the inspiration for the grading function from project 7  		  		  		    	 		 		   		 		  
        self,  		  	   		     		  		  		    	 		 		   		 		  
        symbol="JPM",  		  	   		     		  		  		    	 		 		   		 		  
        sd=dt.datetime(2008, 1, 1),  		  	   		     		  		  		    	 		 		   		 		  
        ed=dt.datetime(2009, 1, 1),  		  	   		     		  		  		    	 		 		   		 		  
        sv=10000,  		  	   		     		  		  		    	 		 		   		 		  
    ):  



        """  		  	   		     		  		  		    	 		 		   		 		  
        Trains your strategy learner over a given time frame.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol to train on  		  	   		     		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		     		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		     		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		     		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		     		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		     		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		     		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
        # add your code to do learning here  		  	   		     		  		  		    	 		 		   		 		  
        syms = [symbol]  		  	   		     		  		  		    	 		 		   		 		  
        dates = pd.date_range(sd, ed) 
        indicator_dates = pd.date_range(sd-dt.timedelta(days=28), ed)  		  	   		     		  		  		    	 		 		   		 		  
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY  		  	   		     		  		  		    	 		 		   		 		  
        prices = prices_all[syms]  # only portfolio symbols  		
        indicator_prices_all = ut.get_data(syms, indicator_dates)   #will this be the same size as the number of days traded?
        indicator_prices = indicator_prices_all[syms]		  		  		    	 		 		   		 		  
        prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		     		  		  		    	 		 		   		 		  
        

        #get 3 indicators using passed in start date minus window required for each
        rm = ind.get_rolling_mean(indicator_prices[symbol], window = 20)
        percent_bb = ind.get_rolling_std(indicator_prices[symbol], window = 20) #aka rstd
        upper_band, lower_band = ind.get_bollinger_bands(rm, percent_bb)

        percent_bb = percent_bb.iloc[19:]
        rm = rm.iloc[19:]
        upper_band = upper_band.iloc[19:]
        lower_band = lower_band.iloc[19:]
        for x in range(rm.size):  #did i format that right??
            #print(prices.iloc[x])
            #print(type(rm.iloc[x]))
            if(prices.iloc[x][0] == rm.iloc[x]):
                percent_bb.iloc[x] = 0.0
            elif(prices.iloc[x][0] > rm.iloc[x]):
                percent_bb.iloc[x] = (prices.iloc[x][0] - rm.iloc[x]) / (upper_band.iloc[x] - rm.iloc[x])
            elif(prices.iloc[x][0] < rm.iloc[x]):
                percent_bb.iloc[x] = (rm.iloc[x] - prices.iloc[x][0]) / (lower_band.iloc[x] - rm.iloc[x])
        
        percent_bb = pd.Series(percent_bb)
        #print(percent_bb)
        indicator_prices = indicator_prices.iloc[9:]
        #print(indicator_prices)
        momentum = pd.Series(ind.get_momentum(indicator_prices[symbol], window = 10))
        momentum = momentum.iloc[10:]
        #print(momentum)
        volatility = pd.Series(ind.get_volatility(indicator_prices[symbol], window = 10))
        volatility = volatility.iloc[10:]
        #print(volatility.values.shape)
                                                                                            
        uglystupidnobodylikesyouvol, vol = pd.qcut(volatility.values, 10, retbins=True)
        esrfergs, momnt = pd.qcut(momentum.values, 10, retbins=True)
        rgefsegrf, bb = pd.qcut(percent_bb.values, 10, retbins=True)
        vol = vol[1:-1]
        momnt = momnt[1:-1]
        bb = bb[1:-1]
        #print(volatility.iloc[0])
        #print(vol)
        #print(volatility.iloc[0] in vol[0])   		



        total_reward = 0
        performance_list = np.zeros(10)
        for z in range(10):
            cash_value = float(sv)
            stock_value =0.0
            portfolio_value = cash_value + stock_value
            #figure out state
            vol_set = False
            mo_set = False
            bb_set = False
            vol_bin= ""
            mo_bin = ""
            bb_bin = ""
            
            for x in range(vol.size):
                if(volatility.iloc[0] <= vol[x] and vol_set == False):
                    
                    vol_bin = str(x)
                    vol_set=True

                elif(x==vol.size-1 and vol_set == False):
                    vol_bin = str(x+1)
                    vol_set=True
                if(momentum.iloc[0] <= momnt[x] and mo_set == False):
                    mo_bin = str(x)
                    mo_set=True
                elif(x==momnt.size-1 and mo_set == False):
                    mo_bin = str(x+1)
                    mo_set=True
                if(percent_bb.iloc[0] <= bb[x] and bb_set == False):
                    bb_bin = str(x)
                    bb_set = True
                elif(x==bb.size-1 and bb_set == False):
                    bb_bin = str(x+1)  
                    bb_set = True
                

            #print(volatility.iloc[0])
            #position = 0   actually this could be action variable below
            final_str = vol_bin + mo_bin + bb_bin
            state = int(final_str)
            #print(state)
            action = self.learner.querysetstate(  		  	   		     		  		  		    	 		 		   		 		  
                state  		  	   		     		  		  		    	 		 		   		 		  
            )-1
            old_action = action
            same_action = False
            count=0
            stock_value = action*1000.0*prices.iloc[0]
            cash_value -= stock_value

            #print("Daily Value for end of Day 1" + ": " + str(portfolio_value))
            for x in range(1, prices.size):   #while the trading simualtion hasnt reached last day

                #at this point assume day 1 or x is over and now calculate the reward of the prior day's 
                #decision to pass in next action call as r. also calculate the 
                # holdings of the bot(maybe just check the action variable for this)
                daily_gain = (prices.values[x,0]-prices.values[x-1,0])/prices.values[x-1,0]  #daily percentage gain
                
                #print(type(daily_gain))
                portfolio_value = ((1.0+daily_gain)*stock_value) + cash_value
                #print(portfolio_value)
                
                if(action == 1):
                    if(daily_gain-self.impact >0.02):
                        reward = 35.0
                    elif(daily_gain-self.impact >0.012):
                        reward = 20.0
                    elif(daily_gain-self.impact>0.004):
                        reward = 7.5
                    elif(daily_gain-self.impact < -0.019):
                        reward = -25.0
                    elif(daily_gain-self.impact < -0.009):
                        reward = -10.0
                    else:
                        reward = daily_gain*10.0
                elif(action == 0):
                    reward = abs(daily_gain-self.impact)*3.5
                elif(action == -1):
                    if(daily_gain-self.impact <-0.026):
                        reward = 35.0
                    elif(daily_gain-self.impact <-0.015):
                        reward = 20.0
                    elif(daily_gain-self.impact <-0.004):
                        reward = 7.5
                    elif(daily_gain-self.impact > 0.019):
                        reward = -25.0
                    elif(daily_gain-self.impact > 0.007):
                        reward = -10.0
                    else:
                        reward = daily_gain*-10.0
                
                if(same_action==True):
                    reward+=(7.5+self.impact)
                #determine state
                vol_set = False
                mo_set = False
                bb_set = False
                vol_bin= ""
                mo_bin = ""
                bb_bin = ""
                for y in range(vol.size):
                    if(volatility.iloc[x] <= vol[y] and vol_set == False):
                        #print(x)
                        vol_bin = str(y)
                        vol_set=True
                    elif(y==vol.size-1 and vol_set == False):
                        vol_bin = str(y+1)
                        vol_set=True
                    if(momentum.iloc[x] <= momnt[y] and mo_set == False):
                        mo_bin = str(y)
                        mo_set=True
                    elif(y==momnt.size-1 and mo_set == False):
                        mo_bin = str(y+1)
                        mo_set=True
                    if(percent_bb.iloc[x] <= bb[y] and bb_set == False):
                        bb_bin = str(y)
                        bb_set = True
                    elif(y==bb.size-1 and bb_set == False):
                        bb_bin = str(y+1)  
                        bb_set = True 
                state = int(vol_bin+mo_bin+bb_bin)
                old_action = action	
                action = self.learner.query(state, reward) - 1 
                same_action = False

                if(old_action==action and reward > 0.0):
                    same_action = True
                stock_value = action*1000.0*prices.iloc[x]
                cash_value = portfolio_value - stock_value
                total_reward += reward
                
                #print("Daily Value for end of Day " + str(x+1) + ": " + str(portfolio_value))
            #print("Ending value for test phase " + str(z+1) + ": " + str(portfolio_value))
            performance_list[z] = portfolio_value
        mat.pyplot.plot(performance_list)
        #mat.pyplot.show()

    def discretize(self, symbol="IBM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31)):
        syms = [symbol]  		  	   		     		  		  		    	 		 		   		 		  
        dates = pd.date_range(sd, ed) 
        indicator_dates = pd.date_range(sd-dt.timedelta(days=28), ed)  		  	   		     		  		  		    	 		 		   		 		  
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY  		  	   		     		  		  		    	 		 		   		 		  
        prices = prices_all[syms]  # only portfolio symbols  		
        indicator_prices_all = ut.get_data(syms, indicator_dates)   #will this be the same size as the number of days traded?
        indicator_prices = indicator_prices_all[syms]		  		  		    	 		 		   		 		  
        prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		     		  		  		    	 		 		   		 		  
        

        #get 3 indicators using passed in start date minus window required for each
        rm = ind.get_rolling_mean(indicator_prices[symbol], window = 20)
        percent_bb = ind.get_rolling_std(indicator_prices[symbol], window = 20) #aka rstd
        upper_band, lower_band = ind.get_bollinger_bands(rm, percent_bb)

        percent_bb = percent_bb.iloc[19:]
        rm = rm.iloc[19:]
        upper_band = upper_band.iloc[19:]
        lower_band = lower_band.iloc[19:]
        for x in range(rm.size):  #did i format that right??
            #print(prices.iloc[x])
            #print(type(rm.iloc[x]))
            if(prices.iloc[x][0] == rm.iloc[x]):
                percent_bb.iloc[x] = 0.0
            elif(prices.iloc[x][0] > rm.iloc[x]):
                percent_bb.iloc[x] = (prices.iloc[x][0] - rm.iloc[x]) / (upper_band.iloc[x] - rm.iloc[x])
            elif(prices.iloc[x][0] < rm.iloc[x]):
                percent_bb.iloc[x] = (rm.iloc[x] - prices.iloc[x][0]) / (lower_band.iloc[x] - rm.iloc[x])
        
        percent_bb = pd.Series(percent_bb)
        #print(percent_bb)
        indicator_prices = indicator_prices.iloc[9:]
        #print(indicator_prices)
        momentum = pd.Series(ind.get_momentum(indicator_prices[symbol], window = 10))
        momentum = momentum.iloc[10:]
        #print(momentum)
        volatility = pd.Series(ind.get_volatility(indicator_prices[symbol], window = 10))
        volatility = volatility.iloc[10:]
        #print(volatility.values.shape)
                                                                                            
        uglystupidnobodylikesyouvol, vol = pd.qcut(volatility.values, 10, retbins=True)
        esrfergs, momnt = pd.qcut(momentum.values, 10, retbins=True)
        rgefsegrf, bb = pd.qcut(percent_bb.values, 10, retbins=True)
        vol = vol[1:-1]
        momnt = momnt[1:-1]
        bb = bb[1:-1]  		 

        return vol, momnt, bb

    # this method should use the existing policy and test it against new data  		  	   		     		  		  		    	 		 		   		 		  
    def testPolicy(  		  	   		     		  		  		    	 		 		   		 		  
        self,  		  	   		     		  		  		    	 		 		   		 		  
        symbol="IBM",  		  	   		     		  		  		    	 		 		   		 		  
        sd=dt.datetime(2008, 1, 1),  		  	   		     		  		  		    	 		 		   		 		  
        ed=dt.datetime(2009, 1, 31),  		  	   		     		  		  		    	 		 		   		 		  
        sv=10000,  		  	   		     		  		  		    	 		 		   		 		  
    ):  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        Tests your learner using data outside of the training data  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol that you trained on on  		  	   		     		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		     		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		     		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		     		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		     		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		     		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		     		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		     		  		  		    	 		 		   		 		  
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		     		  		  		    	 		 		   		 		  
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		     		  		  		    	 		 		   		 		  
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		     		  		  		    	 		 		   		 		  
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		     		  		  		    	 		 		   		 		  
        :rtype: pandas.DataFrame  		

        """  		  	 
        syms = [symbol]  		  	   		     		  		  		    	 		 		   		 		  
        dates = pd.date_range(sd, ed) 
        indicator_dates = pd.date_range(sd-dt.timedelta(days=28), ed)  		  	   		     		  		  		    	 		 		   		 		  
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY  		  	   		     		  		  		    	 		 		   		 		  
        prices = prices_all[syms]  # only portfolio symbols  		
        indicator_prices_all = ut.get_data(syms, indicator_dates)   #will this be the same size as the number of days traded?
        indicator_prices = indicator_prices_all[syms]		  		  		    	 		 		   		 		  
        prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  

        rm = ind.get_rolling_mean(indicator_prices[symbol], window = 20)
        percent_bb = ind.get_rolling_std(indicator_prices[symbol], window = 20) #aka rstd
        upper_band, lower_band = ind.get_bollinger_bands(rm, percent_bb)

        percent_bb = percent_bb.iloc[19:]
        rm = rm.iloc[19:]
        upper_band = upper_band.iloc[19:]
        lower_band = lower_band.iloc[19:]
        for x in range(rm.size):  #did i format that right??
            #print(prices.iloc[x])
            #print(type(rm.iloc[x]))
            if(prices.iloc[x][0] == rm.iloc[x]):
                percent_bb.iloc[x] = 0.0
            elif(prices.iloc[x][0] > rm.iloc[x]):
                percent_bb.iloc[x] = (prices.iloc[x][0] - rm.iloc[x]) / (upper_band.iloc[x] - rm.iloc[x])
            elif(prices.iloc[x][0] < rm.iloc[x]):
                percent_bb.iloc[x] = (rm.iloc[x] - prices.iloc[x][0]) / (lower_band.iloc[x] - rm.iloc[x])
        
        percent_bb = pd.Series(percent_bb)
        #print(percent_bb)
        indicator_prices = indicator_prices.iloc[9:]
        #print(indicator_prices)
        momentum = pd.Series(ind.get_momentum(indicator_prices[symbol], window = 10))
        momentum = momentum.iloc[10:]
        #print(momentum)
        volatility = pd.Series(ind.get_volatility(indicator_prices[symbol], window = 10))
        volatility = volatility.iloc[10:]  	   		     		  		  		    	 		 		   		 		  
        # here we build a fake set of trades  		  	   		     		  		  		    	 		 		   		 		  
        # your code should return the same sort of data  		  	   		     		  		  		    	 		 		   		 		  
        dates = pd.date_range(sd, ed)  		  	   		     		  		  		    	 		 		   		 		  
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY  		  	   		     		  		  		    	 		 		   		 		  
        trades = prices_all[[symbol,]]  # only portfolio symbols  		  	   		     		  		  		    	 		 		   		 		  
        trades_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		     		  		  		    	 		 		   		 		  
        trades.values[:, :] = 0  # set them all to nothing  
        vol, momnt, bb = self.discretize(symbol)
        #print(volatility.iloc[0])
        #print(vol)
        vol_bin= ""
        mo_bin = ""
        bb_bin = ""
		  		     		  		  		    	 		 		   		 		  
        holdings = 0
        for x in range(trades.size):
            vol_set = False
            mo_set = False
            bb_set = False
            for y in range(vol.size):
                if(volatility.iloc[x] <= vol[y] and vol_set == False):
                    #print(x)
                    vol_bin = str(y)
                    vol_set=True
                elif(y==vol.size-1 and vol_set == False):
                    vol_bin = str(y+1)
                    vol_set=True
                if(momentum.iloc[x] <= momnt[y] and mo_set == False):
                    mo_bin = str(y)
                    mo_set=True
                elif(y==momnt.size-1 and mo_set == False):
                    mo_bin = str(y+1)
                    mo_set=True
                if(percent_bb.iloc[x] <= bb[y] and bb_set == False):
                    bb_bin = str(y)
                    bb_set = True
                elif(y==bb.size-1 and bb_set == False):
                    bb_bin = str(y+1)  
                    bb_set = True
            state = int(vol_bin+mo_bin+bb_bin)	
            #print(state)
            action = self.learner.querysetstate(state)-1
            #print(action)
            if(action==1):
                if(holdings==0):
                    trades.values[x, :] = 1000
                elif(holdings==-1000):
                    trades.values[x, :] = 2000
                elif(holdings==1000):
                    trades.values[x, :] = 0
                holdings = 1000
            elif(action==0):
                if(holdings==0):
                    trades.values[x, :] = 0
                elif(holdings==-1000):
                    trades.values[x, :] = 1000
                elif(holdings==1000):
                    trades.values[x, :] = -1000
                holdings = 0
            elif(action==-1):
                if(holdings==0):
                    trades.values[x, :] = -1000
                elif(holdings==-1000):
                    trades.values[x, :] = 0
                elif(holdings==1000):
                    trades.values[x, :] = -2000
                holdings = -1000
        #print(trades)
        if self.verbose:  		  	   		     		  		  		    	 		 		   		 		  
            print(type(trades))  # it better be a DataFrame!  		  	   		     		  		  		    	 		 		   		 		  
        if self.verbose:  		  	   		     		  		  		    	 		 		   		 		  
            print(trades)  		  	   		     		  		  		    	 		 		   		 		  
        if self.verbose:  		  	   		     		  		  		    	 		 		   		 		  
            print(prices_all)  	
        #print(trades)	  	   		     		  		  		    	 		 		   		 		  
        return trades	  	   		     	


    def testLearner(  		  	   		   #this is where you could use the inspiration for the grading function from project 7  		  		  		    	 		 		   		 		  
        self,  		  	   		     		  		  		    	 		 		   		 		  
        symbol="JPM",  		  	   		     		  		  		    	 		 		   		 		  
        sd=dt.datetime(2008, 1, 1),  		  	   		     		  		  		    	 		 		   		 		  
        ed=dt.datetime(2009, 12, 31),  		  	   		     		  		  		    	 		 		   		 		  
        sv=10000,  		  	   		     		  		  		    	 		 		   		 		  
    ):  



        """  		  	   		     		  		  		    	 		 		   		 		  
        Trains your strategy learner over a given time frame.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol to train on  		  	   		     		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		     		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		     		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		     		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		     		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		     		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		     		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
        # add your code to do learning here  		  	   		     		  		  		    	 		 		   		 		  
        syms = [symbol]  		  	   		     		  		  		    	 		 		   		 		  
        dates = pd.date_range(sd, ed) 
        indicator_dates = pd.date_range(sd-dt.timedelta(days=28), ed)  		  	   		     		  		  		    	 		 		   		 		  
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY  		  	   		     		  		  		    	 		 		   		 		  
        prices = prices_all[syms]  # only portfolio symbols  	
        test_returns = prices_all[syms]  # only portfolio symbols  		
        indicator_prices_all = ut.get_data(syms, indicator_dates)   #will this be the same size as the number of days traded?
        indicator_prices = indicator_prices_all[syms]		  		  		    	 		 		   		 		  
        prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		     		  		  		    	 		 		   		 		  
        

        #get 3 indicators using passed in start date minus window required for each
        rm = ind.get_rolling_mean(indicator_prices[symbol], window = 20)
        percent_bb = ind.get_rolling_std(indicator_prices[symbol], window = 20) #aka rstd
        upper_band, lower_band = ind.get_bollinger_bands(rm, percent_bb)

        percent_bb = percent_bb.iloc[19:]
        rm = rm.iloc[19:]
        upper_band = upper_band.iloc[19:]
        lower_band = lower_band.iloc[19:]
        for x in range(rm.size):  #did i format that right??
            #print(prices.iloc[x])
            #print(type(rm.iloc[x]))
            if(prices.iloc[x][0] == rm.iloc[x]):
                percent_bb.iloc[x] = 0.0
            elif(prices.iloc[x][0] > rm.iloc[x]):
                percent_bb.iloc[x] = (prices.iloc[x][0] - rm.iloc[x]) / (upper_band.iloc[x] - rm.iloc[x])
            elif(prices.iloc[x][0] < rm.iloc[x]):
                percent_bb.iloc[x] = (rm.iloc[x] - prices.iloc[x][0]) / (lower_band.iloc[x] - rm.iloc[x])
        
        percent_bb = pd.Series(percent_bb)
        #print(percent_bb)
        indicator_prices = indicator_prices.iloc[9:]
        #print(indicator_prices)
        momentum = pd.Series(ind.get_momentum(indicator_prices[symbol], window = 10))
        momentum = momentum.iloc[10:]
        #print(momentum)
        volatility = pd.Series(ind.get_volatility(indicator_prices[symbol], window = 10))
        volatility = volatility.iloc[10:]
        #print(volatility.values.shape)
                                                                                            
        uglystupidnobodylikesyouvol, vol = pd.qcut(volatility.values, 10, retbins=True)
        esrfergs, momnt = pd.qcut(momentum.values, 10, retbins=True)
        rgefsegrf, bb = pd.qcut(percent_bb.values, 10, retbins=True)
        vol = vol[1:-1]
        momnt = momnt[1:-1]
        bb = bb[1:-1]
        #print(volatility.iloc[0])
        #print(vol)
        #print(volatility.iloc[0] in vol[0])   		



        total_reward = 0
        #performance_list = np.zeros(5)
        for z in range(1):
            cash_value = float(sv)
            stock_value =0.0
            portfolio_value = cash_value + stock_value
            #print(type(portfolio_value))
            test_returns.iloc[0]=portfolio_value
            #figure out state
            vol_set = False
            mo_set = False
            bb_set = False
            vol_bin= ""
            mo_bin = ""
            bb_bin = ""
            
            for x in range(vol.size):
                if(volatility.iloc[0] <= vol[x] and vol_set == False):
                    
                    vol_bin = str(x)
                    vol_set=True

                elif(x==vol.size-1 and vol_set == False):
                    vol_bin = str(x+1)
                    vol_set=True
                if(momentum.iloc[0] <= momnt[x] and mo_set == False):
                    mo_bin = str(x)
                    mo_set=True
                elif(x==momnt.size-1 and mo_set == False):
                    mo_bin = str(x+1)
                    mo_set=True
                if(percent_bb.iloc[0] <= bb[x] and bb_set == False):
                    bb_bin = str(x)
                    bb_set = True
                elif(x==bb.size-1 and bb_set == False):
                    bb_bin = str(x+1)  
                    bb_set = True
                

            #print(volatility.iloc[0])
            #position = 0   actually this could be action variable below
            final_str = vol_bin + mo_bin + bb_bin
            state = int(final_str)
            #print(state)
            action = self.learner.querysetstate(  		  	   		     		  		  		    	 		 		   		 		  
                state  		  	   		     		  		  		    	 		 		   		 		  
            )-1
            old_action = action
            same_action = False
            count=0
            stock_value = action*1000.0*prices.iloc[0]
            cash_value -= stock_value

            #print("Daily Value for end of Day 1" + ": " + str(portfolio_value))
            for x in range(1, prices.size):   #while the trading simualtion hasnt reached last day

                #at this point assume day 1 or x is over and now calculate the reward of the prior day's 
                #decision to pass in next action call as r. also calculate the 
                # holdings of the bot(maybe just check the action variable for this)
                daily_gain = (prices.values[x,0]-prices.values[x-1,0])/prices.values[x-1,0]  #daily percentage gain
                #print(type(daily_gain))
                portfolio_value = ((1.0+daily_gain)*stock_value) + cash_value
                #print(portfolio_value)
                test_returns.iloc[x]=portfolio_value
                if(action == 1):
                    if(daily_gain >0.02):
                        reward = 35.0
                    elif(daily_gain >0.012):
                        reward = 20.0
                    elif(daily_gain>0.004):
                        reward = 7.5
                    elif(daily_gain < -0.019):
                        reward = -25.0
                    elif(daily_gain < -0.009):
                        reward = -10.0
                    else:
                        reward = daily_gain*10.0
                elif(action == 0):
                    reward = abs(daily_gain)*3.5
                elif(action == -1):
                    if(daily_gain <-0.026):
                        reward = 35.0
                    elif(daily_gain <-0.015):
                        reward = 20.0
                    elif(daily_gain <-0.004):
                        reward = 7.5
                    elif(daily_gain > 0.019):
                        reward = -25.0
                    elif(daily_gain > 0.007):
                        reward = -10.0
                    else:
                        reward = daily_gain*-10.0
                
                if(same_action==True):
                    reward+=7.5
                #determine state
                vol_set = False
                mo_set = False
                bb_set = False
                vol_bin= ""
                mo_bin = ""
                bb_bin = ""
                for y in range(vol.size):
                    if(volatility.iloc[x] <= vol[y] and vol_set == False):
                        #print(x)
                        vol_bin = str(y)
                        vol_set=True
                    elif(y==vol.size-1 and vol_set == False):
                        vol_bin = str(y+1)
                        vol_set=True
                    if(momentum.iloc[x] <= momnt[y] and mo_set == False):
                        mo_bin = str(y)
                        mo_set=True
                    elif(y==momnt.size-1 and mo_set == False):
                        mo_bin = str(y+1)
                        mo_set=True
                    if(percent_bb.iloc[x] <= bb[y] and bb_set == False):
                        bb_bin = str(y)
                        bb_set = True
                    elif(y==bb.size-1 and bb_set == False):
                        bb_bin = str(y+1)  
                        bb_set = True 
                state = int(vol_bin+mo_bin+bb_bin)
                old_action = action	
                action = self.learner.query(state, reward) - 1 
                same_action = False

                if(old_action==action and reward > 0.0):
                    same_action = True
                stock_value = action*1000.0*prices.iloc[x]
                cash_value = portfolio_value - stock_value
                total_reward += reward
                
                #print("Daily Value for end of Day " + str(x+1) + ": " + str(portfolio_value))
            #print("Ending value for test phase " + str(z+1) + ": " + str(portfolio_value))
            #performance_list[z] = portfolio_value
        return test_returns	    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		     		  		  		    	 		 		   		 		  
    print("One does not simply think up a strategy")  		  	
    #__init__()  
     		     		  		  		    	 		 		   		 		  
