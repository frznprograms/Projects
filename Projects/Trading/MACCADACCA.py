from AlgorithmImports import *
from collections import deque

class MaccaDacca(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2021, 1, 1)
        self.SetEndDate(2023, 11, 8)
        self.SetCash(3000)

        self.symbol = self.AddEquity('AAPL', Resolution.Hour).Symbol
        self.window = deque(maxlen=3)  # Use deque as a rolling window
        self.macd = self.MACD(self.symbol, 1, 2, 2, MovingAverageType.Exponential)
        self.candle_states = []
        self.SetBenchmark(self.symbol)

        self.Settings.FreePortfolioValuePercentage = 0.05


    def OnData(self, data: Slice):
        if data.ContainsKey(self.symbol) and not self.macd.IsReady: 
            self.window.append(data[self.symbol])

        if len(self.window) == 3:  # Check if the rolling window is ready
            recent_candle = self.window[-1]
            middle_candle = self.window[-2]
            oldest_candle = self.window[-3]
            self.CheckCandle(recent_candle, middle_candle, oldest_candle)


        self.macd_value = self.macd.Current.Value
        self.macd_indicator = self.CheckMACD(self.macd_value)

        # if you have yet to invest, open a long position 
        if self.symbol not in self.Portfolio.Keys: 
            quantity = (self.Portfolio.Cash / 2) / data[self.symbol].Price
            price = data[self.symbol].Price

            # long positions
            if self.macd_indicator == 'Bull' and self.candle_states[0] == 'Bullish Engulfing'\
                    or self.macd_indicator == 'Bull' and self.candle_states[0] == 'Bearish Doji'\
                    or self.macd_indicator == 'Bull' and self.candle_states[0] == 'Bullish Momentum'\
                    or self.macd_indicator == 'Bull' and self.candle_states[0] == 'Bullish Hammer':
                self.MarketTicket = self.LimitOrder(self.symbol, quantity, price)
                self.TrailingStop = self.TrailingStopOrder(self.symbol, quantity, price * 0.9, 1.2, False)

            # short positions
            elif self.macd_indicator == 'Bear' and self.candle_states[0] == 'Bearish Engulfing'\
                    or self.macd_indicator == 'Bear' and self.candle_states[0] == 'Bearish Doji'\
                    or self.macd_indicator == 'Bear' and self.candle_states[0] == 'Bearish Momentum'\
                    or self.macd_indicator == 'Bear' and self.candle_states[0] == 'Bearish Hammer':
                self.MarketTicket = self.LimitOrder(self.symbol, -quantity, price)
                self.TrailingStop = self.StopMarketOrder(self.symbol, -quantity, price * 1.1)

        # if you already have an open position, hold until the trailing stop loss or indicator turns to 
        # a large red candle or a doji green candle 

    def CheckBull(self, data): 
        if data.Close > data.Open: 
            return True
        else: 
            return False

    def CheckBear(self, data):
        if data.Close < data.Open: 
            return True
        else: 
            return False


    def RollCandleStates(self, candle_states): 
        if len(candle_states) > 3:
            candle_states.pop()


    def CheckCandle(self, recent_candle, middle_candle, oldest_candle): 
        # Check for engulfing candles  
        if CheckBull(recent_candle) and CheckBear(middle_candle):
            if recent_candle.Close > middle_candle.Open and recent_candle.Open < middle_candle.Close:
                self.candle_states.insert(0, 'Bullish Engulfing')
                RollCandleStates(self.candle_states)

        if CheckBear(recent_candle) and CheckBull(middle_candle):
            if recent_candle.Close < middle_candle.Open and recent_candle.Open > middle_candle.Close: 
                self.candle_states.insert(0, 'Bearish Engulfing')
                RollCandleStates(self.candle_states)

        if CheckBear(recent_candle) and CheckBear(middle_candle):
            if recent_candle.Close < middle_candle.Close and recent_candle.Open > middle_candle.Open: 
                self.candle_states.insert(0, 'Bearish Engulfing')
                RollCandleStates(self.candle_states)

        elif CheckBull(recent_candle) and CheckBull(middle_candle):
            if recent_candle.Close > middle_candle.Close and recent_candle.Open < middle_candle.Open:
                self.candle_states.insert(0, 'Bullish Engulfing')
                RollCandleStates(self.candle_states)

        # Check for doji candles 
        if CheckBull(recent_candle) and recent_candle.Open * 1.1 < recent_candle.Close: 
            self.candle_states.insert(0, 'Bullish Doji')
            RollCandleStates(self.candle_states)

        elif CheckBear(recent_candle) and recent_candle.Close * 1.1 < recent_candle.Open: 
            self.candle_states.insert(0, 'Bearish Doji')
            RollCandleStates(self.candle_states)

        # Check for momentum candles (momentum candles)
        if CheckBull(recent_candle) and 'Engulfing' in self.candle_states[0]:
            size = abs(recent_candle.Close - recent_candle.Open)
            previous_size = abs(middle_candle.Close - middle_candle.Open)
            if size >= previous_size:
                self.candle_states.insert(0, 'Bullish Momentum')
                RollCandleStates(self.candle_states)
                    
        elif CheckBear(recent_candle) and 'Engulfing' in self.candle_states[0]:
            size = abs(recent_candle.Close - recent_candle.Open)
            previous_size = abs(middle_candle.Close - middle_candle.Open)
            if size >= previous_size:
                self.candle_states.insert(0, 'Bearish Momentum')
                RollCandleStates(self.candle_states)     
        
        # check for Maribozu candles 
        if CheckBull(recent_candle) and recent_candle.Open == recent_candle.Low and recent_candle.Close == recent_candle.High:
            self.candle_states.insert(0, 'Bullish Maribozu')
            RollCandleStates(self.candle_states)

        elif CheckBear(recent_candle) and recent_candle.Open == recent_candle.High and recent_candle.Close == recent_candle.Low:
            self.candle_states.insert(0, 'Bearish Maribozu') 
            RollCandleStates(self.candle_states)

        # check for hammer candle (wick is longer than body of candle)
        if CheckBull(recent_candle) and recent_candle.Open - recent_candle.Low > recent_candle.Close - recent_candle.Open: 
            self.candle_states.insert(0, 'Bullish Hammer')
            RollCandleStates(self.candle_states)

        elif CheckBear(recent_candle) and recent_candle.Close - recent_candle.Low > recent_candle.Open - recent_candle.Close: 
            self.candle_states.insert(0, 'Bearish Shooting Star')
            RollCandleStates(self.candle_states)


    def CheckMACD(self, macd_value):
        if macd_value > 0: 
            return 'Bull'
        else:
            return 'Bear'


    def OnOrderEvent(self, orderEvent): 
        if orderEvent.Status != OrderStatus.Filled: 
            return 
        
        elif orderEvent.Status == OrderStatus.Filled and self.Portfolio[self.symbol].IsLong:
            # Check if the order event was for a trailing stop order
            if orderEvent.OrderId == self.TrailingStop.Id:
                self.Liquidate(self.symbol)
        
        elif orderEvent.Status == OrderStatus.Filled and self.Portfolio[self.symbol].IsShort:
            # Check if the order event was for a trailing stop order
            if orderEvent.OrderId == self.TrailingStop.Id:
                self.Liquidate(self.symbol)
