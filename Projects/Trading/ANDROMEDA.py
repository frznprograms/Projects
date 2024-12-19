from AlgorithmImports import *

class SecurityInfo:
    def __init__(self, symbol):
        self.Symbol = symbol
        self.EntryTicket = None
        self.EntryPrice = 0
        self.MaxPrice = 0
        self.StopPrice = 0


class SelectionData(object):
    def __init__(self, symbol, period):
        self.symbol = symbol
        self.ema = ExponentialMovingAverage(self.symbol, period)
        self.is_above_ema = False
        self.volume = 0

    def update(self, time, price, volume):
        self.volume = volume
        self.ema.Update(time, price)
            # use Indicator primitive update method to update data for indicator 
        if price > self.ema.Current.Value:
            self.is_above_ema = True


class Adam(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2022, 2, 10)
        self.SetEndDate(2023, 1, 1)
        self.SetCash(3000)

        # Create a new Dynamic universe
        self.AddUniverse(self.CoarseFilterFunction, self.FineSelectionFunction)
        self.UniverseSettings.Resolution = Resolution.Hour
        
        self.security_infos = {}  # Dictionary to hold SecurityInfo object
        self.stateData = {} # track state of indicators 
        self.securities = [] # list of active securities in the universe
        
        self.Settings.FreePortfolioValuePercentage = 0.05


    def CoarseFilterFunction(self, coarse):
        for c in coarse:
            if c.Symbol not in self.stateData:
                # if not in dictionary, create new object and add it in 
                self.stateData[c.Symbol] = SelectionData(c.Symbol, 14)
            avg = self.stateData[c.Symbol]
            # update the moving average data 
            avg.update(c.EndTime, c.AdjustedPrice, c.DollarVolume)

        values = [x for x in self.stateData.values() if x.is_above_ema and x.volume > 100000000]
        values_sorted = sorted(values, key=lambda x: x.volume, reverse=True)
        return [x.symbol for x in values_sorted[:10]]

    def FineSelectionFunction(self, fine):
        sortedByPeRatio = sorted(fine, key=lambda x: x.ValuationRatios.PERatio, reverse=False)
        return [x.Symbol for x in sortedByPeRatio[:10]]

    def OnSecuritiesChanged(self, changes):
        for security in changes.AddedSecurities: 
            security.indicator = self.SMA(security.Symbol, 7)
            self.WarmUpIndicator(security.Symbol, security.indicator)
        self.securities.extend(changes.AddedSecurities)

        for security in list(self.securities):  # Create a copy of the list before iterating
            if security in changes.RemovedSecurities:
                self.DeregisterIndicator(security.indicator)
                self.securities.remove(security)

        # Get the currently active securities in the universe
        active_securities = self.UniverseManager.ActiveSecurities
        for security_active in active_securities:
            security = security_active.Key # it is in key: value pairs of symbol: security
            equity_symbol = self.AddEquity(security, Resolution.Hour).Symbol
            if equity_symbol not in self.security_infos:
                self.security_infos[equity_symbol] = SecurityInfo(equity_symbol)



    def OnData(self, data: Slice):
        for security_info in self.security_infos.values():
            if not security_info.EntryTicket: 
                # set the information for each security invested in 
                security = self.Securities[security_info.Symbol]  # Access the security object
                quantity = self.CalculateOrderQuantity(security.Symbol, 0.1)
                security_info.EntryPrice = security.Price
                security_info.EntryTicket = self.MarketOrder(security.Symbol, quantity)
                security_info.MaxPrice = security_info.EntryPrice * 1.5
                security_info.StopPrice = security_info.EntryPrice * 0.95



    def OnOrderEvent(self, orderEvent): 
        if orderEvent.Status != OrderStatus.Filled: 
            return 
        # once the event is triggered 
        # order events do not trigger trades by default, do them on your own 
        else:
            for security_info in self.security_infos.values():
                if security_info.EntryTicket and security_info.EntryTicket.OrderId == orderEvent.OrderId:
                    try: 
                        close_price = self.Securities[security_info.Symbol].Price
                        if close_price >= security_info.MaxPrice:
                            self.Liquidate(security_info.Symbol)
                        elif close_price > security_info.EntryPrice and close_price < security_info.MaxPrice:
                            new_stop_price = close_price * 0.95
                            if new_stop_price > security_info.StopPrice:
                                update_fields = UpdateOrderFields()
                                update_fields.StopPrice = new_stop_price
                                self.Transactions.UpdateOrderFields(security_info.Symbol, security_info.EntryTicket.OrderId, update_fields)
                        elif close_price < security_info.StopPrice: 
                            self.Liquidate(security_info.Symbol)
                    except: 
                        self.Debug('There is an Error. Check again.')
