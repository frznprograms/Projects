from AlgorithmImports import *

class SecurityInfo:
    def __init__(self, symbol):
        self.Symbol = symbol
        self.EntryTicket = None
        self.EntryPrice = 0
        self.MaxPrice = 0
        self.StopPrice = 0

class ADAM(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2022, 2, 10)
        self.SetEndDate(2023, 1, 1)
        self.SetCash(3000)
        
        self.equities = ['SPY', 'BND', 'AAPL']
        self.security_infos = {}  # Dictionary to hold SecurityInfo objects
        
        for equity in self.equities: 
            equity_symbol = self.AddEquity(equity, Resolution.Hour).Symbol
            self.security_infos[equity_symbol] = SecurityInfo(equity_symbol)
        
        self.Settings.FreePortfolioValuePercentage = 0.05
        # set minimum free portfolio to be left untouched

    def OnData(self, data: Slice):
        for security_info in self.security_infos.values():
            if not security_info.EntryTicket: 
                quantity = self.CalculateOrderQuantity(security_info.Symbol, 0.3)
                # log entry price
                security_info.EntryPrice = data[security_info.Symbol].Close
                # purchase stocks, strictly long positions 
                security_info.EntryTicket = self.MarketOrder(security_info.Symbol, quantity)
                security_info.MaxPrice = security_info.EntryPrice * 1.5
                security_info.StopPrice = security_info.EntryPrice * 0.95

    def OnOrderEvent(self, orderEvent): 
        if orderEvent.Status != OrderStatus.Filled: 
            return 
        else:
            for security_info in self.security_infos.values():
                if security_info.EntryTicket and security_info.EntryTicket.OrderId == orderEvent.OrderId:
                    try: 
                        close_price = data[security_info.Symbol].Close
                        # once hit intended profit, sell stocks
                        if close_price >= security.MaxPrice:
                            self.Liquidate(security.Symbol)
                        # anywhere between EntryPrice and MaxPrice, custom trailing stop loss 
                        elif close_price > security.EntryPrice and close_price < security.MaxPrice:
                            new_stop_price = close_price * 0.95
                            if new_stop_price > security.StopPrice:
                                update_fields = UpdateOrderFields()
                                update_fields.StopPrice = new_stop_price
                                self.Transactions.UpdateOrderFields(security.Symbol, security.EntryTicket.OrderId, update_fields)
                        # if stock hits initial stop loss, sell immediately 
                        elif close_price < security.StopPrice: 
                            self.Liquidate(security.Symbol)
                    except: 
                        self.Debug('There is an Error. Check again.')
                    
