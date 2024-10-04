import jwt
import uuid
import hashlib
import time
from urllib.parse import urlencode
import requests
import json
import os
import logging
from dotenv import load_dotenv
from typing import Dict, Any, Tuple, Optional, List

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='trading_bot.log', filemode='a')
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'API_KEY': os.getenv('BITHUMB_ACCESS_KEY_2'),
    'SECRET_KEY': os.getenv('BITHUMB_SECRET_KEY_2'),
    'API_URL': 'https://api.bithumb.com',
    'TICKER': 'ETH',
    'ORDER_AMOUNT_KRW': 50000,
    'CHECK_INTERVAL': 5,
    'MAIN_LOOP_INTERVAL': 60,
    'TRADING_CONFIG': {
        'fee': 0.00,
        'profit_ratio': 1.002,
        'loss_ratio': 0.999
    }
}

class BithumbAPI:
    def __init__(self, api_key: str, secret_key: str, api_url: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.api_url = api_url

    def create_token(self, query_params: Dict[str, Any] = None) -> str:
        payload = {
            'access_key': self.api_key,
            'nonce': str(uuid.uuid4()),
            'timestamp': int(time.time() * 1000)
        }
        if query_params:
            query = urlencode(query_params).encode()
            m = hashlib.sha512()
            m.update(query)
            query_hash = m.hexdigest()
            payload['query_hash'] = query_hash
            payload['query_hash_alg'] = 'SHA512'

        jwt_token = jwt.encode(payload, self.secret_key)
        return f'Bearer {jwt_token}'

    def api_call(self, method: str, endpoint: str, params: Dict[str, Any] = None, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        url = self.api_url + endpoint
        headers = {
            'Authorization': self.create_token(params),
            'Content-Type': 'application/json'
        }

        for attempt in range(max_retries):
            try:
                if method == 'GET':
                    response = requests.get(url, headers=headers, params=params)
                elif method == 'POST':
                    response = requests.post(url, headers=headers, data=json.dumps(params))
                elif method == 'DELETE':
                    response = requests.delete(url, headers=headers, params=params)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.error(f"API call error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff

    def get_balance(self, currency: str = 'KRW') -> Optional[Dict[str, float]]:
        response = self.api_call('GET', '/v1/accounts')
        if response and isinstance(response, list):
            for item in response:
                if item.get('currency') == currency:
                    return {
                        'balance': float(item.get('balance', '0')),
                        'locked': float(item.get('locked', '0')),
                        'avg_buy_price': float(item.get('avg_buy_price', '0')),
                        'avg_buy_price_modified': item.get('avg_buy_price_modified', False),
                        'unit_currency': item.get('unit_currency', '')
                    }
            logger.warning(f"Currency {currency} not found in balance")
        else:
            logger.error(f"Unexpected response format: {response}")
        return None

    def place_order(self, ticker: str, side: str, price: float, quantity: float, ord_type: str = 'limit') -> Optional[Dict[str, Any]]:
        params = {
            "market": f"KRW-{ticker}",
            "side": side,
            "volume": str(quantity),
            "price": str(price),
            "ord_type": ord_type
        }
        response = self.api_call('POST', '/v1/orders', params)
        if response and isinstance(response, dict):
            if 'uuid' in response:
                return {
                    'uuid': response['uuid'],
                    'side': response.get('side'),
                    'ord_type': response.get('ord_type'),
                    'price': response.get('price'),
                    'state': response.get('state'),
                    'market': response.get('market'),
                    'created_at': response.get('created_at'),
                    'volume': response.get('volume'),
                    'remaining_volume': response.get('remaining_volume'),
                    'reserved_fee': response.get('reserved_fee'),
                    'remaining_fee': response.get('remaining_fee'),
                    'paid_fee': response.get('paid_fee'),
                    'locked': response.get('locked'),
                    'executed_volume': response.get('executed_volume'),
                    'trades_count': response.get('trades_count')
                }
            else:
                logger.error(f"Order placement failed: {response}")
        else:
            logger.error(f"Unexpected response format: {response}")
        return None

    def get_orderbook(self, ticker: str) -> Optional[Dict[str, Any]]:
        endpoint = "/v1/orderbook"
        params = {"markets": f"KRW-{ticker}"}
        response = self.api_call('GET', endpoint, params)
        if response and isinstance(response, list) and len(response) > 0:
            orderbook_data = response[0]
            return {
                'market': orderbook_data.get('market'),
                'timestamp': int(orderbook_data.get('timestamp', 0)),
                'total_ask_size': float(orderbook_data.get('total_ask_size', 0)),
                'total_bid_size': float(orderbook_data.get('total_bid_size', 0)),
                'orderbook_units': [
                    {
                        'ask_price': float(unit['ask_price']),
                        'bid_price': float(unit['bid_price']),
                        'ask_size': float(unit['ask_size']),
                        'bid_size': float(unit['bid_size'])
                    }
                    for unit in orderbook_data.get('orderbook_units', [])
                ]
            }
        else:
            logger.error(f"Unexpected response format: {response}")
            return None

    def get_order_status(self, order_uuid: str) -> Optional[Dict[str, Any]]:
        endpoint = "/v1/order"
        params = {"uuid": order_uuid}
        
        response = self.api_call('GET', endpoint, params)
        if response and isinstance(response, dict):
            return {
                'uuid': response.get('uuid'),
                'side': response.get('side'),
                'ord_type': response.get('ord_type'),
                'price': float(response.get('price', '0')),
                'state': response.get('state'),
                'market': response.get('market'),
                'created_at': response.get('created_at'),
                'volume': float(response.get('volume', '0')),
                'remaining_volume': float(response.get('remaining_volume', '0')),
                'reserved_fee': float(response.get('reserved_fee', '0')),
                'remaining_fee': float(response.get('remaining_fee', '0')),
                'paid_fee': float(response.get('paid_fee', '0')),
                'locked': float(response.get('locked', '0')),
                'executed_volume': float(response.get('executed_volume', '0')),
                'trades_count': int(response.get('trades_count', '0'))
            }
        else:
            logger.error(f"Unexpected response format: {response}")
            return None

    def cancel_order(self, order_uuid: str) -> bool:
        params = {"uuid": order_uuid}
        response = self.api_call('DELETE', '/v1/order', params)
        if response and isinstance(response, dict):
            if response.get('status') == '0000':  # 성공 상태 코드 확인
                logger.info(f"Successfully canceled order: {order_uuid}")
                return True
            else:
                logger.error(f"Failed to cancel order: {response}")
                return False
        else:
            logger.error(f"Unexpected response format: {response}")
            return False
        
    def get_candlestick(self, ticker: str, interval: str = "10m", count: int = 200) -> Optional[List[Dict[str, Any]]]:
        interval_map = {"1m": 1, "3m": 3, "5m": 5, "10m": 10, "15m": 15, "30m": 30, "60m": 60, "240m": 240}
        minutes = interval_map.get(interval, 10)

        endpoint = f"/v1/candles/minutes/{minutes}"
        params = {
            "market": f"KRW-{ticker}",
            "count": count
        }
        
        response = self.api_call('GET', endpoint, params)
        if response and isinstance(response, list):
            return [
                {
                    'market': candle['market'],
                    'candle_date_time_utc': candle['candle_date_time_utc'],
                    'candle_date_time_kst': candle['candle_date_time_kst'],
                    'opening_price': float(candle['opening_price']),
                    'high_price': float(candle['high_price']),
                    'low_price': float(candle['low_price']),
                    'trade_price': float(candle['trade_price']),
                    'timestamp': int(candle['timestamp']),
                    'candle_acc_trade_price': float(candle['candle_acc_trade_price']),
                    'candle_acc_trade_volume': float(candle['candle_acc_trade_volume']),
                    'unit': int(candle['unit'])
                }
                for candle in response
            ]
        else:
            logger.error(f"Unexpected response format: {response}")
            return None    
    

    
class BithumbTrader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api = BithumbAPI(config['API_KEY'], config['SECRET_KEY'], config['API_URL'])
        self.trading_config = config['TRADING_CONFIG']

    def should_buy(self, ticker: str) -> bool:
        candles = self.api.get_candlestick(ticker, interval="10m", count=20)
        if not candles:
            return False
        
        closes = [candle['trade_price'] for candle in candles]
        ma5 = sum(closes[-5:]) / 5
        ma20 = sum(closes) / 20
        
        if ma5 > ma20 and sum(closes[-6:-1]) / 5 <= ma20:
            return True
        #return False
        return True # Temp, 항상 True 로 
    
    def buy_coin(self, ticker: str) -> Tuple[Optional[float], float]:
        balance = self.api.get_balance('KRW')
        if not balance:
            logger.error("Failed to retrieve KRW balance information")
            return None, 0
        
        available_krw = balance['balance']
        
        if available_krw < self.config['ORDER_AMOUNT_KRW']:
            logger.warning(f"Insufficient balance. Available: {available_krw} KRW")
            return None, 0

        orderbook = self.api.get_orderbook(ticker)
        if not orderbook or not orderbook['orderbook_units']:
            logger.error("Failed to retrieve orderbook or no orderbook units available")
            return None, 0

        best_ask_price = orderbook['orderbook_units'][0]['ask_price']

        limit_price = best_ask_price
        units = round(self.config['ORDER_AMOUNT_KRW'] / limit_price * (1.0 - self.trading_config['fee']), 4)
        
        order = self.api.place_order(ticker, "bid", limit_price, units)
        if not order:
            logger.error("Failed to place limit order")
            return None, 0

        logger.info(f"Placed limit buy order for {ticker}: {order}")
        order_uuid = order['uuid']
        
        start_time = time.time()
        max_wait_time = 120  # 2 minutes
        
        while time.time() - start_time < max_wait_time:
            order_status = self.api.get_order_status(order_uuid)
            if not order_status:
                logger.error("Failed to get order status")
                break
            
            if order_status['state'] == 'done':
                logger.info(f"Order fully executed for {ticker}")
                return limit_price, float(order_status['executed_volume'])
            
            executed_volume = float(order_status['executed_volume'])
            if executed_volume > 0:
                logger.info(f"Order partially executed. Executed volume: {executed_volume}")
            
            time.sleep(self.config['CHECK_INTERVAL'])
        
        # After 2 minutes
        final_order_status = self.api.get_order_status(order_uuid)
        if not final_order_status:
            logger.error("Failed to get final order status")
            self.api.cancel_order(order_uuid)
            return None, 0
        
        executed_volume = float(final_order_status['executed_volume'])
        remaining_volume = float(final_order_status['remaining_volume'])
        
        if executed_volume == units:
            # 1. 모두 전체 체결됐을 경우
            logger.info(f"Order fully executed. Executed volume: {executed_volume}")
            return limit_price, executed_volume
        
        # Cancel the remaining order only if it's not fully executed
        if remaining_volume > 0:
            try:
                cancel_result = self.api.cancel_order(order_uuid)
                if not cancel_result:
                    logger.warning(f"Failed to cancel order: {order_uuid}")
            except Exception as e:
                logger.error(f"Error while cancelling order: {e}")
        
        if executed_volume > 0:
            # 2. 일부 체결된 경우
            logger.info(f"Partial execution. Executed volume: {executed_volume}")
            remaining_amount = self.config['ORDER_AMOUNT_KRW'] - (executed_volume * limit_price)
        else:
            # 3. 모두 체결 안된 경우
            logger.info("Order not executed. Placing full market order.")
            remaining_amount = self.config['ORDER_AMOUNT_KRW']
        
        # Place market order for remaining amount
        try:
            current_price = self.api.get_orderbook(ticker)['orderbook_units'][0]['ask_price']
            market_units = remaining_amount / current_price * (1.0 - self.trading_config['fee'])
            
            market_order = self.api.place_order(ticker, "bid", current_price, market_units, ord_type='market')
            if market_order:
                logger.info(f"Placed market buy order for remaining amount: {market_order}")
                market_executed_volume = float(market_order['executed_volume'])
                total_executed_volume = executed_volume + market_executed_volume
                average_price = ((executed_volume * limit_price) + (market_executed_volume * current_price)) / total_executed_volume
                return average_price, total_executed_volume
            else:
                logger.error("Failed to place market order for remaining amount")
        except Exception as e:
            logger.error(f"Error while placing market order: {e}")
        
        # If market order failed or caused an exception
        if executed_volume > 0:
            return limit_price, executed_volume  # Return partial execution result
        else:
            return None, 0  # Return failure if no execution at all

    def sell_coin(self, ticker: str, buy_price: float, units: float) -> bool:
        orderbook = self.api.get_orderbook(ticker)
        if not orderbook or not orderbook['orderbook_units']:
            logger.error("Failed to retrieve orderbook or no orderbook units available")
            return False

        current_price = orderbook['orderbook_units'][0]['bid_price']
        if current_price >= buy_price * self.trading_config['profit_ratio']:
            order = self.api.place_order(ticker, "ask", current_price, units)
            if order:
                logger.info(f"Selling {ticker} at {self.trading_config['profit_ratio']*100-100}% gain: {order}")
                return True
        elif current_price <= buy_price * self.trading_config['loss_ratio']:
            order = self.api.place_order(ticker, "ask", current_price, units)
            if order:
                logger.info(f"Selling {ticker} at {100-self.trading_config['loss_ratio']*100}% loss: {order}")
                return True
        return False

    def run(self):
        ticker = self.config['TICKER']
        
        while True:
            logger.info(f"Checking {ticker} for trading opportunity...")
            
            if self.should_buy(ticker):
                buy_price, units = self.buy_coin(ticker)
                if buy_price is not None and units > 0:
                    while True:
                        if self.sell_coin(ticker, buy_price, units):
                            break
                        time.sleep(self.config['CHECK_INTERVAL'])
            
            time.sleep(self.config['MAIN_LOOP_INTERVAL'])

if __name__ == "__main__":
    trader = BithumbTrader(CONFIG)
    trader.run()