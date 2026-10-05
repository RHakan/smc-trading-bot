import ccxt, pandas as pd
from bot.strategies.breakout_short import analyze

ex = ccxt.binanceusdm({'enableRateLimit': True})

raw_1h = ex.fetch_ohlcv('BTC/USDT:USDT', '1h', limit=200)
raw_1d = ex.fetch_ohlcv('BTC/USDT:USDT', '1d', limit=60)

def to_df(raw):
    df = pd.DataFrame(raw, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    return df.set_index('ts')

df_1h = to_df(raw_1h).iloc[:-1]
df_1d = to_df(raw_1d)

result = analyze(df_1h, df_1d)

print('=== BTC/USDT:USDT - Breakout SHORT ===')
print('Sinal:', result['signal'])
print('Entry:', result['entry'])
print('SL:   ', result['stop_loss'])
print('TP:   ', result['take_profit'])
print('R:R:  ', round(result['rr'], 2))
print('ATR:  ', round(result['atr'], 4))
print()
print('Condicoes:')
for c in result['conditions']:
    mark = '[OK]' if c['passed'] else '[--]'
    print(' ', mark, c['name'])
    print('       ', c['detail'])
