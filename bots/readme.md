## ############### 2/25/26 - 4h filter for 1h charts

Good, I have the full picture. Now let me make the changes:

Add confirm_interval to BotEntry
Update load_bot_registry to read it
Add a helper to fetch and compute the higher timeframe signal
Modify tick_bot to gate BUY signals through the confirmation filter

## Running on 4h only:
Check signals every 4 hours
Can only enter/exit on 4h candle closes
If a trend starts mid-candle, you wait up to 4 hours to act

## Running on 1h with 4h confirmation:
Check signals every 1 hour (more responsive)
1h says BUY → check if 4h also agrees → if yes, enter NOW
1h says EXIT → exit immediately, no waiting for 4h
You get the speed of 1h exits with the quality filter of 4h entries

The key difference is exits are fast, entries are filtered. That's the advantage.


## ######################### logs
this is to know all bots are running.
Add to your .env:
JUPBOT_TRADE_LOG_MIRROR_DIR=/home/rchak007/github/jobMyTrading/outputs/bot
JUPBOT_HEARTBEAT_MIRROR_DIR=/home/rchak007/github/jobMyTrading/outputs/bot
Overwrites at midnight PST (new day = fresh file). Mirrors to jobMyTrading for GitHub push.


