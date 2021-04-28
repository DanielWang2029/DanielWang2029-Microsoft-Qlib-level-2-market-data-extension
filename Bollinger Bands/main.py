import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import ast
import decimal
import math
import seaborn as sns
from jupyterthemes import jtplot
from datetime import datetime
import re
from dateutil.rrule import rrule, DAILY


df = pd.read_csv("data.csv")
temp = df["0"][df["0"].notnull()].values.tolist()


def parse_string(input_str):

    source = input_str
    tree = ast.parse(source, mode='eval')

    class Transformer(ast.NodeTransformer):
        ALLOWED_NAMES = {'Decimal', 'None', 'False', 'True'}
        ALLOWED_NODE_TYPES = {'Expression', 'Tuple', 'Call', 'Name', 'Load', 'Str', 'Num', 'List', 'Dict', 'Constant'}

        def visit_Name(self, node):
            if not node.id in self.ALLOWED_NAMES:
                raise RuntimeError("Name access to %s is not allowed" % node.id)
            return self.generic_visit(node)

        def generic_visit(self, node):
            nodetype = type(node).__name__
            if nodetype not in self.ALLOWED_NODE_TYPES:
                raise RuntimeError("Invalid expression: %s not allowed" % nodetype)
            return ast.NodeTransformer.generic_visit(self, node)

    transformer = Transformer()
    transformer.visit(tree)
    clause = compile(tree, '<AST>', 'eval')
    result = eval(clause, dict(Decimal=decimal.Decimal))

    return result


bids, asks = [], []
data = [parse_string(x) for x in temp]
for x in data:
    if x['data']['bids']:
        bids.append(x['data']['bids'][0]['price'])
    if x['data']['asks']:
        asks.append(x['data']['asks'][0]['price'])

last_20 = bids[0:20]
pbids = [None] * (len(bids) - 19)
ubids = [None] * (len(bids) - 19)
lbids = [None] * (len(bids) - 19)
pbids[0] = sum(last_20) / 20
ubids[0] = pbids[0] + 2 * math.sqrt(sum([x * x for x in [y - pbids[0] for y in last_20]]) / 20)
lbids[0] = pbids[0] - 2 * math.sqrt(sum([x * x for x in [y - pbids[0] for y in last_20]]) / 20)
for i in range(0, len(bids) - 20):
    last_20.pop(0)
    last_20.append(bids[i + 20])
    pbids[i + 1] = sum(last_20) / 20
    ubids[i + 1] = pbids[i + 1] + 2 * math.sqrt(sum([x * x for x in [y - pbids[i + 1] for y in last_20]]) / 20)
    lbids[i + 1] = pbids[i + 1] - 2 * math.sqrt(sum([x * x for x in [y - pbids[i + 1] for y in last_20]]) / 20)

# print(len(bids))
# print(len(pbids))
date_long = list(rrule(DAILY, dtstart=datetime(2020, 4, 22), until=datetime(2021,1,27)))
date_short = list(rrule(DAILY, dtstart=datetime(2020, 5, 11), until=datetime(2021,1,27)))

# df = pd.read_csv("data2.csv")
# epd = df["Expiration"].values.tolist()
# bid = df["Bid"].values.tolist()
# ask = df["Ask"].values.tolist()
# print(epd[0])
#
# assert len(epd) == len(bid)
# assert len(epd) == len(ask)
#
# current_epd = epd[0]
# temp_ask = []
# temp_bid = []
# final_epd= []
# final_ask = []
# final_bid = []
# for i in range(len(epd)):
#     if current_epd != epd[i]:
#         if len(temp_bid) != 0 and len(temp_ask) != 0:
#             final_bid.append(sum(temp_bid) / len(temp_bid))
#             final_ask.append(sum(temp_ask) / len(temp_ask))
#             final_epd.append(current_epd)
#         temp_bid = []
#         temp_ask = []
#         current_epd = epd[i]
#     if bid[i] != 0:
#         temp_bid.append(bid[i])
#     if ask[i] != 0:
#         temp_ask.append(ask[i])
#
# final_bid.append(sum(temp_bid) / len(temp_bid))
# final_ask.append(sum(temp_ask) / len(temp_ask))
# final_epd.append(current_epd)
#
# final_date = [datetime.date(int(re.split('/', x)[2]), int(re.split('/', x)[0]), int(re.split('/', x)[1])) for x in final_epd]
#
# assert len(final_epd) == len(final_bid)
# assert len(final_epd) == len(final_ask)

jtplot.style(theme='onedork', context='talk', fscale=1.6)
fig = plt.figure(figsize=(20,10), constrained_layout=True)
# plt.rcParams['figure.constrained_layout.use'] = True
ax = fig.add_subplot(111)
ax.plot(date_long, bids, color='royalblue', lw=2)
ax.plot(date_short, pbids, color='orange', lw=2)
ax.plot(date_short, ubids, color='green', lw=2)
ax.plot(date_short, lbids, color='red', lw=2)
ax.set_title('Bollinger Bands', fontsize=20)
plt.ylabel("Price (USD)", fontsize=15)
plt.xlabel("Date", fontsize=15)
plt.legend(["Adj Close", "20 Day Mean Average", "Upper Bond", "Lower Bond"], loc="lower left")
plt.savefig('fig1.png', bbox_inches='tight')
plt.show()

# fig, ax = plt.subplots()
# ax.plot(np.array(pbids))
# ax.plot(np.array(ubids))
# ax.plot(np.array(lbids))
# plt.show()

# fig=plt.figure(figsize=(15,10), constrained_layout=True)
# plt.rcParams['figure.constrained_layout.use'] = True
# ax = fig.add_subplot()
# ax=sns.lineplot(x='date',y='close',data=pbids,ax=ax)

