from collections import defaultdict
def pick_card(player, b):
	mp = {
		'A': 1,
		'2': 2,
		'3': 3,
		'4': 4,
		'5': 5,
		'6': 6,
		'7': 7,
		'8': 8,
		'9': 9,
		'D': 10,
		'Q': 11,
		'J': 12,
		'K': 13,
		'C': 100
	}
	
	if 'C' in player and not b:
		return 'C'

	copy = [x for x in player if x != 'C']
	
	counter = defaultdict(int)
	for c in copy:
		counter[c] += 1

	inv = {}
	for k, v in counter.items():
		curr = inv.get(v)
		if curr is not None and mp[curr] < mp[k]:
			continue
		inv[v] = k

	if inv.get(1):
		return inv.get(1)
	elif inv.get(2):
		return inv[2]
	elif inv.get(3):
		return inv[3]
	elif inv.get(4):
		return inv[4]

n, k = [int(x) for x in input().split()]
k = k-1
players = []
for i in range(n):
	players.append(list(input()))

cp = k
blocked = False
players[k].append('C')
while True:
	next = (cp + 1) % n
	if 'C' in players[cp]:
		blocked = False if blocked else True
	to_pass = pick_card(players[cp], blocked)
	players[cp].remove(to_pass)
	players[next].append(to_pass) 
	if players[cp][0] == players[cp][1] and players[cp][1] == players[cp][2] and players[cp][2] == players[cp][3]:
		break 
	cp = next

print(cp+1)