def main():
	f = open("Bridge_5000000.txt", "r")
	lines = f.readlines()
	f.close()
	idx = 0

	parsed_len = 0

	with open("Bridge_5000000_parsed.txt", 'w') as g:
		g.write(lines[0])
		g.write(lines[1])
		g.write(lines[2])
		idx += 3
		valid_cards_list = [set(list(range(13*0, 13*1))),
							set(list(range(13*1, 13*2))),
							set(list(range(13*2, 13*3))),
							set(list(range(13*3, 13*4)))]

		while idx < len(lines):
			opening_suit = int(lines[idx][12])
			valid_cards = valid_cards_list[opening_suit]
			l_idx = lines[idx].index("{")
			r_idx = lines[idx].index("}")
			cards_played = list(map(int, lines[idx][l_idx+1:r_idx].split(", ")))
			partner_card = int(lines[idx].strip()[-2:])
			if partner_card == -1 and len(cards_played) < 2 or partner_card != -1:
				for card in cards_played:
					if card in valid_cards:
						g.write(lines[idx])
						g.write(lines[idx+1])
						g.write(lines[idx+2])
						parsed_len += 1
						break
			idx += 3

	print(parsed_len // 3)
	print(len(lines) // 3)

if __name__ == "__main__":
	main()
