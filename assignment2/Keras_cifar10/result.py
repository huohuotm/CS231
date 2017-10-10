
with open("res_drop_out.txt", "w") as w:
	with open("drop_out.txt", "r") as f:
		for line in f: 
			if line[0] in ['(','*','a']:
				w.write(line)
			if line[0] == 'E':
				w.write(last)
			last = line






 




