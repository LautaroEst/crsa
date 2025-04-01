

meanings_A = ["AAA", "AAB", "ABA", "ABB", "BAA", "BAB", "BBA", "BBB"]
meanings_B = ["111", "112", "121", "122", "211", "212", "221", "222"]
utterances_A = ["1st", "2nd", "3rd"]
utterances_B = ["1st", "2nd", "3rd"]

print("Lexicon A:")
lexicon_A = []
for a1 in utterances_A:
    for b1 in utterances_B:
        for a2 in utterances_A:
            dialogue = [0 for _ in range(len(meanings_A))]
            dialogue[-1] = 1
            for m in meanings_A:
                if m[utterances_A.index(a1)] == "A" and m[utterances_A.index(a2)] == "A":
                    dialogue[meanings_A.index(m)] = 1
            lexicon_A.append((f"{a1} {b1} {a2}", dialogue))
            print(f"{a1} {b1} {a2}: {dialogue},")
print()
print("Lexicon B:")
lexicon_B = []
for a1 in utterances_A:
    for b1 in utterances_B:
        for a2 in utterances_A:
            dialogue = [0 for _ in range(len(meanings_B))]
            dialogue[-1] = 1
            for m in meanings_B:
                if m[utterances_B.index(b1)] == "1":
                    dialogue[meanings_B.index(m)] = 1
            lexicon_A.append((f"{a1} {b1} {a2}", dialogue))
            print(f"{a1} {b1} {a2}: {dialogue},")
                