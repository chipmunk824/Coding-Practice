DNAstring = ""
complement = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}
DNA_list = list(DNAstring)
DNA_list.reverse()
DNA_reverse = [ ]
for i in DNA_list:
    DNA_reverse.append(complement[i])
"".join(DNA_reverse) 
