'''
2022/1/10
Jinghao Feng
this script is used for parsing ../wiki-RfA.txt into csv file
'''
import csv
import collections

text = open("../data/wiki-RfA.txt",'r', encoding='utf-8' )
csv_ = open("../data/parsed.csv",'w', encoding='utf-8')
csv_w = csv.writer(csv_)

name_dict = collections.defaultdict(dict)
line = text.readline()
while line:
    row = []
    if line[0:3]=='SRC':
        src = line[ 4 : len(line) - 1 ]
        if src not in name_dict.keys():
            name_dict[src] = len(name_dict)
        row.append( name_dict[src] )

        line = text.readline()
        tgt = line[4:len(line)-1]
        if tgt not in name_dict.keys():
            name_dict[tgt] = len(name_dict)
        row.append( name_dict[tgt] )

        line = text.readline()
        row.append(line[4:len(line)-1])

        csv_w.writerow(row)
    line = text.readline()

print(len(name_dict))
text.close()
csv_.close()
