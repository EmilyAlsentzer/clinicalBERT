import re
import sys
import xml.etree.ElementTree as ET


i = 0
with open(sys.argv[1]) as f:
    for line in f.readlines():
        # filter out non-lines
        if line.startswith( '<RECORD') or line.startswith( '<TEXT') or line.startswith( '<ROOT>') or \
           line.startswith('</RECORD') or line.startswith('</TEXT') or line.startswith('</ROOT>'):
            continue

        # Parse PHI type/tokens
        regex = '(<PHI TYPE="(\w+)">(.*?)</PHI>)'
        phi_tags = re.findall(regex, line)
        for tag in phi_tags:
            line = line.replace(tag[0], '__phi__').strip()

        # Walk through sentence
        phi_ind = 0
        for w in line.split():
            if w == '__phi__':
                phi = phi_tags[phi_ind]
                tag = phi[1]
                toks = phi[2].split()
                print(toks[0], 'B-%s'%tag)
                for t in toks[1:]:
                    print(t, 'I-%s'%tag)
                phi_ind += 1
            # Two elif statements check for edge cases with Dates
            elif w.startswith('__phi__'):
                # examples like following format:
                # <PHI TYPE="DATE">01/01</PHI>/1995 or <PHI TYPE="DATE">01-01</PHI>-95
                phi = phi_tags[phi_ind]
                tag = phi[1]
                toks = phi[2].split()
                print(toks[0], 'B-%s'%tag)
                if w[7:8] == '/' or w[7:8] == '-':
                    print(w[8:], 'O') # remove the / or - in the year
                else:
                    print(w[7:], 'O')
                phi_ind += 1
            elif w.endswith('__phi__'):
                # 1995<PHI TYPE="DATE">0101</PHI>
                phi = phi_tags[phi_ind]
                tag = phi[1]
                toks = phi[2].split()
                print(w[:-7], 'O')
                print(toks[0], 'B-%s'%tag)
                phi_ind += 1
            else:
                print(w, 'O')
        print()
        i+=1


