"""
Trim the full FastChem input tables (_logK.dat and _logK_condensates.dat) down to
a user-defined subset of species, writing the reduced tables to logK_new.dat and
logK_condensates_new.dat respectively.

Usage
-----
Edit the ``elements``, ``molecules``, and ``condensates`` lists below to contain
only the species you want to keep, then run this script once.  The resulting
output files can be used as drop-in replacements for the FastChem input tables.

Notes
-----
In practice, using the full _logK.dat table does not noticeably slow down the
equilibrium-chemistry calculation.  The condensates table, however, can
significantly increase the runtime, so trimming it is worthwhile if condensates
are not needed.  The current retrieval code does not make use of condensates,
so the full _logK.dat is passed directly and only the condensates table is
reduced here for reference.

Authors
-------
Chenyang Ji (2026-04-08)
"""

import numpy as np

elements = ['C', 'O', 'He', 'H', 'K', 'Ca', 'Na', 'Mg', 'Si', 'Fe', 'S', 'Ti', 'Al', 'N', 'V', 'Cr', 'Rb', 'e-']  # include ion species in logK.dat

molecules = ['H2', 'H2O1', 'C1O1', 'C1O2', 'C1H4', 'H3N1', 'N2', 'H2S1', 'Cl1H1', 'Cl1Na1', 
            'Cl1K1', 'H3P1', 'P2', 'H2P1', 'C1H3', 'H4Si1', 'O1P1', 'O1V1', 'O1Ti1']  # VO and TiO added

condensates = ['H2O(s,l)', 'Na2S(s,l)', 'KCl(s,l)', 'NH4Cl(s)', 'SiO(s)', 'Mg2SiO4(s,l)',
            'MgSiO3(s,l)', 'SiO2(s,l)', 'Fe(s,l)', 'H3PO4(s,l)', 'VO(s,l)', 'TiO(s,l)']  # VO and TiO added

# ========== for condensates ==========
with open('_logK_condensates.dat', 'r') as ipt:
    with open('logK_condensates_new.dat', 'w') as opt:
        while(True):
            line = ipt.readline()
            if not line:
                break
            
            # preserve lines that are comments or empty
            if line.startswith('#'):
                opt.write(line)
                continue
            if line == '\n':
                continue

            phrases = line.strip().split(' ')

            # if the gas molecule species is not what we want to consider, ignore it
            writeflag = True
            if phrases[0] not in condensates:
                writeflag = False

            # for condensates, should write until the '\n' line
            if writeflag:
                while(line and line != '\n'):
                    opt.write(line)
                    line = ipt.readline()
                opt.write('\n')

# ========== for elements ==========
with open('_logK.dat', 'r') as ipt:
    with open('logK_new.dat', 'w') as opt:
        while(True):
            line = ipt.readline()
            if not line:
                break

            # preserve lines that are comments or empty
            if line.startswith('#'):
                opt.write(line)
                continue
            if line == '\n':
                continue

            phrases = line.strip().split(' ')

            idxstart = phrases.index(':') + 1
            idxend = phrases.index('#')  # index of characters between ':' and '#'
            writeflag = True
            for i in range(idxstart, idxend, 2):
                if phrases[i] not in elements:
                    writeflag = False
                    line = ipt.readline()
                    break

            if writeflag:
                opt.write(line)
                line = ipt.readline()
                opt.write(line)
                opt.write('\n')

# ========== for molecules ==========
with open('_logK.dat', 'r') as ipt:
    with open('logK_new.dat', 'w') as opt:
        while(True):
            line = ipt.readline()
            if not line:
                break

            # preserve lines that are comments or empty
            if line.startswith('#'):
                opt.write(line)
                continue
            if line == '\n':
                continue

            phrases = line.strip().split(' ')

            idxstart = phrases.index(':') + 1
            idxend = phrases.index('#')  # index of characters between ':' and '#'
            writeflag = True
            for i in range(idxstart, idxend, 2):
                if phrases[i] not in molecules:
                    writeflag = False
                    line = ipt.readline()
                    break

            if writeflag:
                opt.write(line)
                line = ipt.readline()
                opt.write(line)
                opt.write('\n')

