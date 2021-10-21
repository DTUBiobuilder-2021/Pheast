#!/usr/bin/env python3

import sys

def get_stoichiometry(seq, sequence_type, protein_name):

    '''
    codon_optimization ={
        'A': 'GCU', 'C': 'UGU', 'D': 'GAC', 'E': 'GAG', 'F': 'UUC',
        'G': 'GGU', 'H': 'CAC', 'I': 'AUU', 'K': 'AAG', 'L': 'UUG',
        'M': 'AUG', 'N': 'AAC', 'P': 'CCA', 'Q': 'CAA', 'R': 'AGA',
        'S': 'UCU', 'T': 'ACU', 'V': 'GUU', 'W': 'UGG', 'Y': 'UAC'
        }

    RNA_seq = ''
    for aa in seq:
        RNA_seq += codon_optimization[aa]
    #add the stop codon (UAA is the most likely for P.pastoris with 75%)
    RNA_seq += 'UAA'


    reverse_transcription = str.maketrans('U', 'T')
    DNA_seq = RNA_seq.translate(reverse_transcription)
    '''

    sbml_aa= {
        'A': 'm153', 'R': 'm158', 'N': 'm161', 'D': 'm156',
        'C': 'm331', 'E': 'm154', 'Q': 'm163', 'G': 'm210',
        'H': 'm490', 'I': 'm239', 'L': 'm227', 'K': 'm203',
        'M': 'm343', 'F': 'm272', 'P': 'm185', 'S': 'm279',
        'T': 'm305', 'W': 'm280', 'Y': 'm274', 'V': 'm222'
        }
        
    #sbml_aa = {id:aa for aa,id in sbml_aa_to_id.items()}

    sbml_dna = {'A': 'm404', 'T': 'm437', 'C': 'm431', 'G': 'm389'}

    sbml_rna = {'A': 'm94', 'U': 'm418', 'C': 'm423', 'G': 'm384'}

    dna_MW = {'A': 331.2, 'T': 322.2, 'G': 347.2, 'C': 307.2}

    rna_MW = {'A': 347.2, 'U': 324.2, 'G': 363.2, 'C': 323.2}

    aa_MW = {
        'A': 89.1,  'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2,
        'E': 147.1, 'Q': 146.2, 'G': 75.1,  'H': 155.2, 'I': 131.2,
        'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
        'S': 105.1,  'T': 119.1, 'W': 204.2,  'Y': 181.2, 'V': 117.1
        }

    seq_len = len(seq)
    counts = {}
    sequence_type = sequence_type.upper() # to allow user introduce lower and uppercase (should be fixed with app, because we control the input!)
    # Step 1: Get count for each amino acid 
    for bp in seq:
        if bp in counts:
            counts[bp] += 1
        else: 
            counts[bp] = 1
    # we need to consider the case that some protein might not contain some specific amino acids, and we need to specify that it is 0 and tell the MW dict to get the info from
    if sequence_type == 'AA':
        MW_dict = aa_MW
        model_ids = sbml_aa
        for amino_acid in sbml_aa.keys():
            if amino_acid not in counts.keys():
                counts[amino_acid] = 0
    elif sequence_type == 'DNA':
        MW_dict = dna_MW
        model_ids = sbml_dna
    elif sequence_type == 'RNA':
        MW_dict = rna_MW
        model_ids = sbml_rna
    else:
        print("ERROR: The sequence type introduced is not correct. Options are DNA, RNA or AA")
        sys.exit(1)
    
    # Step 2: Get gr/mol of protein for each amino acid
    gr_mol = {}
    for bp in counts:
        gr_mol[bp] = MW_dict[bp] * counts[bp] / seq_len

    # Step 4: Get mmol/ gr of protein for each amino acid
    mmol_gr = {}
    for bp in counts:
        mmol_gr[bp] = round(gr_mol[bp] / sum(gr_mol.values()) / MW_dict[bp] * 1000, 5)
    
    # change it to the ids used for the model, as we need it to automize the reaction definition
    mmol_gr_sbml = {}
    for bp in mmol_gr:   
        mmol_gr_sbml[model_ids[bp]] = - mmol_gr[bp]
    
    # we add the ATP stoichiometry as well as ADP, water and phosphate
    if sequence_type == 'DNA':
        atp_multiplier = 3.4
    elif sequence_type == 'RNA':
        atp_multiplier = 2.4
    elif sequence_type == 'AA':
        atp_multiplier = 4.3

    atp = sum(mmol_gr.values()) * atp_multiplier; 

    mmol_gr['ATP'] = round(atp, 5); 
    for count, metabolite in enumerate(['m1', 'm5', 'm3', 'm7']):
        if count < 2:
            mmol_gr_sbml[metabolite] = - round(atp, 5)
        else:
            mmol_gr_sbml[metabolite] = round(atp, 5)
    # finally include the product

    mmol_gr_sbml[protein_name + '_' + sequence_type] = 1 
    return mmol_gr_sbml

'''
try:
    input_file = open(original_model_file, 'r')
    output_file = open(new_model_file, 'w')
except IOError as err:
    print("Cant open file:", str(err));
    sys.exit(1)

reaction_flag = False

#dict for cleaner coding
reaction_dict = {
    'R_r1100': [aa_stoch, sbml_id_to_aa],
    'R_r1101': [dna_stoch, sbml_dna],
    'R_r1103': [rna_stoch, sbml_rna]
    }

for line in input_file: 
    
    #copy file line by line while looking for reaction
    if not reaction_flag:
        output_file.write(line)
        
        #look for reaction ID's specified in dict
        for reaction in reaction_dict:
            if line.find('id="' + reaction) != -1:
                
                #when reaction lines found, update flag and dictionaries
                reaction_flag = True
                stochiometry_dict = reaction_dict[reaction][0]
                sbml_ids = reaction_dict[reaction][1]
    

    #when reaction lines are reached
    if reaction_flag:
        
        #regex group 1 is the species, group 2 is the stoichiometric value
                           #<speciesReference species="M_m1360" stoichiometry="0.997" constant="true"/>
        regex = re.search(r'<speciesReference species="(.*)" stoichiometry="(.*)" constant="true"/>', line)
        if regex:
            species = regex.group(1)
            old_stochiometry = regex.group(2)
            
            #M_m3, -5 and -7 are all equivalent to M_m1
            if species in ('M_m3', 'M_m5', 'M_m7'):
                species = 'M_m1'
            
            #don't edit lines with stoic. value of 1
            if old_stochiometry == "1":
                output_file.write(line)
            #rewrite line with new value
            else:
                new_stochiometry = str(stochiometry_dict[sbml_ids[species]])
                print(new_stochiometry)
                output_file.write(line.replace(old_stochiometry, new_stochiometry))
        
        #keep copying file if reaction lines not found
        else:
            output_file.write(line)
            
            #update flag if end of reaction lines
            if line.find('</listOfProducts>') != -1:
                reaction_flag = False
'''