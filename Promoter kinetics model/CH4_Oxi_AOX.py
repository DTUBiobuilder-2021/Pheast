import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np

####
# Define constants
###


### all parameters taken from cerevisiae


#Transcription:
# Pelechano, V., Chávez, S., & Pérez-Ortín, J. E. (2010). A complete set of nascent transcription rates for yeast genes. PloS one, 5(11), e15442.
# average transcription rate of 0.12 molecules/min. we take a higher number as AOX promoter should be highly induced under methanol condition. in the study peroxisomal gene CIT2 was transcribed at 1 molecule/min
# rna-seq data from phaffii reveals that it is highly transcribed; 85* higher number of reads (normalized) than average. we take 85 * 0.12 = 10.2 it seems too high. the highest observed in the pelechano study was 7.5, so 5 molecules/min might be good guess 

ktx = 1/12                  #M/s maximum transcription rate
ktx_pMMO = 1/18


# Bonven B, Gulløv K. Peptide chain elongation rate and ribosomal activity in Saccharomyces cerevisiae as a function of the growth rate. Mol Gen Genet. 1979 Feb 26 170(2):225-30
# numer of nucleotides is 481 --> (481/3)/7.5 (7.5 is "medium" value from paper for AA elongation per second) --> translation per second 
# Riba, A., Di Nanni, N., Mittal, N., Arhné, E., Schmidt, A., & Zavolan, M. (2019). Protein synthesis rates and ribosome occupancies reveal determinants of translation elongation rates. Proceedings of the national academy of sciences, 116(30), 15023-15032.
# in the second paper, ribosomes per codon and protein synthesis rates are considered. it is quite clear that one can expect a ribosome every 10-100 codons. they state the AA/s rate as ranging from 1-20, so we will keep 7.5 and simulate with 5 ribosomes per transcript as a conservative guess and within their average rates. 

ktl =  4.6e-2*5              #M/s maximum translation constant
# difference as different length
ktl_pMMO = 8.1e-3*5

# Geisberg, Joseph V., et al. "Global analysis of mRNA isoform half-lives reveals stabilizing and destabilizing elements in yeast." Cell 156.4 (2014): 812-824.

# mRNA half life medians usually between 20-30 minutes, we take 25 = 1500 second = 1/1500 per second
# no good reason/expertise to use two different rates here
deg_mRNA = 6.7e-4           #/s degredation constant of mRNA

# Christiano R, Nagaraj N, Fröhlich F, Walther TC. Global proteome turnover analyses of the Yeasts S. cerevisiae and S. pombe. Cell Rep. 2014 Dec 11 9(5):1959-65. doi: 10.1016/j.celrep.2014.10.065. Supplemental Information p.S12 table S4
# 
# average and median half life is 43 min according to : Belle A, Tanay A, Bitincka L, Shamir R, O'Shea EK. Quantification of protein half-lives in the budding yeast proteome. Proc Natl Acad Sci U S A. 2006 Aug 29 103(35):13004-9 p.13004 right column 4th paragraph

# as hemoglobin is big and fulfils important function, it could have longer half life, like that of the ones with high half life defined in paper by christiano et al.
# so 5hrs = 21600s are taken ### we could look up the half life of our particular hemoglobin further
# again no good reason to use two different rates here

deg_Protein = 1/21600#1.67e-5      #/s degredation constant of Protein



###
#Define ODE
###

def ODEs(variables, t):
    #variables = list of concentrations, so here, [mRNA , Protein]. t = time
    mRNA = variables[0] 
    Protein = variables[1]
    mRNA_GAP = variables[2]
    Protein_pMMO = variables[3]
    methanol = variables[4]
    
    # Kumar, N. V., & Rangarajan, P. N. (2012). The zinc finger proteins Mxr1p and repressor of phosphoenolpyruvate carboxykinase (ROP) have the same DNA binding specificity but regulate methanol metabolism antagonistically in Pichia pastoris. Journal of biological chemistry, 287(41), 34465-34473.
    # this paper describes mxr1 Kd bound to AOX promoter at 200 nM
    
    # Tschopp, J. F., Brust, P. F., Cregg, J. M., Stillman, C. A., & Gingeras, T. R. (1987). Expression of the lacZ gene from two methanol-regulated promoters in Pichia pastoris. Nucleic acids research, 15(9), 3859-3876.
    # this paper uses 0.5% methanol and gives a rough time course of aox activation
    # Santoso, A., Herawati, N., & Rubiana, Y. (2012). Effect of methanol induction and incubation time on expression of human erythropoietin in methylotropic yeast Pichia pastoris. Makara Journal of Technology, 16(1), 5.
    # this paper found that at 2.5% methanol, cell growth is optimal. they tried out different levels and measured aox expression but did not provide data
    # Van Dijken, L. P., Otto, R., & Harder, W. (1976). Growth of Hansenula polymorpha in a methanol-limited chemostat. Archives of microbiology, 111(1), 137-144.
    # this paper describes the oxygen consumption rates at 3 different levels of methanol. might be a usable indirect measurement of the methanol oxidation activity and therefore AOX presence


    Km_AOX = 0.187 #%

   
    hill_coeff_AOX_methanol = 1.264

   # Anggiani, M., Helianti, I., & Abinawanto, A. (2018, October). Optimization of methanol induction for expression of synthetic gene Thermomyces lanuginosus lipase in Pichia pastoris. In AIP Conference Proceedings (Vol. 2023, No. 1, p. 020157). AIP Publishing LLC.
   # they measured aox expression for methanol concentrations 0.5-3. we fit the data read by eye (also asked for raw data) and we get a coefficient of about 1.264



# arbitrary numbers, try so that concentration is just a little above Kd (steep curve will make it big fast)
# as it is just an artificial hill term for repression

    coef_repr = 10
    K_repressor = 50
    conc_repr = 0
    repressor = K_repressor**coef_repr/(K_repressor**coef_repr+conc_repr**coef_repr)

    # GAP
    dmRNA_GAP_dt = ktx - deg_mRNA * mRNA_GAP

    yeast_weight = 4.6e-11 # g 
    # 0.05 because 10% of cell protein = 5% of total cell weight,then how many gram per liter, then divided by weight of pmmo to get max number of molecules
    yeast_per_liter = 100/yeast_weight
    # max_pMMO = 0.05*yeast_weight*yeast_per_liter/4.981620599999999e-19# molecules/yeast cell, corresponding to 10% of cell protein
    factor = 10
    #bound_term_GAP = max_pMMO**factor/(max_pMMO**factor+Protein_pMMO**factor)
    
    dProtein_pMMO_dt = ktl_pMMO * mRNA_GAP - deg_Protein * Protein_pMMO

    # weight_pMMO = 4.9816e-19 # g # used as number later
     
    # as turnover rate is about 1/s for methane -> methanol of pMMO
    # Verachtert, H. (1989). In Yeast: Biotechnology and Biocatalysis (p. 410). CRC Press. 
    # they actually state P.pastoris has dry weights of up to 125g, we go with conservative guess
    # Siegel, R. S., & Brierley, R. A. (1989). Methylotrophic yeast Pichia pastoris produced in high‐cell‐density fermentations with high cell yields as vehicle for recombinant protein production. Biotechnology and bioengineering, 34(3), 403-404.
    # they report cell density of 125 but not specified as dry/wet cell weight. In bakers yeast from supermarket dry-wet weight is also stated with a factor of 4-5
    yeast_per_liter = 25/yeast_weight
    mole = 6.02214076e23
    # to get from pMMO no of molecules to mM methanol per liter fermentation solution, 
    #dmethanol_dt = 1000*((Protein_GAP*yeast_per_liter)/mole)
    
    # 100 for percent, 32.04 is molar mass, 792 is density g/liter, 1000 because its ml and it was calculated per liter
    gram_methanol_per_liter = (32.04*((Protein_pMMO*yeast_per_liter)/mole))
    # divided by 0.792 gives ml of methanol in one liter
    # 

    dmethanol_dt = 100*(gram_methanol_per_liter/0.792)/1000

    
    hill_eq_AOX_vs_methanol = methanol**hill_coeff_AOX_methanol/(Km_AOX**hill_coeff_AOX_methanol+methanol**hill_coeff_AOX_methanol)

    # RNA
    
    yeast_weight = 4.6e-11 # g 
    # max_Hemo = 0.05*yeast_weight/2.65686246e-20 # g molecules/yeast cell, corresponding to 10% of cell protein
    # factor = 10
    # bound_term = max_Hemo**factor/(max_Hemo**factor+Protein**factor)

    dmRNA_dt = repressor*ktx*hill_eq_AOX_vs_methanol - deg_mRNA*mRNA
    
    # Protein
 
    dProtein_dt = ktl*mRNA - deg_Protein*Protein
    

    # without bound term
    dProtein_dt =   ktl*mRNA - deg_Protein*Protein
    

    return [dmRNA_dt, dProtein_dt,dmRNA_GAP_dt,dProtein_pMMO_dt,dmethanol_dt] 


#####
#Solving the ODEs
#####
t0 = 0              #Initial time
t1 = 1200000    #Final time
total =  100000    #Number of time steps (larger the better)

initial_conditions = [0.0, 0.0, 0.0, 0.0, 0.0]        #set the initial values for [mRNA] and [Protein]
t = sp.linspace(t0,t1,total)                       #set the array of time values to integrate over

solution = odeint(ODEs , initial_conditions , t) #Produces an 2d array of solutions
               
    
yeast_weight = 4.6e-11 # g 


    #for each variable wrt time
mRNA = solution[:,0]    #Index all values in first column
Protein = solution[:,1] #Index all values in second column
mRNA_GAP = solution[:,2]    #Index all values in first column
pMMO = solution[:,3]*1000*(1/yeast_weight)*4.981620599999999e-19
# same conversion from no. molecules into mg/gDW with mol. weight of pMMO at about     #Index all values in first column
methanol = solution[:,4]    #Index all values in first column
#####
#Plot the data; different settings (outcommented) for different approaches
#####

# convert # molecules Protein into grams

# yeast dry weight: https://tipbiosystems.com/wp-content/uploads/2020/05/AN102-Yeast-Cell-Count_2019_03_17.pdf
# specifically dry weight




Protein = Protein*2.65686246e-20*(1/yeast_weight)*1000

### Ploting should be made prettier, some of the plt functions dont work with figures
fig , axs = plt.subplots(2)
#fig.tight_layout()
#p1 = axs.plot(t/3600 , pMMO, color='#1A3274',label='pMMO')


#p2 = axs.plot(t/3600 , Protein, color='#9A1819', label='Hemoglobin')
#axs.set_ylabel('mg Protein/gDW')

#axs2 = axs.twinx()
#axs2.axhline(y=0.187, color='#5CA08E', linestyle='dotted',label='K_m AOX')
#axs2.text(7,0.197, color="#5CA08E",s='Km AOX')
#axs2.set_ylabel('% methanol')
#p3 = axs2.plot(t/3600, methanol, color='orange', label='methanol',linestyle='dashed')

#ps = p1+p2+p3
#labs = [l.get_label() for l in ps]
#axs.legend(ps, labs, loc=0)

axs[1].set_xlabel("days")
axs[0].set_ylabel("mg Protein/gDW")
axs[1].set_ylabel("mg Protein/gDW")
axs[0].grid()
axs[1].grid()

fig.suptitle("Methane Oxidation and AOX Induction")


p1 = axs[0].plot(t/3600/24 , pMMO, label = "pMMO",color='#1A3274',alpha=0.5,linestyle='dotted')
axs2 = axs[0].twinx()
p2 = axs2.plot(t/3600/24 , methanol, label = "methanol",color='orange')
axs2.set_ylabel('% methanol')
axs[1].plot(t/3600/24 , Protein,color='#9A1819', label='Leghemoglobin')
ps = p1+p2
labs = [l.get_label() for l in ps]
axs2.legend(ps,labs,loc='best', bbox_to_anchor=(0.2, 0.4, 0.5, 0.5))
#axs[1].legend(loc='center right')
axs2.axhline(y=0.187, color='#5CA08E',linestyle='dashed')
axs[0].text(11,0.3, color="#5CA08E",s='Km AOX')
#axs[4].plot(t/3600 , Protein, label = "Hemo",color='black')
fig.tight_layout()
plt.savefig('GAP_3.png')
plt.show()

###
#first attempt of dynamically plotting subplotes
###

#step = 100
#
#
#data = [mRNA, Protein, pMMO, mRNA_GAP, methanol]
#sim_len = len(mRNA)
#
#for i in range (0, sim_len, step):
#    axs[0].plot(t[:i]/60,data[0][:i])
#    axs[1].plot(t[:i]/60,data[1][:i])
#    axs[2].plot(t[:i]/60,data[2][:i])
#    axs[3].plot(t[:i]/60,data[3][:i])
#    axs[4].plot(t[:i]/60,data[4][:i])
#    plt.draw()
#    plt.pause(1e-20)
##plt.show()

#from matplotlib import font_manager
#dirs = ['~/Downloads/Helvetica-Font']
#files = font_manager.findSystemFonts(fontpaths=dirs)
#for file in files:
#    font_manager.fontManager.addfont(font_file)
#plt.rcParams['font.family'] = 'sans-serif'
#plt.rcParams['font.sans-serif'] = 'Comic Sans MS'
#plt.plot(methanol)
#plt.plot(pMMO)
#plt.plot(Protein)


