# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 18:28:55 2021

@author: TCA
"""

## Imports ##

import kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.factory import Factory
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.textinput import TextInput
from functools import partial

from kivy.core.window import Window
Window.size = (1000, 500)
Window.fullscreen = True

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.integrate import odeint


## Default Values ##

Aniger_transcript_rate = 1.01
Aniger_translate_rate = 1.02
Aniger_prot_deg = 1.03
Aniger_mrna_deg = 1.04

Bsubtilis_transcript_rate = 2.01
Bsubtilis_translate_rate = 2.02
Bsubtilis_prot_deg = 2.03
Bsubtilis_mrna_deg = 2.04

Ecoli_transcript_rate = 3.01
Ecoli_translate_rate = 3.02
Ecoli_prot_deg = 3.03
Ecoli_mrna_deg = 3.04

Kphaffii_transcript_rate = 1/12
Kphaffii_translate_rate = 4.6e-2*5
Kphaffii_prot_deg = 1.67e-5
Kphaffii_mrna_deg = 1.7e-4

Scervisiae_transcript_rate = 5.01
Scervisiae_translate_rate = 5.02
Scervisiae_prot_deg = 5.03
Scervisiae_mrna_deg = 5.04


## Variables for storing settings

set2dict = dict()   #dictionary for settings2screen variables

## Model Code ##

def plot(input_variables):

    ####
    # Define constants (come from the organism)
    ###

    
    ktx = float(input_variables[0])              #M/s maximum transcription rate
    ktl =  float(input_variables[1])         #M/s maximum translation constant
    deg_mRNA = float(input_variables[3])       #M/s degredation constant of mRNA
    deg_Protein = float(input_variables[2])   #M/s degredation constant of Protein
    
    
    # helper function to get activators & repressors concentration depending on time
    
    def transcription_induction(conc, t1, t2, t):
        return conc if t>t1 and t<t2 else 0
    
    ###
    #Define ODE
    ###
    
    def ODEs(variables, t, set2dict):
        #variables = list of concentrations, so here, [mRNA , Protein]. t = time
        mRNA = variables[0] 
        Protein = variables[1]  
        hill_coeff = 1 # allow them to insert app (otherwise, default 1)
        activator_component = 0
        repressor_component = 0
        for key in set2dict:
            inducer_concentration = transcription_induction(set2dict[key][5], set2dict[key][2], set2dict[key][3],t)
            if inducer_concentration > 0:
                if set2dict[key][1] == 'Activator':           
                    activator_component += inducer_concentration**hill_coeff/(set2dict[key][4]**hill_coeff+inducer_concentration**hill_coeff)
                elif set2dict[key][1] == 'Repressor': 
                    repressor_component += set2dict[key][4]**hill_coeff/(set2dict[key][4]**hill_coeff+inducer_concentration**hill_coeff)
        #if activator_component == 0 and repressor_component > 0:
            #activator_component = 1
        if activator_component > 0 and repressor_component == 0:
            repressor_component = 1
        
        leakiness = 0.00001  #s^-1
    
        dmRNA_dt = leakiness*ktx + (1-leakiness)*repressor_component*ktx*activator_component - deg_mRNA*mRNA
        
        # Protein
        dProtein_dt = ktl*mRNA - deg_Protein*Protein
    
        return [dmRNA_dt, dProtein_dt] 
    
    
    #####
    #Solving the ODEs
    #####
    
    t0 = 0      #Initial time
    t1 = 0
    for key in set2dict:
        if set2dict[key][3] > t1:
            t1 = set2dict[key][3]   #Final time
    total =  100000    #Number of time steps (larger the better)
    
    initial_conditions = [0.0, 0.0]        #set the initial values for [mRNA] and [Protein]
    t = np.linspace(t0,t1,total)                       #set the array of time values to integrate over
    
    
    solution = odeint(ODEs , initial_conditions , t, args = (set2dict,)) #Produces an 2d array of solutions
                                                         #for each variable wrt time
    mRNA = solution[:,0]    #Index all values in first column
    Protein = solution[:,1] #Index all values in second column
    
    
    #####
    #Plot the data
    #####
    
    # convert # molecules Protein into grams
    
    
    plt.plot(t/60/60, Protein, color = 'black')
    
    lab_count = 1
    for key in set2dict:
        plt.axvspan(set2dict[key][2]/3600,set2dict[key][3]/3600, fill = False, edgecolor = set2dict[key][6], hatch = set2dict[key][7], alpha = 0.6)
        plt.text((3*set2dict[key][2] + set2dict[key][3]) / 14400, 0.5e6*lab_count, set2dict[key][0], fontsize = 13, weight='bold')
        lab_count+=1
    
    plt.title("Effect of Activator(s)/Repressor(s) Concentration on Protein Production")
    plt.xlabel("Hours")
    plt.ylabel("mg Protein/gDW")
    plt.grid()
    plt.gcf().set_size_inches(10, 7)
    plt.savefig('./plot.png', dpi = 200)
    plt.clf()


## Kivy Code

## Written as .kv file (design file)
Builder.load_string("""
<PlotScreen>:
    
    BoxLayout:
        size: root.width, root.height
        
        Image:
            id: plot_img
            source: root.img
        
    AnchorLayout:
        anchor_x: 'right'
        anchor_y: 'top'
        
        Button:
            text: 'Settings'
            size_hint: 0.1, 0.1
            
            on_press:
                root.manager.transition.direction = 'down'
                root.manager.current = 'settings1'
    
    AnchorLayout:
        anchor_x: 'center'
        anchor_y: 'bottom'
    
        Button:
            text: 'Read Me'
            size_hint: 0.3, 0.1
            
            on_press:
                root.manager.transition.direction = 'up'
                root.manager.current = 'readme'
                
    AnchorLayout:
        anchor_x: 'left'
        anchor_y: 'top'
    
        Button:
            text: 'Save Figure'
            size_hint: 0.1, 0.1
            
            on_press:
                root.show_pop()


<ReadMeScreen>:
    
    BoxLayout:
        
        size: root.width, root.height
        Label:
            text: root.read_me
            text_size: self.size
            halign: 'left'
            valign: 'top'
    
    AnchorLayout:
        anchor_x: 'right'
        anchor_y: 'top'
        
        Button:
            text: 'Back'
            size_hint: 0.1, 0.1
            on_press:
                root.manager.transition.direction = 'down'
                root.manager.current = 'plot'
        

<Settings1Screen>:
    
    GridLayout:
        size: root.width, root.height
        rows: 6
        
        GridLayout:
            cols: 3
            
            Spinner:
                id: spinner_organism
                text: 'K. phaffii'
                values: ["A. niger", "B. subtilis", "E. coli", "K. phaffii", "S. cervisiae", "Custom"]
                
                on_text: root.orgspin_clicked(spinner_organism.text)
                
            TextInput:
                id: protein_sequence
                text: 'Insert Protein Sequence Here'
                
            Button:
                text: 'Repressors/Activators'
                on_press:
                    root.manager.transition.direction = 'left'
                    root.manager.current = 'settings2'
        
        GridLayout:
            cols: 2
            
            Label:
                size_hint: (0.5, 1)
                text: 'Transcription Rate'
                
            TextInput:
                id: transcript_rate
                text: '0.08333333333333333'
        
        GridLayout:
            cols: 2
            
            Label:
                size_hint: (0.5, 1)
                text: 'Translation Rate'
                
            TextInput:
                id: translate_rate
                text: '0.22999999999999998'
        
        GridLayout:
            cols: 2
            
            Label:
                size_hint: (0.5, 1)
                text: 'Protein Degradation Rate'
                
            TextInput:
                id: prot_deg
                text: '1.67e-05'
        
        GridLayout:
            cols: 2
            
            Label:
                size_hint: (0.5, 1)
                text: 'mRNA Degradation Rate'
                
            TextInput:
                id: mrna_deg
                text: '0.00017'
        
        Button:
            text: 'Plot'
            on_press:
                root.manager.transition.direction = 'up'
                root.manager.current = 'plot'
                
                root.plot_button()
                

<Settings2Screen>:
    
    GridLayout:
        cols: 6
        
        Button:
            text: 'Save'
            on_press:
                root.save_settings2()
                root.manager.transition.direction = 'right'
                root.manager.current = 'settings1'
        
        GridLayout:
            rows: 12
            
            TextInput:
                id: actrep_name1
                text: 'Methanol'
                disabled: True if actrep_spin1.text == 'Off' else False
            
            Spinner:
                id: actrep_spin1
                text: 'Activator'
                values: ["Activator", "Repressor", "Off"]
            
            Label:
                text: 'Start Time (h)'
                text_size: self.size
                halign: 'center'
                valign: 'middle'
            TextInput:
                id: actrep_start1
                text: '0.00'
                disabled: True if actrep_spin1.text == 'Off' else False
            
            Label:
                text: 'End Time (h)'
                text_size: self.size
                halign: 'center'
                valign: 'middle'
            TextInput:
                id: actrep_end1
                text: '100.00'
                disabled: True if actrep_spin1.text == 'Off' else False
            
            Label:
                text: 'Kd - Dissociation Constant (nM)'
                text_size: self.size
                halign: 'center'
                valign: 'middle'
            TextInput:
                id: actrep_kd1
                text: '200.00'
                disabled: True if actrep_spin1.text == 'Off' else False
            
            Label:
                text: 'Concentration (nM)'
                text_size: self.size
                halign: 'center'
                valign: 'middle'
            TextInput:
                id: actrep_conc1
                text: '300.00'
                disabled: True if actrep_spin1.text == 'Off' else False
            
            Spinner:
                id: actrep_color1
                text: 'Blue'
                values: ["Blue", "Red", "Yellow", "Green", "Purple"]
        
        GridLayout:
            rows: 12
            
            TextInput:
                id: actrep_name2
                text: 'Glucose'
                disabled: True if actrep_spin2.text == 'Off' else False
            
            Spinner:
                id: actrep_spin2
                text: 'Repressor'
                values: ["Activator", "Repressor", "Off"]
            
            Label:
                text: 'Start Time (h)'
                text_size: self.size
                halign: 'center'
                valign: 'middle'
            TextInput:
                id: actrep_start2
                text: '100.00'
                disabled: True if actrep_spin2.text == 'Off' else False
            
            Label:
                text: 'End Time (h)'
                text_size: self.size
                halign: 'center'
                valign: 'middle'
            TextInput:
                id: actrep_end2
                text: '200.00'
                disabled: True if actrep_spin2.text == 'Off' else False
            
            Label:
                text: 'Kd - Dissociation Constant (nM)'
                text_size: self.size
                halign: 'center'
                valign: 'middle'
            TextInput:
                id: actrep_kd2
                text: '100.00'
                disabled: True if actrep_spin2.text == 'Off' else False
            
            Label:
                text: 'Concentration'
            TextInput:
                id: actrep_conc2
                text: '200.00'
                disabled: True if actrep_spin2.text == 'Off' else False
            
            Spinner:
                id: actrep_color2
                text: 'Red'
                values: ["Blue", "Red", "Yellow", "Green", "Purple"]
        
        GridLayout:
            rows: 12
            
            TextInput:
                id: actrep_name3
                text: 'Name'
                disabled: True if actrep_spin3.text == 'Off' else False
            
            Spinner:
                id: actrep_spin3
                text: 'Off'
                values: ["Activator", "Repressor", "Off"]
            
            Label:
                text: 'Start Time (h)'
                text_size: self.size
                halign: 'center'
                valign: 'middle'
            TextInput:
                id: actrep_start3
                text: '0.00'
                disabled: True if actrep_spin3.text == 'Off' else False
            
            Label:
                text: 'End Time (h)'
                text_size: self.size
                halign: 'center'
                valign: 'middle'
            TextInput:
                id: actrep_end3
                text: '0.00'
                disabled: True if actrep_spin3.text == 'Off' else False
            
            Label:
                text: 'Kd - Dissociation Constant (nM)'
                text_size: self.size
                halign: 'center'
                valign: 'middle'
            TextInput:
                id: actrep_kd3
                text: '0.00'
                disabled: True if actrep_spin3.text == 'Off' else False
            
            Label:
                text: 'Concentration'
            TextInput:
                id: actrep_conc3
                text: '0.00'
                disabled: True if actrep_spin3.text == 'Off' else False
            
            Spinner:
                id: actrep_color3
                text: 'Yellow'
                values: ["Blue", "Red", "Yellow", "Green", "Purple"]

        GridLayout:
            rows: 12
            
            TextInput:
                id: actrep_name4
                text: 'Name'
                disabled: True if actrep_spin4.text == 'Off' else False
            
            Spinner:
                id: actrep_spin4
                text: 'Off'
                values: ["Activator", "Repressor", "Off"]
            
            Label:
                text: 'Start Time (h)'
                text_size: self.size
                halign: 'center'
                valign: 'middle'
            TextInput:
                id: actrep_start4
                text: '0.00'
                disabled: True if actrep_spin4.text == 'Off' else False
            
            Label:
                text: 'End Time (h)'
                text_size: self.size
                halign: 'center'
                valign: 'middle'
            TextInput:
                id: actrep_end4
                text: '0.00'
                disabled: True if actrep_spin4.text == 'Off' else False
            
            Label:
                text: 'Kd - Dissociation Constant (nM)'
                text_size: self.size
                halign: 'center'
                valign: 'middle'
            TextInput:
                id: actrep_kd4
                text: '0.00'
                disabled: True if actrep_spin4.text == 'Off' else False
            
            Label:
                text: 'Concentration'
            TextInput:
                id: actrep_conc4
                text: '0.00'
                disabled: True if actrep_spin4.text == 'Off' else False
            
            Spinner:
                id: actrep_color4
                text: 'Green'
                values: ["Blue", "Red", "Yellow", "Green", "Purple"]
        
        GridLayout:
            rows: 12
            
            TextInput:
                id: actrep_name5
                text: 'Name'
                disabled: True if actrep_spin5.text == 'Off' else False
            
            Spinner:
                id: actrep_spin5
                text: 'Off'
                values: ["Activator", "Repressor", "Off"]
            
            Label:
                text: 'Start Time (h)'
                text_size: self.size
                halign: 'center'
                valign: 'middle'
            TextInput:
                id: actrep_start5
                text: '0.00'
                disabled: True if actrep_spin5.text == 'Off' else False
            
            Label:
                text: 'End Time (h)'
                text_size: self.size
                halign: 'center'
                valign: 'middle'
            TextInput:
                id: actrep_end5
                text: '0.00'
                disabled: True if actrep_spin5.text == 'Off' else False
            
            Label:
                text: 'Kd - Dissociation Constant (nM)'
                text_size: self.size
                halign: 'center'
                valign: 'middle'
            TextInput:
                id: actrep_kd5
                text: '0.00'
                disabled: True if actrep_spin5.text == 'Off' else False
            
            Label:
                text: 'Concentration'
            TextInput:
                id: actrep_conc5
                text: '0.00'
                disabled: True if actrep_spin5.text == 'Off' else False
            
            Spinner:
                id: actrep_color5
                text: 'Purple'
                values: ["Blue", "Red", "Yellow", "Green", "Purple"]
""")


## Screen Classes

class PlotScreen(Screen):
    img = './main.jpg'

    def show_pop(self):
        
        pop_label = Label(text='Insert Path to save file:')
        pop_input = TextInput(text='C:\\Example\\Path\\Documents',
                                          multiline=False)
        pop_button = Button(text='Save')
        
        save_content = GridLayout(rows=3)
        save_content.add_widget(pop_label)
        save_content.add_widget(pop_input)
        save_content.add_widget(pop_button)
        
        popupWindow = Popup(title="Export Figure",
                            content=save_content,
                            size_hint=(None,None),
                            size=(400,200))
        
        #pop_button.bind(on_press=partial(print_something, popupWindow))
        pop_button.bind(on_press=partial(print, pop_input.text))
        pop_button.bind(on_press=popupWindow.dismiss)
        
        popupWindow.open()
    

class Settings1Screen(Screen):
    
    def orgspin_clicked(self, value):
        #Update all variables according to user choice
        if value == "A. niger":
            self.ids.transcript_rate.text = str(Aniger_transcript_rate)
            self.ids.translate_rate.text = str(Aniger_translate_rate)
            self.ids.prot_deg.text = str(Aniger_prot_deg)
            self.ids.mrna_deg.text = str(Aniger_mrna_deg)

        elif value == "B. subtilis":
            self.ids.transcript_rate.text = str(Bsubtilis_transcript_rate)
            self.ids.translate_rate.text = str(Bsubtilis_translate_rate)
            self.ids.prot_deg.text = str(Bsubtilis_prot_deg)
            self.ids.mrna_deg.text = str(Bsubtilis_mrna_deg)

        elif value == "E. coli":
            self.ids.transcript_rate.text = str(Ecoli_transcript_rate)
            self.ids.translate_rate.text = str(Ecoli_translate_rate)
            self.ids.prot_deg.text = str(Ecoli_prot_deg)
            self.ids.mrna_deg.text = str(Ecoli_mrna_deg)

        elif value == "K. phaffii":
            self.ids.transcript_rate.text = str(Kphaffii_transcript_rate)
            self.ids.translate_rate.text = str(Kphaffii_translate_rate)
            self.ids.prot_deg.text = str(Kphaffii_prot_deg)
            self.ids.mrna_deg.text = str(Kphaffii_mrna_deg)

        elif value == "S. cervisiae":
            self.ids.transcript_rate.text = str(Scervisiae_transcript_rate)
            self.ids.translate_rate.text = str(Scervisiae_translate_rate)
            self.ids.prot_deg.text = str(Scervisiae_prot_deg)
            self.ids.mrna_deg.text = str(Scervisiae_mrna_deg)

        elif value == "Custom":
            self.ids.transcript_rate.text = "0.00"
            self.ids.translate_rate.text = "0.00"
            self.ids.prot_deg.text = "0.00" 
            self.ids.mrna_deg.text = "0.00"
    
    def plot_button(self):
        
        self.manager.get_screen("settings2").save_settings2()
        
        variables = [
            self.ids.transcript_rate.text,
            self.ids.translate_rate.text,
            self.ids.prot_deg.text,
            self.ids.mrna_deg.text
            ]
        
        plot(variables)
        
        self.manager.get_screen("plot").ids.plot_img.source = './plot.png'
        self.manager.get_screen("plot").ids.plot_img.reload()


class Settings2Screen(Screen):
    
    def error_message(self):
        
        err_label = Label(text='''Start Time can not be higher than End Time.
            Start Time has been adjusted.''')
        err_button = Button(text='OK')
        
        err_content = GridLayout(rows=2)
        err_content.add_widget(err_label)
        err_content.add_widget(err_button)
        
        popupWindow = Popup(title="Warning Message",
                            content=err_content,
                            size_hint=(None,None),
                            size=(400,200))
        
        err_button.bind(on_press=popupWindow.dismiss)
        
        popupWindow.open()
    
    def save_settings2(self):
        #activator repressor 1 variables into dict (if not off)
        if self.ids.actrep_spin1.text == "Off":
            set2dict.pop('actrep1', None)
        else:
            #check if start time is higher than end time
            if float(self.ids.actrep_start1.text) > float(self.ids.actrep_end1.text):
                self.ids.actrep_start1.text = self.ids.actrep_end1.text
                
                self.error_message()
            
            set2dict['actrep1'] = [
                self.ids.actrep_name1.text,
                self.ids.actrep_spin1.text,
                float(self.ids.actrep_start1.text)*3600,
                float(self.ids.actrep_end1.text)*3600,
                float(self.ids.actrep_kd1.text),
                float(self.ids.actrep_conc1.text),
                self.ids.actrep_color1.text,
                '\\\\\\'
                ]
        
        #activator repressor 2 variables into dict (if not off)
        if self.ids.actrep_spin2.text == "Off":
            set2dict.pop('actrep2', None)
        else:
            set2dict['actrep2'] = [
                self.ids.actrep_name2.text,
                self.ids.actrep_spin2.text,
                float(self.ids.actrep_start2.text)*3600,
                float(self.ids.actrep_end2.text)*3600,
                float(self.ids.actrep_kd2.text),
                float(self.ids.actrep_conc2.text),
                self.ids.actrep_color2.text,
                '///'
                ]
        
        #activator repressor 3 variables into dict (if not off)
        if self.ids.actrep_spin3.text == "Off":
            set2dict.pop('actrep3', None)
        else:
            set2dict['actrep3'] = [
                self.ids.actrep_name3.text,
                self.ids.actrep_spin3.text,
                float(self.ids.actrep_start3.text)*3600,
                float(self.ids.actrep_end3.text)*3600,
                float(self.ids.actrep_kd3.text),
                float(self.ids.actrep_conc3.text),
                self.ids.actrep_color3.text,
                '||'
                ]
        
        #activator repressor 4 variables into dict (if not off)
        if self.ids.actrep_spin4.text == "Off":
            set2dict.pop('actrep4', None)
        else:
            set2dict['actrep4'] = [
                self.ids.actrep_name4.text,
                self.ids.actrep_spin4.text,
                float(self.ids.actrep_start4.text)*3600,
                float(self.ids.actrep_end4.text)*3600,
                float(self.ids.actrep_kd4.text),
                float(self.ids.actrep_conc4.text),
                self.ids.actrep_color4.text,
                '--'
                ]
        
        #activator repressor 5 variables into dict (if not off)
        if self.ids.actrep_spin5.text == "Off":
            set2dict.pop('actrep5', None)
        else:
            set2dict['actrep5'] = [
                self.ids.actrep_name5.text,
                self.ids.actrep_spin5.text,
                float(self.ids.actrep_start5.text)*3600,
                float(self.ids.actrep_end5.text)*3600,
                float(self.ids.actrep_kd5.text),
                float(self.ids.actrep_conc5.text),
                self.ids.actrep_color5.text,
                '.'
                ]

class ReadMeScreen(Screen):
    
    read_me = """
    
Welcome to Enzymatics!


The DTU Biobuilders 2021 team developed this app to help you get an idea of the dynamics your cell factory will \
follow for protein production. Our intention is to make a very straightforward application that allows you to \
easily and quickly get an idea of how different activators and repressors would affect your protein expression. \
One of the main advantages of this app is that it allows you to work with different organisms with default kinetic \
parameters found in the literature. However, you are also allowed to use your own custom organism if you know \
their specific dynamics.
Let us just give you a quick tour through our app and its different screens so you can quickly start using it!

- Main screen: This is the first screen you will see when running Enzymatics with our welcome message, and also \
the screen where your plot will be shown once you introduce your data.
- Settings: On the top right corner of the main screen you will find the "Settings" button that will bring you to \
the page where the transcription rate, translation rate, mRNA and protein degradation can be defined, as well as \
the option to include the protein of interest you want to express. Inside this page you have:
      
- Organism: On the top left corner you can choose your organism for the expression. As mentioned above, we \
already defined the default values for some widely used organisms obtained from the literature, \
but you are also allowed to use the Custom option to include your own organism and its dynamics.
- Protein sequence: In the top middle part you can include your protein sequence to express. From this we \
will update the proten degradation rate depending on its sequence following literature. Again, a \
default sequence for the xxx protein is added.
- Repressor & Activator settings: Top right corner you will find this button that will bring you to the' \
next screen.
      
- Repressor & Activator settings: On this last screen you can define up to 5 inducers, either activators or \
repressors, which you can freely choose. For each of those you are required to include the kinetics data being:
    
- Initial time: When the inducer is added to the experiment
- Final time: When the inducer is removed from the experiment
- Kd : Enzymatic constant for the given inducer when attaching to the TF.
- Concentration: Inducer concentration
    
In order to help you some default values have been added so you can get a default plot.
Once you are done setting everything, you hit the "Save" button on the left-hand side to go to the setting screen, \
where hitting the "Plot!" button at the bottom will bring you to the main screen with your awesome plot!
"""


class AppV2(App):
    
    def build(self):
        
        # Create the screen manager
        sm = ScreenManager()
        sm.add_widget(PlotScreen(name='plot'))
        sm.add_widget(Settings1Screen(name='settings1'))
        sm.add_widget(Settings2Screen(name='settings2'))
        sm.add_widget(ReadMeScreen(name='readme'))
        
        return sm


if __name__ == "__main__":
    AppV2().run()

