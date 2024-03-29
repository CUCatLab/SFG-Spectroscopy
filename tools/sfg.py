import os
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
from pandas import DataFrame as df
import matplotlib.pyplot as plt
import cmath
import yaml
import re
from lmfit import model, Model
from lmfit.models import GaussianModel, SkewedGaussianModel, VoigtModel, ConstantModel, LinearModel, QuadraticModel, PolynomialModel
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import ipywidgets as widgets
from ipywidgets import Button, Layout
from IPython.display import clear_output
from multiprocessing import Pool
from . import datatools as dt


class fitTools :
    
    def __init__(self,Data,FitInfo,Name='') :
        
        self.Data = Data
        self.FitInfo = FitInfo
        self.Name = Name
        
        try :
            FitInfo['ModelType']
            FitInfo['Models']
        except:
            ModelType = 'None'
            ModelString = ''
        else :
            if FitInfo['ModelType'] == 'BuiltIn' :
                self.BuiltInModels()
            if FitInfo['ModelType'] == 'SFG' :
                self.SFGModel()
    
    def BuiltInModels(self) :
        
        FitInfo = self.FitInfo
        
        ModelString = list()
        for key in FitInfo['Models'] :
            ModelString.append((key,FitInfo['Models'][key]['model']))
        
        for Model in ModelString :
            try :
                FitModel
            except :
                if Model[1] == 'Constant' :
                    FitModel = ConstantModel(prefix=Model[0]+'_')
                if Model[1] == 'Linear' :
                    FitModel = LinearModel(prefix=Model[0]+'_')
                if Model[1] == 'Gaussian' :
                    FitModel = GaussianModel(prefix=Model[0]+'_')
                if Model[1] == 'SkewedGaussian' :
                    FitModel = SkewedGaussianModel(prefix=Model[0]+'_')
                if Model[1] == 'Voigt' :
                    FitModel = VoigtModel(prefix=Model[0]+'_')
            else :
                if Model[1] == 'Constant' :
                    FitModel = FitModel + ConstantModel(prefix=Model[0]+'_')
                if Model[1] == 'Linear' :
                    FitModel = FitModel + LinearModel(prefix=Model[0]+'_')
                if Model[1] == 'Gaussian' :
                    FitModel = FitModel + GaussianModel(prefix=Model[0]+'_')
                if Model[1] == 'SkewedGaussian' :
                    FitModel = FitModel + SkewedGaussianModel(prefix=Model[0]+'_')
                if Model[1] == 'Voigt' :
                    FitModel = FitModel + VoigtModel(prefix=Model[0]+'_')
        
        self.FitModel = FitModel
        self.ModelParameters = FitModel.make_params()
        
    def SFGModel(self) :
        
        FitInfo = self.FitInfo
        
        ModelString = list()
        for key in FitInfo['Models'] :
            ModelString.append([key])
        
        if len(ModelString) == 2 :
            def SFGFunction(x,NonRes_amp,
                            Peak1_amp,Peak1_phi,Peak1_omega,Peak1_gamma) :
                Peaks = NonRes_amp
                Peaks+= Peak1_amp*(cmath.exp(Peak1_phi*1j)/(x-Peak1_omega+Peak1_gamma*1j))
                return np.real(Peaks*np.conjugate(Peaks))
        elif len(ModelString) == 3 :
            def SFGFunction(x,NonRes_amp,
                            Peak1_amp,Peak1_phi,Peak1_omega,Peak1_gamma,
                            Peak2_amp,Peak2_phi,Peak2_omega,Peak2_gamma) :
                Peaks = NonRes_amp
                Peaks+= Peak1_amp*(cmath.exp(Peak1_phi*1j)/(x-Peak1_omega+Peak1_gamma*1j))
                Peaks+= Peak2_amp*(cmath.exp(Peak2_phi*1j)/(x-Peak2_omega+Peak2_gamma*1j))
                return np.real(Peaks*np.conjugate(Peaks))
        elif len(ModelString) == 4 :
            def SFGFunction(x,NonRes_amp,
                            Peak1_amp,Peak1_phi,Peak1_omega,Peak1_gamma,
                            Peak2_amp,Peak2_phi,Peak2_omega,Peak2_gamma,
                            Peak3_amp,Peak3_phi,Peak3_omega,Peak3_gamma) :
                Peaks = NonRes_amp
                Peaks+= Peak1_amp*(cmath.exp(Peak1_phi*1j)/(x-Peak1_omega+Peak1_gamma*1j))
                Peaks+= Peak2_amp*(cmath.exp(Peak2_phi*1j)/(x-Peak2_omega+Peak2_gamma*1j))
                Peaks+= Peak3_amp*(cmath.exp(Peak3_phi*1j)/(x-Peak3_omega+Peak3_gamma*1j))
                return np.real(Peaks*np.conjugate(Peaks))
        elif len(ModelString) == 5 :
            def SFGFunction(x,NonRes_amp,
                            Peak1_amp,Peak1_phi,Peak1_omega,Peak1_gamma,
                            Peak2_amp,Peak2_phi,Peak2_omega,Peak2_gamma,
                            Peak3_amp,Peak3_phi,Peak3_omega,Peak3_gamma,
                            Peak4_amp,Peak4_phi,Peak4_omega,Peak4_gamma) :
                Peaks = NonRes_amp
                Peaks+= Peak1_amp*(cmath.exp(Peak1_phi*1j)/(x-Peak1_omega+Peak1_gamma*1j))
                Peaks+= Peak2_amp*(cmath.exp(Peak2_phi*1j)/(x-Peak2_omega+Peak2_gamma*1j))
                Peaks+= Peak3_amp*(cmath.exp(Peak3_phi*1j)/(x-Peak3_omega+Peak3_gamma*1j))
                Peaks+= Peak4_amp*(cmath.exp(Peak4_phi*1j)/(x-Peak4_omega+Peak4_gamma*1j))
                return np.real(Peaks*np.conjugate(Peaks))
        elif len(ModelString) == 6 :
            def SFGFunction(x,NonRes_amp,
                            Peak1_amp,Peak1_phi,Peak1_omega,Peak1_gamma,
                            Peak2_amp,Peak2_phi,Peak2_omega,Peak2_gamma,
                            Peak3_amp,Peak3_phi,Peak3_omega,Peak3_gamma,
                            Peak4_amp,Peak4_phi,Peak4_omega,Peak4_gamma,
                            Peak5_amp,Peak5_phi,Peak5_omega,Peak5_gamma) :
                Peaks = NonRes_amp
                Peaks+= Peak1_amp*(cmath.exp(Peak1_phi*1j)/(x-Peak1_omega+Peak1_gamma*1j))
                Peaks+= Peak2_amp*(cmath.exp(Peak2_phi*1j)/(x-Peak2_omega+Peak2_gamma*1j))
                Peaks+= Peak3_amp*(cmath.exp(Peak3_phi*1j)/(x-Peak3_omega+Peak3_gamma*1j))
                Peaks+= Peak4_amp*(cmath.exp(Peak4_phi*1j)/(x-Peak4_omega+Peak4_gamma*1j))
                Peaks+= Peak5_amp*(cmath.exp(Peak5_phi*1j)/(x-Peak5_omega+Peak5_gamma*1j))
                return np.real(Peaks*np.conjugate(Peaks))
        elif len(ModelString) == 7 :
            def SFGFunction(x,NonRes_amp,
                            Peak1_amp,Peak1_phi,Peak1_omega,Peak1_gamma,
                            Peak2_amp,Peak2_phi,Peak2_omega,Peak2_gamma,
                            Peak3_amp,Peak3_phi,Peak3_omega,Peak3_gamma,
                            Peak4_amp,Peak4_phi,Peak4_omega,Peak4_gamma,
                            Peak5_amp,Peak5_phi,Peak5_omega,Peak5_gamma,
                            Peak6_amp,Peak6_phi,Peak6_omega,Peak6_gamma) :
                Peaks = NonRes_amp
                Peaks+= Peak1_amp*(cmath.exp(Peak1_phi*1j)/(x-Peak1_omega+Peak1_gamma*1j))
                Peaks+= Peak2_amp*(cmath.exp(Peak2_phi*1j)/(x-Peak2_omega+Peak2_gamma*1j))
                Peaks+= Peak3_amp*(cmath.exp(Peak3_phi*1j)/(x-Peak3_omega+Peak3_gamma*1j))
                Peaks+= Peak4_amp*(cmath.exp(Peak4_phi*1j)/(x-Peak4_omega+Peak4_gamma*1j))
                Peaks+= Peak5_amp*(cmath.exp(Peak5_phi*1j)/(x-Peak5_omega+Peak5_gamma*1j))
                Peaks+= Peak6_amp*(cmath.exp(Peak6_phi*1j)/(x-Peak6_omega+Peak6_gamma*1j))
                return np.real(Peaks*np.conjugate(Peaks))
        elif len(ModelString) == 8 :
            def SFGFunction(x,NonRes_amp,
                            Peak1_amp,Peak1_phi,Peak1_omega,Peak1_gamma,
                            Peak2_amp,Peak2_phi,Peak2_omega,Peak2_gamma,
                            Peak3_amp,Peak3_phi,Peak3_omega,Peak3_gamma,
                            Peak4_amp,Peak4_phi,Peak4_omega,Peak4_gamma,
                            Peak5_amp,Peak5_phi,Peak5_omega,Peak5_gamma,
                            Peak6_amp,Peak6_phi,Peak6_omega,Peak6_gamma,
                            Peak7_amp,Peak7_phi,Peak7_omega,Peak7_gamma) :
                Peaks = NonRes_amp
                Peaks+= Peak1_amp*(cmath.exp(Peak1_phi*1j)/(x-Peak1_omega+Peak1_gamma*1j))
                Peaks+= Peak2_amp*(cmath.exp(Peak2_phi*1j)/(x-Peak2_omega+Peak2_gamma*1j))
                Peaks+= Peak3_amp*(cmath.exp(Peak3_phi*1j)/(x-Peak3_omega+Peak3_gamma*1j))
                Peaks+= Peak4_amp*(cmath.exp(Peak4_phi*1j)/(x-Peak4_omega+Peak4_gamma*1j))
                Peaks+= Peak5_amp*(cmath.exp(Peak5_phi*1j)/(x-Peak5_omega+Peak5_gamma*1j))
                Peaks+= Peak6_amp*(cmath.exp(Peak6_phi*1j)/(x-Peak6_omega+Peak6_gamma*1j))
                Peaks+= Peak7_amp*(cmath.exp(Peak7_phi*1j)/(x-Peak7_omega+Peak7_gamma*1j))
                return np.real(Peaks*np.conjugate(Peaks))
        elif len(ModelString) == 9 :
            def SFGFunction(x,NonRes_amp,
                            Peak1_amp,Peak1_phi,Peak1_omega,Peak1_gamma,
                            Peak2_amp,Peak2_phi,Peak2_omega,Peak2_gamma,
                            Peak3_amp,Peak3_phi,Peak3_omega,Peak3_gamma,
                            Peak4_amp,Peak4_phi,Peak4_omega,Peak4_gamma,
                            Peak5_amp,Peak5_phi,Peak5_omega,Peak5_gamma,
                            Peak6_amp,Peak6_phi,Peak6_omega,Peak6_gamma,
                            Peak7_amp,Peak7_phi,Peak7_omega,Peak7_gamma,
                            Peak8_amp,Peak8_phi,Peak8_omega,Peak8_gamma) :
                Peaks = NonRes_amp
                Peaks+= Peak1_amp*(cmath.exp(Peak1_phi*1j)/(x-Peak1_omega+Peak1_gamma*1j))
                Peaks+= Peak2_amp*(cmath.exp(Peak2_phi*1j)/(x-Peak2_omega+Peak2_gamma*1j))
                Peaks+= Peak3_amp*(cmath.exp(Peak3_phi*1j)/(x-Peak3_omega+Peak3_gamma*1j))
                Peaks+= Peak4_amp*(cmath.exp(Peak4_phi*1j)/(x-Peak4_omega+Peak4_gamma*1j))
                Peaks+= Peak5_amp*(cmath.exp(Peak5_phi*1j)/(x-Peak5_omega+Peak5_gamma*1j))
                Peaks+= Peak6_amp*(cmath.exp(Peak6_phi*1j)/(x-Peak6_omega+Peak6_gamma*1j))
                Peaks+= Peak8_amp*(cmath.exp(Peak8_phi*1j)/(x-Peak8_omega+Peak8_gamma*1j))
                return np.real(Peaks*np.conjugate(Peaks))
        
        FitModel = Model(SFGFunction)
        ModelParameters = FitModel.make_params()
        
        self.FitModel = FitModel
        self.ModelParameters = ModelParameters
    
    def SetParameters(self, Value = None) :
        
        FitInfo = self.FitInfo
        ModelParameters = self.ModelParameters
        
        ParameterList = ['amp','phi','omega','gamma','center','sigma','c']
        Parameters = {'Standard': FitInfo['Models']}

        if 'Cases' in FitInfo and Value != None:
            for Case in FitInfo['Cases'] :
                if Value >= min(FitInfo['Cases'][Case]['zRange']) and Value <= max(FitInfo['Cases'][Case]['zRange']) :
                    Parameters[Case] = FitInfo['Cases'][Case]
        
        for Dictionary in Parameters :
            for Peak in Parameters[Dictionary] :
                for Parameter in Parameters[Dictionary][Peak] :
                    if Parameter in ParameterList :
                        for Key in Parameters[Dictionary][Peak][Parameter] :
                            if Key != 'set' :
                                exec('ModelParameters["'+Peak+'_'+Parameter+'"].'+Key+'='+str(Parameters[Dictionary][Peak][Parameter][Key]))
                            else :
                                exec('ModelParameters["'+Peak+'_'+Parameter+'"].'+Key+str(Parameters[Dictionary][Peak][Parameter][Key]))
        
        self.ModelParameters = ModelParameters
    
    def Fit(self,**kwargs) :
        
        for kwarg in kwargs :
            if kwarg == 'fit_x':
                fit_x = kwargs[kwarg]
        
        Data = self.Data
        Name = self.Name
        FitModel = self.FitModel
        ModelParameters = self.ModelParameters
        FitInfo = self.FitInfo
        
        if 'xRange' in FitInfo :
            Data = dt.trimData(Data,FitInfo['xRange'][0],FitInfo['xRange'][1])
        x = Data.index.values
        try:
            fit_x
        except :
            try :
                NumberPoints
            except :
                fit_x = x
            else :
                for i in NumberPoints :
                    fit_x[i] = min(x) + i * (max(x) - min(x)) / (Numberpoints - 1)
        
        Fits = df(index=fit_x,columns=Data.columns.values)
        FitsParameters = df(index=ModelParameters.keys(),columns=Data.columns.values)
        FitsResults = list()
        FitsComponents = list()
        
        for idx,Column in enumerate(Data) :
            
            self.SetParameters(Column)
            
            y = Data[Column].values
            FitResults = FitModel.fit(y, ModelParameters, x=x, nan_policy='omit')
            fit_comps = FitResults.eval_components(FitResults.params, x=fit_x)
            fit_y = FitResults.eval(x=fit_x)
            ParameterNames = [i for i in FitResults.params.keys()]
            for Parameter in (ParameterNames) :
                FitsParameters[Column][Parameter] = FitResults.params[Parameter].value
            Fits[Column] = fit_y
            FitsResults.append(FitResults)
            FitsComponents.append(fit_comps)
            
            sys.stdout.write(("\rFitting %i out of "+str(Data.shape[1])) % (idx+1))
            sys.stdout.flush()
        
        self.Fits = Fits
        self.FitsParameters = FitsParameters
        self.FitsResults = FitsResults
        self.FitsComponents = FitsComponents
    
    def ShowFits(self,xLabel='',yLabel='') :
        
        Data = self.Data
        Fits = self.Fits
        FitInfo = self.FitInfo
        
        FitsParameters = self.FitsParameters
        FitsComponents = self.FitsComponents
        
        for idx,Column in enumerate(Data) :
            
            plt.figure(figsize = [6,4])
            plt.plot(Data.index, Data[Column],'k.', label='Data')
            plt.plot(Fits.index, Fits[Column], 'r-', label='Fit')
            for Component in FitsComponents[idx] :
                if not isinstance(FitsComponents[idx][Component],float) :
                    plt.fill(Fits.index, FitsComponents[idx][Component], '--', label=Component, alpha=0.5)
            plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(1.2, 1), ncol=1)
            plt.xlabel(xLabel), plt.ylabel(yLabel)
            if 'xRange' in FitInfo :
                plt.xlim(FitInfo['xRange'][0],FitInfo['xRange'][1])
            plt.title(str(Column))
            plt.show()
            
            Peaks = list()
            for Parameter in FitsParameters.index :
                Name = Parameter.split('_')[0]
                if Name not in Peaks :
                    Peaks.append(Name)

            string = ''
            for Peak in Peaks :
                string = string + Peak + ' | '
                for Parameter in FitsParameters.index :
                    if Peak == Parameter.split('_')[0] : 
                        string = string + Parameter.split('_')[1] + ': ' + str(round(FitsParameters[Column][Parameter],2))
                        string = string + ', '
                string = string[:-2] + '\n'
            print(string)
            print(75*'_')

class sfg :
    
    def __init__(self) :
        
        with open('parameters.yaml', 'r') as stream :
            self.folders = yaml.safe_load(stream)['folders']
        
    def LoadData(self, File) :
        
        def FolderPath(File) :
            date = re.split(r'sfg|_',File)[1]
            return self.folders['data']+'/'+'20'+date[0:2]+'/'+'20'+date[0:2]+'.'+date[2:4]+'.'+date[4:6]

        with open(File, 'r') as stream :
            Info = yaml.safe_load(stream)
        
        if 'FolderPath' not in Info :
            Info['FolderPath'] = FolderPath(File)
        
        Data = dt.loadSFG(Info)
        Threshold = Info['Background']['Threshold']
        Data = dt.removeEmptyDataSets(Data,Threshold)
        
        if 'Files' in Info['Background'] :
            xlist = list()
            ylist = list()
            for file in Info['Background']['Files'] :
                print('Loading background file: '+Info['Background']['Files'][file]['FileName'])
                if 'FolderPath' not in Info['Background']['Files'][file] :
                    Info['Background']['Files'][file]['FolderPath'] = FolderPath(Info['Background']['Files'][file]['FileName'])
                TempData = dt.loadSFG(Info['Background']['Files'][file])
                xlist.append(TempData.index.values)
                ylist.append(np.transpose(TempData.values)[0])
            BackgroundFromFile = np.array((np.average(xlist,axis=0),np.average(ylist,axis=0)))
            self.BackgroundFromFile = BackgroundFromFile
        
        DataName = Path(File)
        DataName = DataName.with_suffix('')
        
        self.ParametersFile = File
        self.Info = Info
        self.Data = Data
        self.DataName = DataName
    
    def FitData(self) :
        
        Data = self.Data
        Info = self.Info
        DataName = str(self.DataName)
        
        print('Data: '+DataName)
        print('Description: '+Info['Description'])
        
        ##### Prepare Data #####
        
        if 'Files' in Info['Background'] :
            BackgroundFromFile = self.BackgroundFromFile
            for column in Data :
                Data[column] = Data[column]-BackgroundFromFile[1]
        if 'Level' in Info['Background'] :
            xRange = Info['Background']['Level']['xRange']
            Mean = Data
            Mean = Mean[Mean.index>min(xRange)]
            Mean = Mean[Mean.index<max(xRange)]
            Mean = Mean.mean().mean()
            Data -= Mean
        
        TBackground = Info['Background']['zRange']
        DataNames = list()
        for i in Data.columns :
            if i >= min(TBackground) and i <= max(TBackground) :
                DataNames.append(i)
        Background = df(Data[DataNames].mean(axis=1),columns=['Data'])
        
        Resolution = Info['Resolution']
        Data = dt.reduceResolution(Data,Resolution)
        
        ##### Fit Data #####

        try :
            Info['Background']['Models']
        except :
            Data_BC = Data.divide(Background['Data'],axis=0)
        else :
            print('Fitting Background')
            fit = fitTools(Background,Info['Background'],'Background')
            fit.Fit()
            Background['Fit'] = fit.Fits['Data']
            Data_BC = Data.divide(Background['Fit'],axis=0)
        
        if 'xRange' in Info['Fit'] :
            Data_BC = dt.trimData(Data_BC,Info['Fit']['xRange'][0],Info['Fit']['xRange'][1])
        
        if 'zRange' in Info['Fit'] :
            T_mask = []
            T_mask.append(Data.columns<=max(Info['Fit']['zRange']))
            T_mask.append(Data.columns>=min(Info['Fit']['zRange']))
            T_mask = np.all(T_mask, axis=0)
            Data_BC = Data_BC.T[T_mask].T
        
        fit = fitTools(Data_BC,Info['Fit'])
        fit.Fit(fit_x=Data.index.values)

        Fits_BC = fit.Fits
        FitsParameters = fit.FitsParameters
        
        if 'Fit' in Background :
            Fits = Fits_BC.multiply(Background['Fit'],axis=0)
        else :
            Fits = Fits_BC.multiply(Background['Data'],axis=0)
        
        print('\n'+100*'_')
        
        ##### Peak Assignments #####
        
        PeakList = list()
        AssignmentList = list()
        for Peak in Info['Fit']['Models'] :
            PeakList.append(Peak)
            if 'assignment' in Info['Fit']['Models'][Peak] :
                AssignmentList.append(Info['Fit']['Models'][Peak]['assignment'])
            else :
                AssignmentList.append(Peak)
        FitsAssignments = df(AssignmentList,index=PeakList,columns=['Assignment'])
        
        ##### Show Fits & Data #####
        
        if 'ShowFits' in Info['Fit'] :
            ShowFits = Info['Fit']['ShowFits']
        else :
            ShowFits = True

        if ShowFits :
            
            plt.figure(figsize = [6,4])
            plt.plot(Background.index, Background['Data'],'k.', label='Data')
            if 'Fit' in Background :
                plt.plot(Background.index, Background['Fit'], 'r-', label='Fit')
            plt.xlabel('WaveNumber (cm$^{-1}$)'), plt.ylabel('Intensity (au)')
            plt.title('Background')
            plt.show()

            print(100*'_')
        
            for Column in Data_BC :

                plt.figure(figsize = [12,4])

                plt.subplot(1, 2, 1)
                plt.plot(Data.index, Data[Column],'k.', label='Data')
                plt.plot(Fits.index, Fits[Column], 'r-', label='Fit')
                plt.xlabel('WaveNumber (cm$^{-1}$)'), plt.ylabel('Intensity (au)')
                plt.title('Temperature: '+str(Column)+' K')

                plt.subplot(1, 2, 2)
                plt.plot(Data_BC.index, Data_BC[Column],'k.', label='Data')
                plt.plot(Fits_BC.index, Fits_BC[Column], 'r-', label='Fit')
                plt.xlabel('WaveNumber (cm$^{-1}$)'), plt.ylabel('Intensity (au)')
                if 'xRange' in Info['Fit'] :
                    plt.xlim(Info['Fit']['xRange'][0],Info['Fit']['xRange'][1])

                plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(1.2, 1), ncol=1)
                plt.show()

                Peaks = list()
                for Parameter in FitsParameters.index :
                    Name = Parameter.split('_')[0]
                    if Name not in Peaks :
                        Peaks.append(Name)

                string = ''
                for Peak in Peaks :
                    if 'assignment' in Info['Fit']['Models'][Peak] :
                        string += Info['Fit']['Models'][Peak]['assignment'] + ' | '
                    else :
                        string += Peak + ' | '
                    for Parameter in FitsParameters.index :
                        if Peak == Parameter.split('_')[0] : 
                            string += Parameter.split('_')[1] + ': ' + str(round(FitsParameters[Column][Parameter],2))
                            string += ', '
                    string = string[:-2] + '\n'
                print(string)
                print(100*'_')
        FitsParameters = FitsParameters.T
        FitsParameters = FitsParameters[np.concatenate((FitsParameters.columns.values[1:],FitsParameters.columns.values[0:1]))]
        
        # Plot 2D Data & Fits
        
        plt.figure(figsize = [8,12])
        
        plt.subplot(2, 1, 1)
        x = Data.index.values
        y = Data.columns.values
        z = np.transpose(Data.values)
        plt.ylabel('Temperature (K)', fontsize=16)
        plt.tick_params(axis = 'both', which = 'major', labelsize = 16)
        plt.title('Data: '+DataName, fontsize=16)
        pcm = plt.pcolor(x, y, z, cmap='jet', shading='auto')
        
        plt.subplot(2, 1, 2)
        x = Fits.index.values
        y = Fits.columns.values
        z = np.transpose(Fits.values)
        plt.xlabel('Wavenumber (cm$^-$$^1$)', fontsize=16)
        plt.ylabel('Temperature (K)', fontsize=16)
        plt.tick_params(axis = 'both', which = 'major', labelsize = 16)
        plt.title('Fits: '+DataName, fontsize=16)
        pcm = plt.pcolor(x, y, z, cmap='jet', shading='auto')
        
        plt.show()
        
        # Plot Trends
        
        UniqueParameters = []
        [UniqueParameters.append(x.split('_')[1]) for x in FitsParameters.columns if x.split('_')[1] not in UniqueParameters][0]
        for uniqueParameter in UniqueParameters :
            fig = go.Figure()
            for parameter in FitsParameters :
                if uniqueParameter in parameter :
                    Name = parameter.split('_')[0]
                    if 'assignment' in Info['Fit']['Models'][Name] :
                        Name = Info['Fit']['Models'][Name]['assignment']
                    fig.add_trace(go.Scatter(x=FitsParameters.index,y=FitsParameters[parameter],name=Name,mode='lines+markers'))
            fig.update_layout(xaxis_title='Temperature (K)',yaxis_title=uniqueParameter,title=DataName,legend_title='',width=800,height=400)
            fig.show()
        
        ##### Store Fits ####
        
        self.Fits = Fits
        self.FitsData = Data
        self.FitsBackground = Background
        self.FitsParameters = FitsParameters
        self.FitsAssignments = FitsAssignments
        
        ##### Widgets #####

        def CopyData_Clicked(b) :
            Data.to_clipboard()
        CopyData = widgets.Button(description="Copy Data")
        CopyData.on_click(CopyData_Clicked)

        def CopyFits_Clicked(b) :
            Fits.to_clipboard()
        CopyFits = widgets.Button(description="Copy Fits")
        CopyFits.on_click(CopyFits_Clicked)

        def CopyParameters_Clicked(b) :
            FitsParameters.to_clipboard()
        CopyParameters = widgets.Button(description="Copy Parameters")
        CopyParameters.on_click(CopyParameters_Clicked)

        def Save2File_Clicked(b) :
            os.makedirs(self.folders['Fits'], exist_ok=True)
            FitsFile = self.folders['Fits'] +'/' + DataName + '.hdf'
            Data.to_hdf(FitsFile,'Data')
            Fits.to_hdf(FitsFile,'Fits',mode='a')
            Data_BC.to_hdf(FitsFile,'Data_BC',mode='a')
            Fits_BC.to_hdf(FitsFile,'Fits_BC',mode='a')
            FitsParameters.to_hdf(FitsFile,'Fits_Parameters',mode='a')
            FitsAssignments.to_hdf(FitsFile,'Fits_Assignments',mode='a')
        Save2File = widgets.Button(description="Save to File")
        Save2File.on_click(Save2File_Clicked)

        display(widgets.Box([CopyData,CopyFits,CopyParameters,Save2File]))
    
    def UI(self) :
        
        out = widgets.Output()
        
        self.ParametersFiles = widgets.Dropdown(
            options=dt.fileList(['.yaml','sfg']),
            description='Select File',
            layout=Layout(width='70%'),
            style = {'description_width': '150px'},
            disabled=False,
        )

        def ShowData_Clicked(b) :
            with out :
                clear_output(True)
                self.LoadData(self.ParametersFiles.value+'.yaml')
                plt.figure(figsize = [8,6])
                x = self.Data.index.values
                y = self.Data.columns.values
                z = np.transpose(self.Data.values)
                plt.xlabel('Wavenumber (cm$^-$$^1$)', fontsize=16)
                plt.ylabel('Temperature (K)', fontsize=16)
                plt.tick_params(axis = 'both', which = 'major', labelsize = 16)
                plt.title(self.DataName, fontsize=16)
                pcm = plt.pcolor(x, y, z, cmap='jet', shading='auto')
                plt.show()
        ShowData = widgets.Button(description="Show Data")
        ShowData.on_click(ShowData_Clicked)
        
        def FitData_Clicked(b) :
            with out :
                clear_output(True)
                self.LoadData(self.ParametersFiles.value+'.yaml')
                self.FitData()
        FitData = widgets.Button(description="Fit Data")
        FitData.on_click(FitData_Clicked)
        
        display(self.ParametersFiles)
        display(widgets.HBox([ShowData,FitData]))
        display(out)