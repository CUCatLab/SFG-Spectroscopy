import os
import numpy as np
from pandas import DataFrame as df
import re
from . import sif_open as sif


def fileList(Filter='') :
    
    FileList = [f for f in os.listdir()]
    for i in range(len(Filter)):
        FileList = [k for k in FileList if Filter[i] in k]
    for i in range(len(FileList)):
        FileList[i] = FileList[i].replace('.yaml','')
    
    return FileList

def loadSFG(Parameters) :
    
    FolderPath = Parameters['FolderPath']
    FileName = Parameters['FileName']

    if FileName.endswith('.ibw') :
        d = binarywave.load(FolderPath + '/' + FileName)
        y = np.transpose(d['wave']['wData'])
        Start = d['wave']['wave_header']['sfB']
        Delta = d['wave']['wave_header']['sfA']
        x = np.arange(Start[0],Start[0]+y.shape[1]*Delta[0]-Delta[0]/2,Delta[0])
        z = np.arange(Start[1],Start[1]+y.shape[0]*Delta[1]-Delta[1]/2,Delta[1])
        print('Igor binary data loaded')
    elif FileName.endswith('.itx') :
        y = np.loadtxt(FolderPath + '/' + FileName,comments =list(string.ascii_uppercase))
        y = np.transpose(y)
        with open(FolderPath + '/' + FileName) as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if 'SetScale/P' in row[0]:
                    SetScale = row[0]
        xScale = re.findall(r' x (.*?),"",', SetScale)
        xScale = xScale[0].split(',')
        zScale = re.findall(r' y (.*?),"",', SetScale)
        zScale = zScale[0].split(',')
        Start = [float(xScale[0]),float(zScale[0])]
        Delta = [float(xScale[1]),float(zScale[1])]
        x = np.arange(Start[0],Start[0]+y.shape[1]*Delta[0]-Delta[0]/2,Delta[0])
        z = np.arange(Start[1],Start[1]+y.shape[0]*Delta[1]-Delta[1]/2,Delta[1])
        print('Igor text data loaded')
    elif FileName.endswith('sif') :
        FileData = sif.xr_open(FolderPath + '/' + FileName)
        y = FileData.values[:,0,:]
        x = [i for i in range(len(np.transpose(y)))]
        z = [i+1 for i in range(len(y))]
        
        try :
            FileData.attrs['WavelengthCalibration0']
            FileData.attrs['WavelengthCalibration1']
            FileData.attrs['WavelengthCalibration2']
            FileData.attrs['WavelengthCalibration3']
        except :
            print('Warning: Wavelength calibration not found')
        else :
            c0 = FileData.attrs['WavelengthCalibration0']
            c1 = FileData.attrs['WavelengthCalibration1']
            c2 = FileData.attrs['WavelengthCalibration2']
            c3 = FileData.attrs['WavelengthCalibration3']
            for i in x :
                x[i] = c0 + c1*i + c2*1**2 + c3*i**3
            x = np.array(x)
            x = 1e7 / x - 12500
        
        try :
            Frame = Parameters['Heating']['Frame']
            Temperature = Parameters['Heating']['Temperature']
        except :
            print('Warning: Temperature data not found')
        else :
            FitModel = QuadraticModel()
            ModelParameters = FitModel.make_params()
            FitResults = FitModel.fit(Temperature, ModelParameters,x=Frame)
            idx = np.array(z)
            z = FitResults.eval(x=idx)
            z = np.round(z,1)
    
    Data = df(np.transpose(y),index=x,columns=z)
        
    return Data

def removeEmptyDataSets(Data,Threshold) :
    
    Index = list()
    for i in Data.columns :
        if np.mean(Data[i]) < Threshold :
            Index.append(i)
    for i in Index :
        del Data[i]
    
    return Data

def trimData(Data,Min,Max) :
    
    Mask = np.all([Data.index.values>Min,Data.index.values<Max],axis=0)
    Data = Data[Mask]
    
    return Data

def reduceResolution(Data,Resolution=1) :
    
    Counter = 0
    ReducedData = df()
    for i in range(int(len(Data.columns.values)/Resolution)) :
        Column = round(np.mean(Data.columns[Counter:Counter+Resolution]),1)
        ReducedData[Column] = Data[Data.columns[Counter:Counter+Resolution]].mean(axis=1)
        Counter = Counter + Resolution
    
    return ReducedData