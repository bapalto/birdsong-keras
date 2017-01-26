import pandas as pd
import numpy as np
	
def appendDfToPickle(df, filePath):
    import os
    import pandas as pd
    if not os.path.isfile(filePath):
        df.to_pickle(filePath)
    else:
        tempDF=pd.read_pickle(filePath)
        tempDF=tempDF.append(df, ignore_index = True)
        tempDF.to_pickle(filePath)
        
def appendDfToExcel(df, excelFilePath):
    import os
    if not os.path.isfile(excelFilePath):
        df.to_excel(excelFilePath, index=False)
    else:
        tempDF=pd.read_excel(excelFilePath)
        tempDF=tempDF.append(df)
        tempDF.to_excel(excelFilePath, index=False)

def appendDfToCSV(df, csvFilePath):
    import os
    if not os.path.isfile(csvFilePath):
        df.to_csv(csvFilePath, index=False, sep='\t')
    else:
        tempDF=pd.read_csv(csvFilePath, sep='\t')
        tempDF=tempDF.append(df)
        tempDF.to_csv(csvFilePath, index=False, sep='\t')

def modelToDict(model):
    layerNumber=0;
    d=dict()
    for layer in model.layers:
        columnHeader="zz_layer({:02d})".format(layerNumber) #zz: to put it to the last columns
        layerNumber+=1;
        d[columnHeader]=str(layerToDict(layer))
    
    #for k,v in model.get_config().iteritems(): ##not working since keras 1.0.0
    #    if k in ['loss', 'optimizer']:
    #        d[k]=str(v)
    d['optimizer']=str(model.optimizer.get_config())
    d['loss']=str(model.loss)
    d['output_dim'] = model.layers[-1].get_config()['output_dim']
    return d

def layerToDict(layer):
    d=dict()
    for k,v in layer.get_config().iteritems():
        if k in [\
        #Basic
        'input_shape', 'init', 'name', 'output_dim', 'activation', 'p', \
        #Convolution1D
        'nb_filter', 'pool_length', 'filter_length', 'border_mode', \
        #Convolution2D
        'nb_row','nb_col',  'subsample', 'pool_size']:
            d[k]=v  
    d['output_shape']=layer.output_shape  
    return d
    
def resultToDict(result):
    d=dict()
    d['epochs']=len(result.epoch)
    minIsGood=['loss', 'val_loss']
    maxIsGood=['acc', 'val_acc']
    for k,v in result.history.iteritems():
        if k in minIsGood:
            d['final_' + k]=v[-1]
            d['best_' + k]=np.min(v)
            d['best_' + k + '_epoch']=np.argmin(v) + 1 #!!!!!
        elif k in maxIsGood:
            d['final_' + k]=v[-1]
            d['best_' + k]=np.max(v)
            d['best_' + k + '_epoch']=np.argmax(v) + 1 #!!!!!
    return d

def logToDataFrame(model=None, fitting_result=None, otherDict=None):
    d=dict()
    if model is not None:
        d.update(modelToDict(model))
    if fitting_result is not None:
        d.update(resultToDict(fitting_result))
    if otherDict is not None:
        d.update(otherDict)
    return pd.DataFrame(d, index=[0])
    
def logToXLS(filepath, model=None, fitting_result=None, otherDict=None):
    appendDfToExcel(logToDataFrame(model, fitting_result, otherDict), filepath)

def logToCSV(filepath, model=None, fitting_result=None, otherDict=None):
    appendDfToCSV(logToDataFrame(model, fitting_result, otherDict), filepath)