import numpy as np

##################  Part 1: Preparing data ##################

T = 600

def cut_same(strokes):    ### Fuse two consecutive same stroke
    L = []
    for i in range(strokes.shape[0]):
        stroke = strokes[i]
        array = []
        array.append(stroke[0,:])
        for j in range(1,stroke.shape[0]):
            if sum(stroke[j,:] == np.zeros((3,))) == 3:
                pass
            else:
                array.append(stroke[j,:])
        array = np.array(array)
        L.append(array)
    return L


def filter_data(strokes,texts):
    indices = []
    for i in range(len(strokes)):
        if strokes[i].shape[0] <= T:
            indices.append(i)
    texts = np.array(texts)[indices]
    strokes = np.array(strokes)[indices]
    return strokes,texts


def preprocess_strokes(strokes):
    m = np.zeros((strokes.shape[0],T,3),dtype = "float32")
    for i in range(len(strokes)):
        m[i,:strokes[i].shape[0],:] = strokes[i]
    return m


def preprocess_text(texts):
    for i in range(len(texts)):
        texts[i] = texts[i][:len(texts[i])-1]
    unique = []   
    for i in range(len(texts)):
        unique.append(list(set(texts[i])))
        
    unique = np.concatenate(unique,axis = 0)
    unique = list(set(unique))
    
    cut = ["!",".","0","8","2","4","+","#",")",'"',";",":","3","/",",","(","9","5","1","2","6","7","-","'","?"]
        
    unique = [o for o in unique if o not in cut]
                
    texts = np.array(texts)
    
    dictionnary = {character:index for index, character in enumerate(unique)}
    
    maxs = len(texts[0])
    for i in range(1,len(texts)):
        if len(texts[i]) > maxs:
            maxs = len(texts[i])
    
    C_vec = np.zeros((texts.shape[0],maxs,len(unique)+1),dtype = "float32")
    
    for i in range(len(texts)):
        for j in range(len(texts[i])):
            if texts[i][j] in cut:
                C_vec[i,j,len(unique)] = 1
            else:
                C_vec[i,j,dictionnary[texts[i][j]]] = 1
            
    return C_vec, dictionnary, cut,texts


def load_data():
    strokes = np.load('strokes.npy',encoding = 'latin1')
    with open('sentences.txt') as f:
        texts = f.readlines()
    strokes = cut_same(strokes)
    strokes,texts = filter_data(strokes,texts)
    strokes = preprocess_strokes(strokes)
    Y = np.zeros(strokes.shape,dtype ="float32")
    for i in range(Y.shape[0]):
        Y[i,:T-1,:] = strokes[i,1:T,:]
    C_vec,dic,cut,texts = preprocess_text(texts)
    return strokes,Y,C_vec,dic,cut,texts


X,Y,C,dic,cut,texts = load_data()      #X contains only 357 examples for the purpose of testing the code


U = C.shape[1]
character_number = C.shape[2]
cells_number = 400 
batch_size = 16
K = 10
M = 20



def preprocess_sentence(sentence):
    out = np.zeros((len(sentence),len(list(dic.values()))+1))    ### dic maps letter to number, defined in loading data part
    
    for i in range(len(sentence)):
        if sentence[i] in cut:   #### cut contain digit and punctuation ommitted in the count of unique characters
            out[i,len(list(dic.values()))] = 1
        else:
            out[i,dic[sentence[i]]] = 1
    return out