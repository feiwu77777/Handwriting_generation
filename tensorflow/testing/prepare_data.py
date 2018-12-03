import numpy as np
##################  Part 1: Preparing data ##################

T = 600

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
    strokes_raw = np.load('strokes.npy',encoding = 'latin1')
    strokes, norm_params = normalize(strokes_raw)
    with open('sentences.txt') as f:
        texts = f.readlines()
    strokes,texts = filter_data(strokes,texts)
    strokes = preprocess_strokes(strokes)
    Y = np.zeros(strokes.shape,dtype ="float32")
    for i in range(Y.shape[0]):
        Y[i,:T-1,:] = strokes[i,1:T,:]
    C_vec,dic,cut,texts = preprocess_text(texts)
    return strokes,Y,C_vec,dic,cut,texts,norm_params

def normalize(strokes):
    tot = 0
    for i in range(6000):
        tot += strokes[i].shape[0]
        
    ha = np.zeros((tot,2))
    ind = 0
    for i in range(6000):
        leng = strokes[i].shape[0]
        ha[ind:ind+leng] = strokes[i][:,1:]
        ind = ind + leng
        
    mean1 = np.mean(ha[:,0])
    mean2 = np.mean(ha[:,1])
    std1 = np.std(ha[:,0])
    std2 = np.std(ha[:,1])
    
    for i in range(tot):
        ha[i,0] = (ha[i,0] - mean1)/std1
        ha[i,1] = (ha[i,1] - mean2)/std2
    
    ind = 0
    for i in range(6000):
        stroke = strokes[i]
        strokes[i][:,1:] = ha[ind:ind+stroke.shape[0]]
        ind = ind + stroke.shape[0]
    return strokes,[mean1,mean2,std1,std2]
        
def denormarlize(strokes,params):
    for i in range(strokes.shape[0]):
        strokes[i,1] = strokes[i,1]*params[2] + params[0]
        strokes[i,2] = strokes[i,2]*params[3] + params[1]
    return strokes
    
X,Y,C,dic,cut,texts,norm_params = load_data()      


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
