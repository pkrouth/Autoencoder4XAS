from ..utils.imports import *
class XasDataset(Dataset):
    """ Custom XAS Dataset"""
    def __init__(self,json_file, root_dir, descriptor = 'CN', transform=None):
        self.XAS_Frame = pd.read_json(json_file, orient='records')
        #self.XAS_Frame.Abs = self.XAS_Frame.Abs.apply(lambda x: x/np.max(x)) ## Questionable Normalization
        self.root_dir = root_dir
        self.transform = transform
        self.descriptor = descriptor
        ## Need Improvement with possible One-Hot Encoding
        if self.descriptor == 'Crystal':
            self.XAS_Frame['Crystal'] = (self.XAS_Frame.Crystal.apply(lambda x: x[0]).astype('category'))
            self.categories = self.XAS_Frame.Crystal.cat.categories
            self.XAS_Frame.Crystal = self.XAS_Frame.Crystal.cat.codes
        
    def __len__(self):
        return len(self.XAS_Frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if type(idx)==int:
            idx=[idx]
        data_Energy = self.XAS_Frame['Energy'].iloc[idx]
        data_Abs    = self.XAS_Frame['Abs'].iloc[idx]
        data_Y      = self.XAS_Frame[self.descriptor].iloc[idx]#.to_numpy()
        data_Y      = np.vstack(data_Y)#.reshape(-1,2) #Not sure why do I need to reshape
        sample      = pd.Series({'Energy': np.vstack(data_Energy), 'Abs': np.vstack(data_Abs), 'Descriptor': data_Y})
        #print(sample)
        if self.transform:
            sample = self.transform(sample)
        return sample
        

class XDataset(Dataset):
    """ Custom XAS Dataset"""
    def __init__(self,json_file, root_dir, transform=None):
        self.dataFrame = pd.read_json(json_file, orient='records')
        self.X_Frame = np.array(self.dataFrame['nonlinear'][0])
        self.Y_Frame = np.array(self.dataFrame['nonlinear'][1])
        #self.XAS_Frame.Abs = self.XAS_Frame.Abs.apply(lambda x: x/np.max(x)) ## Questionable Normalization
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.X_Frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if type(idx)==int:
            idx=[idx]
        data_X = self.X_Frame[idx]
        data_Y = self.Y_Frame[idx]
        data_Y      = np.vstack(data_Y).reshape(-1,1)
        sample      = pd.Series({'X': np.vstack(data_X), 'Y': data_Y})
        #print(sample)
        if self.transform:
            sample = self.transform(sample)
        return sample
    
class XasMultiTaskDataset(Dataset):
    """ Custom XAS Dataset
        v2: But only Crystal apply operation removed.
    """
    def __init__(self,json_file, root_dir, descriptor = 'CN', transform=None, MTL=True, cat_list=None):
        self.XAS_Frame = pd.read_json(json_file, orient='records')
        #self.XAS_Frame.Abs = self.XAS_Frame.Abs.apply(lambda x: x/np.max(x)) ## Questionable Normalization
        self.root_dir = root_dir
        self.transform = transform
        #if type(descriptor)==type(''):
        self.descriptor = descriptor
        ## Need Improvement with possible One-Hot Encoding
        self.MTL = MTL
        if cat_list is None:
            self.XAS_Frame['Crystal'] = self.XAS_Frame.Crystal.astype('category')
            self.categories = self.XAS_Frame.Crystal.cat.categories
            self.XAS_Frame.Crystal = self.XAS_Frame.Crystal.cat.codes
        else:
            self.categories = cat_list
            cat_dict = dict(zip(self.categories, list(range(len(self.categories)))))
            self.XAS_Frame.Crystal = self.XAS_Frame.Crystal.map(cat_dict)
        self.XAS_Frame['MTL_Label'] = self.XAS_Frame[['CN', 'Distances', 'H_fraction', 'Crystal']].apply(lambda x: x.CN+x.Distances+[x.Crystal, x.H_fraction], axis=1)
        
    def __len__(self):
        return len(self.XAS_Frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if type(idx)==int:
            idx=[idx]
        data_Energy = self.XAS_Frame['Energy'].iloc[idx]
        data_Abs    = self.XAS_Frame['Abs'].iloc[idx]
        if self.MTL:
            data_Y      = self.XAS_Frame['MTL_Label'].iloc[idx]#.to_numpy()
        else:
            data_Y      = self.XAS_Frame[self.descriptor].iloc[idx]#.to_numpy()
        data_Y      = np.vstack(data_Y)#.reshape(-1,2) #Not sure why do I need to reshape
        sample      = pd.Series({'Energy': np.vstack(data_Energy), 'Abs': np.vstack(data_Abs), 'Descriptor': data_Y})
        #print(sample)
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def show(self, idx):
        x, y = self.__getitem__(idx)
        data = pd.Series({'x_axis':np.array(energymesh).reshape(1,-1).squeeze(), 'y_axis':x.numpy().squeeze()})
        print(data)
        _ = sns.lineplot(x='x_axis', y='y_axis', data=data)
        _ = plt.title(f"Sample_#{idx}_{'MTL[CN+Distances+Crystal+H_fraction]:' if self.MTL else self.descriptor}{y}")
        
        
        
        
class XasMultiTaskDataset_v1(Dataset):
    """ Custom XAS Dataset
        v1: Crystal as category operation in here.
    """
    def __init__(self,json_file, root_dir, descriptor = 'CN', transform=None, MTL=True):
        self.XAS_Frame = pd.read_json(json_file, orient='records')
        #self.XAS_Frame.Abs = self.XAS_Frame.Abs.apply(lambda x: x/np.max(x)) ## Questionable Normalization
        self.root_dir = root_dir
        self.transform = transform
        #if type(descriptor)==type(''):
        self.descriptor = descriptor
        ## Need Improvement with possible One-Hot Encoding
        self.MTL = MTL
        self.XAS_Frame['Crystal'] = (self.XAS_Frame.Crystal.apply(lambda x: x[0]).astype('category'))
        self.categories = self.XAS_Frame.Crystal.cat.categories
        self.XAS_Frame.Crystal = self.XAS_Frame.Crystal.cat.codes
        self.XAS_Frame['MTL_Label'] = self.XAS_Frame[['CN', 'Distances', 'H_fraction', 'Crystal']].apply(lambda x: x.CN+x.Distances+[x.Crystal, x.H_fraction], axis=1)
        
    def __len__(self):
        return len(self.XAS_Frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if type(idx)==int:
            idx=[idx]
        data_Energy = self.XAS_Frame['Energy'].iloc[idx]
        data_Abs    = self.XAS_Frame['Abs'].iloc[idx]
        if self.MTL:
            data_Y      = self.XAS_Frame['MTL_Label'].iloc[idx]#.to_numpy()
        else:
            data_Y      = self.XAS_Frame[self.descriptor].iloc[idx]#.to_numpy()
        data_Y      = np.vstack(data_Y)#.reshape(-1,2) #Not sure why do I need to reshape
        sample      = pd.Series({'Energy': np.vstack(data_Energy), 'Abs': np.vstack(data_Abs), 'Descriptor': data_Y})
        #print(sample)
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def show(self, idx):
        x, y = self.__getitem__(idx)
        data = pd.Series({'x_axis':np.array(energymesh).reshape(1,-1).squeeze(), 'y_axis':x.numpy().squeeze()})
        print(data)
        _ = sns.lineplot(x='x_axis', y='y_axis', data=data)
        _ = plt.title(f"Sample_#{idx}_{'MTL[CN+Distances+Crystal+H_fraction]:' if self.MTL else self.descriptor}{y}")