from .imports import *

class Compose(object):
    """
    Composes several transforms together.

    """

    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
    
class Normalize(object):
    def __call__(self, sample):
        max = np.max(sample.X)
        min = np.min(sample.X)
        normalized_X = (sample.X-min)/(max-min)
        assert normalized_X.shape == sample.X.shape
        
        return pd.Series({'X':normalized_X,
               'Y':sample.Y})



class ToTensor(object):
    def __call__(self, sample):
        
  
        return (torch.from_numpy(sample.X), ## numpy.squeeze dropped for Conv input shape match of  
           torch.from_numpy(sample.Y))
    
    def __repr__(self):
        return self.__class__.__name__ + '()'
    
    
    
class XasToTensor(object):
    def __call__(self, sample):
        
  
        return (torch.from_numpy(sample.Abs).float(), ## numpy.squeeze dropped for Conv input shape match of  
           torch.from_numpy(sample.Descriptor).float())
    
    def __repr__(self):
        return self.__class__.__name__ + '()'





         ## Change CN to Descriptor or create and alias for backward compatibility.
class XasInterpolate(object):
    
    def __call__(self, sample, energymesh=np.array(energymesh)):
        #print(energy.shape, absorption.shape)
        #Object needs to be 1D array to interrpolate not 100 by 1, which is 2D.
        
        assert len(energymesh)==100
        assert sample.Energy.shape==sample.Abs.shape
        interpolated_abs = np.array([np.interp(x=energymesh, xp=sample.Energy[i,:], fp=sample.Abs[i,:]) for i in range(sample.Energy.shape[0])])
        #assert interpolated_abs.shape == sample.Abs.shape
        energymesh = energymesh.reshape(1,-1)
        assert interpolated_abs.shape == energymesh.shape
        
        return pd.Series({'Abs':interpolated_abs,
               'Energy': np.repeat(energymesh,sample.Energy.shape[0],axis=0),
               'Descriptor': sample.Descriptor}) ## Change CN to Descriptor or create and alias for backward compatibility.

class XasNormalize(object):
    def __call__(self, sample):
        max = np.max(sample.Abs)
        min = np.min(sample.Abs)
        normalized_Abs = (sample.Abs-min)/(max-min)
        assert normalized_Abs.shape == sample.Abs.shape
        
        return pd.Series({'Abs':normalized_Abs,
               'Energy': sample.Energy,
               'Descriptor': np.squeeze(sample.Descriptor)})
               
    
#    def __init__(self, transforms):
#        self.transforms = transforms
#    
#    def __call__(self, sample):
#        for t in self.transforms:
#            sample = t(sample)
#        return sample
#    
#    def __repr__(self):
#        format_string = self.__class__.__name__ + '('
#        for t in self.transforms:
#            format_string += '\n'
#            format_string += '    {0}'.format(t)
#        format_string += '\n)'
#        return format_string
#    
#class Normalize(object):
#    def __call__(self, sample):
#        
#        normalized_abs = sample.Abs/np.max(sample.Abs)
#        assert normalized_abs.shape == sample.Abs.shape
#        
#        return pd.Series({'Abs':normalized_abs,
#               'Energy': sample.Energy,
#               'CN': sample.CN})

#class Interpolate(object):
#    def __call__(self, sample, energymesh=np.array(energymesh)):
#        #print(energy.shape, absorption.shape)
#        #Object needs to be 1D array to interrpolate not 100 by 1, which is 2D.
#        
#        assert len(energymesh)==100
#        assert sample.Energy.shape==sample.Abs.shape
#        interpolated_abs = np.array([np.interp(x=energymesh, xp=sample.Energy[i,:], fp=sample.Abs[i,:]) for i in range(sample.Energy.shape[0])])
#        assert interpolated_abs.shape == sample.Abs.shape

#        return pd.Series({'Abs':interpolated_abs,
#               'Energy': np.repeat(energymesh.reshape(1,-1),sample.Energy.shape[0],axis=0),
#               'CN': np.squeeze(sample.CN)})

#    
#class Conv3dShape(object): ## Not needed
#    def __call__(self,sample):
#        Abs = sample.Abs
#        a,b = Abs.shape
#        return pd.Series({'Abs':Abs.reshape(a,1,b),
#               'Energy': sample.Energy,
#               'CN': sample.CN})

#class ToTensor(object):
#    def __call__(self, sample):
#        
#  
#        return (torch.from_numpy(sample.Abs), ## numpy.squeeze dropped for Conv input shape match of  
#           torch.from_numpy(sample.CN))
#    
#    def __repr__(self):
#        return self.__class__.__name__ + '()'



#    
    