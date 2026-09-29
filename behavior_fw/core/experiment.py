import pandas as pd



class BaseExperiment :

    def __init__(self, data_path):
        # path to the experiment folder
        self.data_path = data_path 
        self.path_rawtracking = None
        self.path_metahighres = None
        

    def load_exp_data(self):
        self.df_track = None
        return self
    
    def get_velocities(self):
        self.df_velocities = None
        return self
    
    def detect_saccades(self, method=None):
        return self
    

