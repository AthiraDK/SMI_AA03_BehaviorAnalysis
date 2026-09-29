from behavior_fw.core.experiment import BaseExperiment


class OptoClosedLoopExperiment(BaseExperiment):

    def __init__(self, data_path, protocol_path, optostim_path):
        super().__init__(data_path)
        self.df_stimulus = None

    def load_exp_data(self):
        """Loads tracking data, protocol, and frame-by-frame LED status."""
        super().load_exp_data()
        return self