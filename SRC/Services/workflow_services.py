from pathlib import Path 
from Services.dataset_indexer import DatasetIndexer
from Services.eda_service import EDAService

class WorkflowServices:
  def __init__(self,data_dir,output_dir):
    self.data_dir = data_dir
    self.output_dir = Path(output_dir)

    self.indexer = DatasetIndexer(data_dir)

    self.dataframe = None
    self.eda_service = None

    # Load and index dataset
    def load_dataset(self):

        self.dataframe = self.indexer.build_dataframe()

        self.eda_service = EDAService(
            dataframe=self.dataframe,
            output_dir=self.output_dir
        )

        return self.dataframe

    # Generate all EDA visualisations
    def generate_eda(self):

        if self.eda_service is None:
            raise ValueError("Dataset not loaded.")

        self.eda_service.plot_class_distribution()
        self.eda_service.plot_image_sizes()

    # Return dataset summary
    def get_summary(self):

        if self.eda_service is None:
            raise ValueError("Dataset not loaded.")

        return self.eda_service.build_summary()

    # Return dataset warnings
    def get_warnings(self):

        if self.eda_service is None:
            raise ValueError("Dataset not loaded.")

        return self.eda_service.generate_dataset_warnings()
  
