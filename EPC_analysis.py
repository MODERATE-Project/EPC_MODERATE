import pandas as pd
from utils import  NN_model

# Generate NN model for different datasets
# READ DATASET 
df_model1_F1=pd.read_csv("data/dEPC_all_builsings_Lombardia.csv", header=0, index_col=0)
dataset_A=pd.read_csv("EPC_Git/data/synt_data_A.csv", header=0, index_col=0)
synthetic_data=pd.read_csv("EPC_Git/data/synt_data_A.csv", header=0, index_col=0)
datasetA_synth = pd.concat([dataset_A, synthetic_data[:300000]])


# GENERATE MODEL to predict ETH (Thermal Energy)
if __name__ == "__main__":
  sim_ = NN_model(
      dataset= dataset_A,
      y_output_name= 'ETH', 
      save_model= False, 
      model_name= 'dnn_model.keras',
      n_predictions= 200 
  )


