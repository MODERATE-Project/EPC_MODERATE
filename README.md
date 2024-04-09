# EPC_MODERATE

Analysis of EPC (Energy Performance Certificate) with the aim of identifying a model capable of synthesizing data and using it as training for deep learning models for the prediction and/or identification of energy consumption patterns for the building stock. Currently, data from the Lombardy region are being analyzed.

## Structure

The repository is structured as follows:

- **data**: folder with different EPC dataset already processed
- **models**: DNN models and generative models
- **EPC_analysis**: generate NN models of EPC
- **synthetic_model**: generate synthetic model of EPC (filtred for labels A and A+)
- **utils**: functions to be used for the analysis

### Models

4 modles have been realized:
- **A_EPC_NN_model**: DNN model for EPC of class A and A+
- **dnn_model**: DNN model for all EPCs
- **EPC_model_A**: generative model to create synthetic EPC data of classes A and A+
- **EPC_model**: generative model to create syntehtic data of all EPCs