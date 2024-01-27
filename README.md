# HyperFusion
**Presenting initial findings with ongoing work. Updates will follow as progress continues.**
## Intro
The integration of multi-modal data has emerged as a promising approach in various fields, enabling a more comprehensive understanding of 
complex phenomena by leveraging the complementary information from different sources. In the realm of medical research, the integration of 
multi-modal data has emerged as a powerful approach for enhancing our understanding of complex diseases and conditions. The fusion of different 
data types, such as tabular data (electronic health records - EHR) encompassing medical records and demographic information, together with 
high-resolution imaging modalities like MRI scans, has unlocked new avenues for comprehensive analysis and diagnosis.

In this work, we propose a novel approach that harnesses the power of hypernetworks to fuse tabular data and MRI brain scans.

## Graphical Abstract
<img src="https://github.com/daniel4725/HyperNetworks4imgNtabular/assets/95569050/dc8e21de-7c0e-42a2-8489-24baa47bc59c" width=90% height=90%>


## Hyper Networks
Training a network, $\mathcal{F}$, to create the weights, $𝜃_\mathcal{H}$, of the main network, $\mathcal{P}_𝜃$. 


<img src="https://github.com/daniel4725/HyperNetworks4imgNtabular/assets/95569050/2415287c-09fb-4532-8ea1-58b63d39fa37" width=35% height=35%>


We use the tabular information as an input to the Hypernetwork ($T$) and the Primary network is an image processing CNN:


<img src="https://github.com/daniel4725/HyperNetworks4imgNtabular/assets/95569050/213185b3-9d12-481c-8b0b-faf34f782408" width=40% height=40%>

## Demonstrating our methodology
We demonstrate the versatility and efficacy of the proposed hypernetwork framework, named HyperFusion, through two distinct brain MRI analysis tasks: brain age prediction conditioned by the subject's sex and classification of subjects into Alzheimer's disease (AD), Mild Cognitive Impairment (MCI), and Cognitively Normal (CN) groups conditioned by their tabular data, which includes clinical measurements, as well as demographic and genetic information.

## The Data
The ADNI dataset, ADNI 1, ADNI 2 and ADNI GO, baseline visits       
ADNI aims to standardize the data collection methods and promote the use of it for research to accelerate discoveries in the disease

![image](https://github.com/daniel4725/HyperNetworks4imgNtabular/assets/95569050/a215529a-c706-4deb-802a-11121123ebaa)


2120 MRI scans - healthy, MCI and AD patients (34%, 48%, 17%)

5 folds (~420 samples each)– one for testing and 4 for cross-validation (same distribution of labels)

The Tabular features used (9):
- demographic: Age, Sex, Education (years)
- genetic risk factor: ApoE4
- cerebrospinal fluid biomarkers : Abeta42, P-tau181, T-tau
- measures derived from PET scan: 18 F-fluorodeoxyglucose (FDG) florbetapir (AV)

## The Architectures
### Brain Age Prediction conditioned by sex
<img src="https://github.com/daniel4725/HyperNetworks4imgNtabular/assets/95569050/4a2669ee-e503-406d-b7a3-f7fe14bf2fb9" width=40% height=40%>

### AD classification
<img src="https://github.com/daniel4725/HyperNetworks4imgNtabular/assets/95569050/c5f083ca-0b71-41fe-801a-01226a22fbb9" width=40% height=40%>

## Results

### Brain Age Prediction conditioned by sex
<img src="https://github.com/daniel4725/HyperNetworks4imgNtabular/assets/95569050/c5d7fdd4-4145-4fa9-8372-09e493481535" width=30% height=30%>


### AD classification
<img src="https://github.com/daniel4725/HyperNetworks4imgNtabular/assets/95569050/75b0d4c5-cd37-4491-80a7-0e0727d8b068" width=70% height=70%>





