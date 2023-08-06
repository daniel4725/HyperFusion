# HyperNetworks4imgNtabular
**Presenting initial findings with ongoing work. Updates will follow as progress continues.**
## Into
The integration of multi-modal data has emerged as a promising approach in various fields, enabling a more comprehensive understanding of 
complex phenomena by leveraging the complementary information from different sources. In the realm of medical research, the integration of 
multi-modal data has emerged as a powerful approach for enhancing our understanding of complex diseases and conditions. The fusion of different 
data types, such as tabular data (electronic health records - EHR) encompassing medical records and demographic information, together with 
high-resolution imaging modalities like MRI scans, has unlocked new avenues for comprehensive analysis and diagnosis.

In this work, we propose a novel approach that harnesses the power of hypernetworks to fuse tabular data and MRI brain scans.

## Hyper Networks
Training a network, $𝑓(𝑧)$, to create the weights, $𝜃$, of the main network, $𝑔_𝜃 (𝑥)$  .
$𝑜𝑢𝑡𝑝𝑢𝑡=𝑔_{𝑓(𝑧)} (𝑥)$

<img src="https://github.com/daniel4725/HyperNetworks4imgNtabular/assets/95569050/2347479b-e1e3-4f1f-aa21-720f903a5fa3" width=35% height=35%>

We use the tabular information as an input to the Hyper Network (z) and the Primary network is an image processing CNN:

<img src="https://github.com/daniel4725/HyperNetworks4imgNtabular/assets/95569050/f3fec655-9549-4c7f-87d9-0af6dc0a59d1" width=80% height=80%>

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

## The Architecture
<img src="https://github.com/daniel4725/HyperNetworks4imgNtabular/assets/95569050/1b38e0f6-13b9-42d5-93d9-34bf626c36e9" width=80% height=80%>

## Results
Evaluation on the test set with varying numbers of tabular features:

1 feature: Age

4 features: Sex, Age, Education, Genetic risk factor

9 features: Sex, Age, Education, Genetic risk factor, 3 cerebrospinal fluid biomarkers, 2 measures derived from PET scan

![image](https://github.com/daniel4725/HyperNetworks4imgNtabular/assets/95569050/0deb7649-fa06-4099-b5dc-7269d2cc35ff)
![image](https://github.com/daniel4725/HyperNetworks4imgNtabular/assets/95569050/1d9209d5-34c9-49f8-b33a-6a572966440c)
![image](https://github.com/daniel4725/HyperNetworks4imgNtabular/assets/95569050/2b051358-af15-4bfd-aa39-6e16f3eca14e)


