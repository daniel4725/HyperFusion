# HyperFusion: A Hypernetwork Approach to Multimodal Integration of Tabular and Medical Imaging Data for Predictive Modeling

<p align="center">
  <a href='https://arxiv.org/abs/2403.13319' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/arXiv-2403.13319-brightgreen' alt='arXiv'>
  </a>
  <a href='https://www.sciencedirect.com/science/article/pii/S1361841525000519' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/Media-2025.103503-red' alt='Media Journal'>
  </a>
</p>

## Intro
The integration of multi-modal data has emerged as a promising approach in various fields, enabling a more comprehensive understanding of 
complex phenomena by leveraging the complementary information from different sources. In the realm of medical research, the integration of 
multi-modal data has emerged as a powerful approach for enhancing our understanding of complex diseases and conditions. The fusion of different 
data types, such as tabular data (electronic health records - EHR) encompassing medical records and demographic information, together with 
high-resolution imaging modalities like MRI scans, has unlocked new avenues for comprehensive analysis and diagnosis.

In this work, we propose a novel approach that harnesses the power of hypernetworks to fuse tabular data and MRI brain scans.

## Methodology
<img src="https://github.com/daniel4725/HyperFusion/assets/95569050/bc6e8b2a-4103-403c-a16d-164ced34a4b4" width=90% height=90%>

Training a network, $\mathcal{H}$, to create the weights, $𝜃_\mathcal{H}$, of the main network, $\mathcal{P}_𝜃$. 
We use the tabular information as an input to the Hypernetwork ($T$) and the Primary network is an image processing CNN.

We demonstrate the versatility and efficacy of the proposed hypernetwork framework, named HyperFusion, through two distinct brain MRI analysis tasks: brain age prediction conditioned by the subject's sex and classification of subjects into Alzheimer's disease (AD), Mild Cognitive Impairment (MCI), and Cognitively Normal (CN) groups conditioned by their tabular data, which includes clinical measurements, as well as demographic and genetic information.

## Brain Age Prediction conditioned by sex
### Architecture + Results

<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/user-attachments/assets/11055daf-d12c-4585-aa39-bcfb5ddc2977" width="45%" height="45%">
  <img src="https://github.com/user-attachments/assets/3350bca5-75ba-4c97-bba3-3dc74ce19b1d" width=50% height=50%>
</div>

### AD classification
### Architecture + Results

<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/user-attachments/assets/defb9dfb-304b-4239-9ce5-7114b1e9b21e" width="37%" height="37%">
  <img src="https://github.com/daniel4725/HyperFusion/assets/95569050/085c5384-c335-4fa2-89e0-139552a514fb" width=70% height=70%>
</div>




