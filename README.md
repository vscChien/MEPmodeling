# MEPmodeling
A Biological Model of Spinal and Peripheral Motor Pathways for TMS-induced MEPs

## Description
<p align="center">
  <a href="model_description.pdf">
    <img src="model_description.png" alt="Click to view PDF" width="400">
  </a>
</p>


## Matlab demo script
### Biological model
```matlab
subj       = 1;  % subject 1~10
withRC     = 1;  % 0: biological model without RC 
                 % 1: biological model with RC
AMPAweight = []; % []:free parameter range [0,1]
                 %    or a fixed value within [0,1]               
reRun      = 0;  % 0: Load fitted result and plot simulated MEP.  
                 % 1: Rerun model fitting. Backup previous fitted result
ga_MEPmodel_bio(subj,withRC,AMPAweight,reRun);
```

### Phenomenological model
```matlab
subj       = 1; % subject 1~10
reRun      = 0; % 0: Load fitted result and plot simulated MEP.  
                % 1: Rerun model fitting. Backup previous fitted result
ga_MEPmodel_pheno(subj,reRun);
```
