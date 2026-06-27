# Project on Residential Energy Consumption Modelling Using Socio-Economic and Geospatial Information

## Overview

This project focuses on generating realistic synthetic residential energy consumption profiles for distribution grid and energy system studies. The main objective is to create representative household load profiles when detailed household-level measurements are limited. The project combines measured Near Open meter data with GenAI-based synthetic profile generation methods and additional energy components such as heat pumps, storage heating, and photovoltaic (PV) systems.

The generated synthetic household profiles are aggregated to represent residential energy demand at grid level. These profiles are combined with other energy assets and compared against available measured grid data, including high-voltage substation measurements, RLM profiles from industrial consumers, PV parks, and other grid-connected components.

By comparing measured and synthetic residual loads, the project enables the evaluation of future residential energy scenarios and supports the optimization and planning of energy systems. The developed approach helps analyse the impact of increasing ev charging, renewable generation, and flexible loads on distribution grid operation.

---

## Project Structure

# General

- **src/pipeline/**  
  Contains the main workflow functions that combine different modelling components. The pipeline connects data preparation, synthetic profile generation, additional energy component modelling, aggregation, and evaluation steps to create complete energy scenarios.

- **src/data_processing/**  
  Contains functions for processing and preparing input data, including household load profiles, temperature data, PV generation data, and other energy-related datasets.

- **src/models/**  
  Contains the pre-trained Models used for synthetic profile generation, including the WGAN-based household load profile generation approach.

- **src/configs/**  
  Contains configuration files, model parameters, scenario settings, and mappings used throughout the project.

- **src/data_access/**  
  Contains functions for accessing measurement data from different sources, including local files and external data interfaces.

- **src/utils/**  
  Contains general utility functions, helper scripts, and execution functions used across different modules.

- **data/processed/**  
  Contains processed and intermediate datasets generated during preprocessing. These files are cached to avoid repeating computationally expensive processing steps.

- **geodatadata/**  
  Contains yaml metadata files for grid data (Geo data)

- **notebooks/**  
  Contains exploratory analysis, visualization, testing, and evaluation notebooks used during the development process.

## Installation

---------------
## Technical Terms - German

| Term | Description | German Translation |
|---|---|---|
| WGAN | Wasserstein Generative Adversarial Network used to generate realistic synthetic household load profiles. | Wasserstein Generative Adversarial Network |
| storage_heating | Electrically powered heating system that stores thermal energy, typically with controlled charging periods. | Nachtspeicherheizung |
| photovoltaic (PV) | Renewable energy generation system based on solar power production. | Photovoltaik (PV) |
| high_voltage_station (UW) | Grid measurement point representing aggregated electricity demand at the high-voltage level, typically connecting transmission and distribution networks. | Umspannwerk (UW) |
| RLM_profile | Registered load measurement profile with high-resolution energy consumption data, typically used for large consumers such as industry and commercial customers. | Registrierende Leistungsmessung (RLM) |
| medium_voltage_grid (MS) | Part of the distribution grid that transfers electricity from substations to local transformer stations and larger consumers. | Mittelspannungsnetz (MS) |
| low_voltage_grid (NS) | Part of the distribution grid that supplies electricity directly to residential customers and small consumers. | Niederspannungsnetz (NS) |
| transformer | Electrical device that converts voltage levels between high voltage, medium voltage, and low voltage networks. | Transformator / Netztransformator |

## References

### Synthetic Load Profile Generation / WGAN
- Fraunhofer IEE - Sylaski Project  
  WGAN-based synthetic load profile generation approach and implementation:  
  https://github.com/FraunhoferIEE/sylaski
  
