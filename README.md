# BeeHave

This repository contains the board and schematic files (designed in Eagle) for the beehive monitoring system, BeeDAS (Design and development of low-power, long-range data acquisition system for beehives). It also contains a python file 'Script.py' and a numpy file containing the dataset with 1,250 days of beehive sensor data and waether recordings. The sample python script reads the numpy file and uses random forest to estimate the daily weight variations. Following packages are required to run the script.

python                    3.7.11  

pandas                    1.3.2 

matplotlib                3.4.2 

numpy                     1.18.5   

scikit-learn              0.24.2  

seaborn                   0.11.2 



The AllFeatures_shuffled_144.npy file can be downloaded using the folowing link, and should be placed in the same folder as Script.py
https://media.githubusercontent.com/media/omaranwar85/BeeHave/main/AllFeatures_shuffled_144.npy



The datasheets of used components are available online, and can be accessed using the following links

CCS811 Gas Sensor
https://www.sciosense.com/wp-content/uploads/2020/01/SC-001232-DS-2-CCS811B-Datasheet-Revision-2.pdf

SanDisk Industrial
https://images-na.ssl-images-amazon.com/images/I/91tTtUMDM3L.pdf

MMA8452Q Accelerometer
https://www.nxp.com/docs/en/data-sheet/MMA8452Q.pdf

MKR NB-1500 Board
https://store-usa.arduino.cc/products/arduino-mkr-nb-1500

HX711 Load Amplifier
https://datasheetspdf.com/pdf/842201/Aviasemiconductor/HX711/1

BME280 Sensor
https://datasheetspdf.com/pdf/1096951/Bosch/BME280/1

ADMP401 MEMS microphone
https://www.analog.com/media/en/technical-documentation/obsolete-data-sheets/ADMP401.pdf

REF3433 Voltage Reference
https://www.ti.com/lit/ds/symlink/ref3433.pdf

Sparkfun Redboard Turbo
https://www.sparkfun.com/products/14812



As this project progresses, we will be adding more to this repository.

==========================================================================

One of our accepted works at AAAI22 workshop for 'Practical deep-learning in the Wild' also uses the data collected using this system, with a significant overlap with the dataset shared in this repository. You can access this paper using the following link
https://www.researchgate.net/publication/358954859_WE-Bee_Weight_Estimator_for_Beehives_Using_Deep_Learning

If you use this dataset, please refer one of our works.

@inproceedings{anwar2022we,
  title={WE-Bee: Weight Estimator for Beehives Using Deep Learning},
  author={Anwar, Omar and Keating, Adrian and Cardell-Oliver, Rachel and Datta, Amitava and Putrino, Gino},
  booktitle={AAAI Conference on Artificial Intelligence 2022: 1st International Workshop on Practical Deep Learning in the Wild},
  year={2022}
}

@article{anwar2022BeeDAS,
title = {Design and development of low-power, long-range data acquisition system for beehives - BeeDAS},
journal = {Computers and Electronics in Agriculture},
volume = {201},
pages = {107281},
year = {2022},
issn = {0168-1699},
doi = {https://doi.org/10.1016/j.compag.2022.107281},
url = {https://www.sciencedirect.com/science/article/pii/S0168169922005932},
author = {Omar Anwar and Adrian Keating and Rachel Cardell-Oliver and Amitava Datta and Gino Putrino},
keywords = {Data acquisition, Low-power, Long-range, Beehive monitoring, Weight estimation, Random forest},
abstract = {Decision making capability of a system is highly dependent upon the quality and quantity of training data. Majority of beehive monitoring systems developed for research purposes are designed to collect data through a small set of sensors, and from locations with little geographic diversity. This hinders the development of a dataset that can be used to effectively train machine learning models. In this work, we explain the design and development of a multi-sensory, remote data acquisition system for beehives (BeeDAS), with focus on low-power consumption and long-range communication. We address design challenges associated with such systems and highlight the critical issues that need consideration. The proposed system enables collection of data from beehives at remote locations and harsh environment. Results of field deployments elucidate the effectiveness of various sensors which measure temperature, humidity, atmospheric pressure, CO2, acoustics, vibrations and the weight of a hive in hostile environment. This work also uses random forest regression to evaluate the feature importance of different sensors, environmental variables such as temperature, humidity, rain, wind speed as well as the information related to seasons, towards estimating the daily hive weight change, on a dataset comprised of 1,250 days of sensor recordings. We also evaluate the protocol designed for communication using Narrow Band Internet of Things (NB-IoT). The issues related to power optimization, sleep intervals and data storage in remote monitoring are also discussed.}
}
