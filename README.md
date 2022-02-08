# BeeHave

This repository contains the board and schematic files (designed in Eagle) for the beehive monitoring system. It also contains a python file 'Script.py' and a numpy file containing the dataset with 1,250 days of beehive sensor data. The sample python script reads the numpy file and uses random forest to estimate the daily weight variations.

The AllFeatures_shuffled_144.npy file can be downloaded using the folowing link, and should be placed in the same folder as Script.py
https://media.githubusercontent.com/media/omaranwar85/BeeHave/main/AllFeatures_shuffled_144.npy

As this project progresses, we will be adding more to this repository.


The datasheets of used components are available onlie using the following links

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
