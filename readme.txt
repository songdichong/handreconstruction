Prerun:
1. All the files should be in the root directory of Freihand (https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html).
File structure:
freihand/
│
├── training/
│   ├── mask/
│   │
│   └── rgb/
│
├── evaluation/
│   └── rgb/
│
├── demo.py
├── evaluate_joints.py
├── .... other python files
├── training_mano.json
├── evaluation_mano.json
├── .... other json files
2. MANOPTH (https://github.com/hassony2/manopth) is installed

Introduction:
1.	For the first part, model-based approach, train the model with train_MANO.py
2.	To evaluate the model-based approach, in demo.py, change the corresponding path of the state dictionary trained from step 1 and run.
3.	For the second part, model-free approach, train the model with train_joints.py and train_vertex.py
4. 	Change the path of the state dictionary from step 3 in train_further_joints.py and train_further_vertex.py 
5.	Run evaluate_modelfreeapproach.py to generate the auc curve
6.  If you want to use our trained model for results the model could be downloaded at https://drive.google.com/drive/folders/1Y4PvXCuGQGdT3fbWKbmUi-Iq-7gZHGwQ?usp=drive_link. 'Failure1' corresponds to the model-based approach; 'joints' is the model-free approach's joint model; 'vertex' is the model-free approach's veretx model.
