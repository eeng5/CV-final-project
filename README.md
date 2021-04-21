# CV-final-project
## How To set up the data ##
- [ ] go to https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=fer2013.tar.gz 
- [ ] create a gaggle account, agree to terms of use, and download icml_face_data.csv and train.csv
- [ ] move icml_face_data.csv and train.csv into the data folder in our repo
- [ ] activate the virtual environment then run 
```python3 run.py --generate-data```

## USE OUR VIRTUAL ENVIRONMENT ##
### To activate the virtual environment ###
``` cd code ```
``` source cs1430_env/bin/activate```
### To Run Our Live Emotion Detection App ####
```python3 run.py --live```

### To Train the Simple Model: ###
```python3 run.py --task 1 --aug 1```  -> less filtered images in augmentation step 
```python3 run.py --task 1 --aug 3```  -> more filters in augmentation step 

### To Train the Complex Model: ###
```python3 run.py --task 3 --aug 3```

### Other Run Commands: ###
Evaluate our Simple Model:

```python3 run.py --task 1 --load-checkpoint checkpoints/simple_model/<TIMESTAMP>/<MODEL_FILENAME>.h5 --evaluate```

Evaluate our Complex Model:

```python3 run.py --task 3 --load-checkpoint checkpoints/complex_model/<TIMESTAMP>/<MODEL_FILENAME>.h5 --evaluate```

View Epochs during Training:

```tensorboard --logdir logs```

# To Do #
- [x] Get Data in folder 
- [x] preprocess data and augment data
- [x] create CNN pipeline for predicting emotion 
- [x] Progress Report
- [x] Get tensorboard logs 
- [x] Connect pretrained model to live video 
- [x] Presentation
- [x] Project Report 
