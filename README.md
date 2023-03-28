
**News:** WDFSR:Normaling Flow based on Wavelet-Domain for Super-Resolution
# WDFSR:Normaling Flow based on Wavelet-Domain for Super-Resolution
<br>
We mainly design our code based on the SRFlow code. If you want to see its base code, you can go to their source code.
<br>
This oneliner will:  

- Setup a python3 virtual env  
- Install the packages from `requirements.txt`  
- Download the pretrained models  
- Download the DIV2K validation data and other data.  
<br>
<br>

# Download models, data
#Here contains the training set and test set of DIV2K.
http://data.vision.ee.ethz.ch/alugmayr/SRFlow/datasets.zip

#For other test sets such as Urban, Set14, T91, Set5 and so on. You can prepare your own or download from github.
https://github.com/xinntao/BasicSR

#Since our encoders are pre-trained, you can download them from the links below (RRDBX4, RRDBX8) to reduce your preparation time.
http://data.vision.ee.ethz.ch/alugmayr/SRFlow/pretrained_models.zip


#We will first publish some of the pre-trained models in BaiduNet, and you can choose to use our pre-trained models or train the models yourself.

link：https://pan.baidu.com/s/128i059tT-dlgtPwj6brD9w  
password：mpdr

# Testing for SR: Apply the included pretrained models
#If you want to change the temperate, you can set heat in yml file.Meanwhile, you should modify model_path to the storage address of the model you want, such as WDFSR+.
```bash
source myenv/bin/activate                      # Use the env you created using setup.sh
cd code  
CUDA_VISIBLE_DEVICES=0 python test.py ./confs/test_WDFSR_4X.yml      # Diverse Images 4X (Dataset Included)  
CUDA_VISIBLE_DEVICES=0 python test.py ./confs/test_WDFSR_8X.yml      # Diverse Images 8X (Dataset Included)
```
For testing, we apply WDFSR to the full images on GPU with 20G of memory.

<br><br>


The following commands train the Super-Resolution network using Normalizing Flow in PyTorch:
```bash
source myenv/bin/activate                      # Use the env you created using setup.sh
cd code
python train.py -opt ./confs/WDFSR_4X.yml      # Diverse Images 4X (Dataset Included)
python train.py -opt ./confs/WDFSR+_4X.yml      # Diverse Images 4X (Dataset Included)
python train.py -opt ./confs/WDFSR++_4X.yml      # Diverse Images 4X (Dataset Included)
python train.py -opt ./confs/WDFSR_8X.yml      # Diverse Images 8X (Dataset Included)
python train.py -opt ./confs/WDFSR+_8X.yml      # Diverse Images 8X (Dataset Included)
python train.py -opt ./confs/WDFSR++_8X.yml      # Diverse Images 8X (Dataset Included)
```
- To reduce the GPU memory, reduce the batch size in the yml file.
# Low-light enhancement
For the Low-light enhancement task, we modify the up-sampling module in the encoder into a down-sampling module, and the other structures are similar. For the training and testing process, we only need to modify the training and testing sets. 
"code_ low_light" file contains the low light enhancement training and test code.
Our code is based on SRFLOW(https://github.com/andreas128/SRFlow)
