# NTIRE2024-AIG-VQA

## Test
* The pretrained model located in https://drive.google.com/drive/folders/1c1POW5PYfyCfsXaBeQEfEsegDCTNDDpa?usp=sharing
* Download and put the pre-trained model in the root of project
* The test script implemented in the notebook file ``test.ipynb``
* Set the path of the file names and relevance prompts to test the test-dataset. 
```
df_valid = pd.read_excel("info_test.xlsx")
```
* Set the path of test-dataset in the CFG class in the video path :
```
class CFG:
    debug = False
    image_path = "<test dataset path>"
```