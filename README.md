# SISAFS-DIBS

- **Lines 14–116:** Define your datasets. Make sure to use your local file paths.  
- **Lines 177–233:** Gamma tuning (otherwise, use a predefined constant gamma).  
- **Lines 238–303:** DISAFS; the accuracy reported at line 292 corresponds to the DISAFS results.  
- **Lines 310–428:** SISAFS; the variable `best_limit` corresponds to *Ltotal* in the paper. If desired, you may assume a predefined constant for *Ltotal* instead of tuning a specific value (`best_limit`) for each dataset. The variable `combined-features` contains the set of selected features for the entire dataset.  
- **Lines 437–463:** Accuracy results using SISAFS-selected features.  
- **Lines 502–717:** DIBS implementation.  
- **Lines 720–751:** Accuracy metrics (accuracy, precision, recall, and F1-score) after applying DIBS.  
- **Lines 754 onward:** Code for generating figures and performing further analysis.  











# SISAFS-DIBS

Lines 14 - 116: Define your datasets. You must use your local address.

Lines 177 - 233: gamma tunning (other wise use a predefined constant gamma)

Lines 238- 303: DISAFS; The accuracy at line 292 reflects the accuracy of DISAFS

Lines 310 - 428: SISAFS; The variale "best_limit" corresponds to Ltotal in the paper. If desired, You can also assume a predefined constant for Ltotal and not tune a specific Ltotal (best_limit) for each dataset. The variable "combined-features" contains a set of selected features for the entire dataset

Lines 437 - 463: Accuracy with SISAFS features

Lines 502 - 717:DIBS

Lines 720 - 751:Accuracy metrics (accuracy, precision, recall, and F1-score) after DIBS

Lines after 754: These lines are used for generating figures and furthere analysis 
